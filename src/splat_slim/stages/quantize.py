"""Stage 4 - quantization.

Four modes:
  * "fp16"         - every field to FP16 (stored as uint16).
  * "int8"         - every field to INT8 with ONE min/max range per field (uint8).
  * "mixed"        - FP16 geometry + INT8 appearance (recommended; see ``quantize``).
  * "int8-subgroup" - every field to INT8 with MANY local ranges (``int8_subgroup``).

Why the INT8 flavour matters
----------------------------
A single global range per field ("int8") collapses reconstruction quality to
~15 dB on every scene: a handful of outliers stretches [min, max], so the 256
levels are spread so thinly that the bulk of the values land on a few codes.
That is an artifact of the *range*, not of 8-bit storage. Giving each small
group of spatially-adjacent Gaussians its own range ("int8-subgroup") keeps
storage at 8 bits per value and recovers the quality (paper section 3.5).
Mixed precision sidesteps the problem by keeping geometry in FP16.

Sub-group INT8 (``int8_subgroup`` / ``dequantize_subgroup``)
-------------------------------------------------------------
1. Sort the Gaussians along a Morton (Z-order) curve over their positions, so
   neighbours in the array are neighbours in space.
2. Split the sorted array into ``G`` contiguous, near-equal groups
   (default ``G = 1000``; sizes differ by at most one Gaussian).
3. For every field and every group, store ``min`` and ``max`` (float32) and
   quantize the group's values to ``uint8`` against that range.

The output is in Morton order, not the input order (splat renderers sort by
depth, so the stored order is irrelevant to rendering). The group boundaries
are a pure function of ``(n, G)``, so a decoder needs only the per-group
ranges, which ``io.save_subgroup`` writes to a sidecar ``<name>.meta.npz``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

GEOMETRY_FIELDS = frozenset(
    {"x", "y", "z", "scale_0", "scale_1", "scale_2",
     "rot_0", "rot_1", "rot_2", "rot_3"}
)

DEFAULT_GROUPS = 1000


def quantize(fields: dict[str, np.ndarray], mode: str = "mixed") -> tuple[np.ndarray, list[str]]:
    """Quantize a fields dict with one range per field (fp16 / int8 / mixed).

    "mixed" stores x, y, z, scale_* and rot_* as FP16 and every other field
    (normals, SH colour, opacity) as INT8. For sub-group INT8 use
    :func:`int8_subgroup`, which returns different metadata.

    Returns a structured numpy array (mixed u1/u2 dtypes) ready for
    ``io.save_quantized``, plus the ``quant_minmax`` PLY comments needed to
    dequantize the int8 fields.
    """
    if mode == "int8-subgroup":
        raise ValueError("int8-subgroup returns per-group ranges; call int8_subgroup() instead")
    if mode not in ("fp16", "int8", "mixed"):
        raise ValueError(f"mode must be fp16/int8/mixed; got {mode}")

    n = len(next(iter(fields.values())))
    new_dtype: list[tuple[str, str]] = []
    int8_minmax: dict[str, tuple[float, float]] = {}

    for fname, arr in fields.items():
        is_fp16 = mode == "fp16" or (mode == "mixed" and fname in GEOMETRY_FIELDS)
        if is_fp16:
            new_dtype.append((fname, "u2"))
        else:
            orig = arr.astype(np.float32)
            int8_minmax[fname] = (float(orig.min()), float(orig.max()))
            new_dtype.append((fname, "u1"))

    out = np.empty(n, dtype=new_dtype)
    for fname, arr in fields.items():
        if dict(new_dtype)[fname] == "u2":
            out[fname] = arr.astype(np.float16).view(np.uint16)
        else:
            mn, mx = int8_minmax[fname]
            rng = max(mx - mn, 1e-9)
            out[fname] = np.round((arr.astype(np.float32) - mn) / rng * 255.0).clip(0, 255).astype(np.uint8)

    comments = [f"quant_minmax {f} {mn} {mx}" for f, (mn, mx) in int8_minmax.items()]
    return out, comments


# --------------------------------------------------------------------------- #
# Sub-group INT8
# --------------------------------------------------------------------------- #

_MORTON_BITS = 21  # 3 axes x 21 bits = 63 bits, fits in uint64


@dataclass(frozen=True)
class SubgroupMeta:
    """Everything besides the uint8 payload needed to dequantize a sub-group model.

    Attributes:
        n: number of Gaussians.
        n_groups: number of groups actually used (``min(requested G, n)``).
        fields: field names, in the order of the first axis of ``ranges``.
        ranges: float32 array ``[len(fields), n_groups, 2]``; ``[..., 0]`` is the
            group minimum and ``[..., 1]`` the group maximum.
    """

    n: int
    n_groups: int
    fields: tuple[str, ...]
    ranges: np.ndarray

    @property
    def nbytes(self) -> int:
        """Size of the per-group range table in bytes (float32 min/max)."""
        return int(self.ranges.nbytes)


def _spread3(v: np.ndarray) -> np.ndarray:
    """Insert two zero bits after each of the low 21 bits of ``v`` (uint64)."""
    u = np.uint64
    v = v & u(0x1FFFFF)
    v = (v | (v << u(32))) & u(0x1F00000000FFFF)
    v = (v | (v << u(16))) & u(0x1F0000FF0000FF)
    v = (v | (v << u(8))) & u(0x100F00F00F00F00F)
    v = (v | (v << u(4))) & u(0x10C30C30C30C30C3)
    v = (v | (v << u(2))) & u(0x1249249249249249)
    return v


def morton_codes(xyz: np.ndarray) -> np.ndarray:
    """63-bit Morton (Z-order) code of each point, over the points' own bounding box.

    Each axis is mapped onto ``2**21`` cells; bit ``i`` of x, y, z becomes bits
    ``3i``, ``3i+1``, ``3i+2`` of the code (x is the least significant axis).
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    lo = xyz.min(axis=0)
    span = xyz.max(axis=0) - lo
    span = np.where(span > 0.0, span, 1.0)  # a flat axis maps to cell 0
    top = (1 << _MORTON_BITS) - 1
    cells = np.minimum(((xyz - lo) / span * (1 << _MORTON_BITS)).astype(np.uint64), np.uint64(top))
    return (_spread3(cells[:, 0])
            | (_spread3(cells[:, 1]) << np.uint64(1))
            | (_spread3(cells[:, 2]) << np.uint64(2)))


def morton_order(fields: dict[str, np.ndarray]) -> np.ndarray:
    """Permutation that sorts the Gaussians along a Morton curve over (x, y, z).

    Stable, so Gaussians in the same cell keep their input order.
    """
    for axis in ("x", "y", "z"):
        if axis not in fields:
            raise ValueError(f"fields must contain '{axis}' to order Gaussians along a Morton curve")
        _require_finite(axis, fields[axis])
    xyz = np.stack([fields["x"], fields["y"], fields["z"]], axis=-1)
    return np.argsort(morton_codes(xyz), kind="stable")


def group_starts(n: int, n_groups: int) -> np.ndarray:
    """Start index of each of ``n_groups`` contiguous, near-equal groups over ``n`` items.

    Group ``k`` covers ``[floor(k*n/G), floor((k+1)*n/G))``: sizes differ by at most one
    and no group is empty (requires ``1 <= n_groups <= n``). Because this is a pure
    function of ``(n, G)``, a decoder can rebuild the boundaries without storing them.
    """
    if not 1 <= n_groups <= n:
        raise ValueError(f"need 1 <= n_groups <= n; got n_groups={n_groups}, n={n}")
    return (np.arange(n_groups, dtype=np.int64) * n) // n_groups


def _group_sizes(n: int, n_groups: int) -> np.ndarray:
    return np.diff(np.append(group_starts(n, n_groups), n))


def _require_finite(name: str, arr: np.ndarray) -> None:
    if not np.isfinite(arr).all():
        raise ValueError(f"field '{name}' contains NaN/inf; remove or fix those Gaussians first")


def int8_subgroup(
    fields: dict[str, np.ndarray], n_groups: int = DEFAULT_GROUPS
) -> tuple[np.ndarray, SubgroupMeta]:
    """Sub-group INT8: Morton-sort, split into ``n_groups`` groups, per-group min/max to uint8.

    Every field (geometry included) is quantized to ``uint8``; each group has its own
    ``[min, max]`` per field, so the 256 levels only have to span a small local range.
    For any value ``v`` in a group, the reconstruction error is at most
    ``(max - min) / 510`` of that group (half a quantization step), up to float32 rounding.

    Args:
        fields: splat dict from ``io.load_splat`` (needs x, y, z; all values finite).
        n_groups: requested group count G (paper: ~1000); clamped to the number of Gaussians.

    Returns:
        ``(structured, meta)``: a structured array of ``u1`` fields in **Morton order**
        (same field order as ``fields``), and the per-group ranges. Persist both with
        ``io.save_subgroup``; invert with :func:`dequantize_subgroup`.
    """
    if not fields:
        raise ValueError("fields is empty")
    n = len(next(iter(fields.values())))
    if n == 0:
        raise ValueError("cannot quantize an empty splat")
    if n_groups < 1:
        raise ValueError(f"n_groups must be >= 1; got {n_groups}")
    g = min(n_groups, n)

    order = morton_order(fields)
    starts = group_starts(n, g)
    sizes = _group_sizes(n, g)
    names = tuple(fields)

    out = np.empty(n, dtype=[(name, "u1") for name in names])
    ranges = np.empty((len(names), g, 2), dtype=np.float32)
    for i, name in enumerate(names):
        arr = np.asarray(fields[name], dtype=np.float32)
        _require_finite(name, arr)
        arr = arr[order]
        mn = np.minimum.reduceat(arr, starts)
        mx = np.maximum.reduceat(arr, starts)
        ranges[i, :, 0], ranges[i, :, 1] = mn, mx

        span = np.repeat((mx.astype(np.float64) - mn), sizes)
        offset = arr.astype(np.float64) - np.repeat(mn, sizes)
        q = np.divide(offset, span, out=np.zeros(n), where=span > 0.0) * 255.0
        out[name] = np.clip(np.rint(q), 0, 255).astype(np.uint8)
    return out, SubgroupMeta(n=n, n_groups=g, fields=names, ranges=ranges)


def dequantize_subgroup(structured: np.ndarray, meta: SubgroupMeta) -> dict[str, np.ndarray]:
    """Invert :func:`int8_subgroup`: uint8 payload + per-group ranges -> float32 fields.

    The result is in the stored (Morton) order. A group with ``min == max`` is
    reconstructed exactly.
    """
    if len(structured) != meta.n:
        raise ValueError(f"payload has {len(structured)} Gaussians but metadata says {meta.n}")
    if tuple(structured.dtype.names or ()) != meta.fields:
        raise ValueError("payload fields do not match metadata fields")
    sizes = _group_sizes(meta.n, meta.n_groups)
    out: dict[str, np.ndarray] = {}
    for i, name in enumerate(meta.fields):
        mn = meta.ranges[i, :, 0].astype(np.float64)
        span = meta.ranges[i, :, 1].astype(np.float64) - mn
        q = structured[name].astype(np.float64)
        out[name] = (np.repeat(mn, sizes) + q / 255.0 * np.repeat(span, sizes)).astype(np.float32)
    return out
