"""Read/write Gaussian-splat PLY files.

A Gaussian splat PLY stores, per vertex: position (x,y,z), optional normals,
spherical-harmonic coefficients (f_dc_*, f_rest_*), opacity, scale (scale_*),
and rotation quaternion (rot_*). This module loads those into a plain dict of
numpy arrays so each stage can operate on them without nerfstudio internals.

Quantized output uses smaller dtypes: fp16 stored as uint16 (plyfile has no
'f2' dtype) and int8 stored as uint8 with per-field (min, max) saved in PLY
comments as `quant_minmax <field> <min> <max>` so a reader can dequantize.

Sub-group INT8 output (``--quant int8-subgroup``) is two files: ``<name>.ply``
(all fields as uint8, in Morton order) and a sidecar ``<name>.meta.npz`` with the
per-group ranges. The ranges are NOT written as PLY comments: 62 fields x G=1000
groups would be 62,000 comment lines (a text header of a few MB), versus ~0.5 MB
as a binary float32 table. The PLY header only carries three
informational comments (``quant_subgroup version/groups/sidecar``); a reader
locates the sidecar by name (``sidecar_path``), never by following the comment.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

from .stages.quantize import SubgroupMeta, dequantize_subgroup

SIDECAR_SUFFIX = ".meta.npz"
SUBGROUP_FORMAT_VERSION = 1


def load_splat(path: str | Path) -> dict[str, np.ndarray]:
    """Load a Gaussian-splat PLY into ``{field_name: np.ndarray}`` (float32)."""
    ply = PlyData.read(str(path))
    verts = ply["vertex"].data
    return {name: np.asarray(verts[name]).astype(np.float32) for name in verts.dtype.names}


def save_splat(path: str | Path, fields: dict[str, np.ndarray]) -> None:
    """Write a fields dict to a binary PLY as float32 (or uint16 for fp16 fields)."""
    dtype, out = [], {}
    for name, arr in fields.items():
        arr = np.asarray(arr)
        if arr.dtype == np.float16:
            out[name] = arr.view(np.uint16)
            dtype.append((name, "u2"))
        else:
            out[name] = arr.astype(np.float32)
            dtype.append((name, "f4"))
    n = len(next(iter(out.values())))
    structured = np.empty(n, dtype=dtype)
    for name, values in out.items():
        structured[name] = values
    PlyData([PlyElement.describe(structured, "vertex")], text=False).write(str(path))


def save_quantized(path: str | Path, structured: np.ndarray, comments: list[str]) -> None:
    """Write a pre-quantized structured array (mixed u1/u2 dtypes) with minmax comments."""
    PlyData([PlyElement.describe(structured, "vertex")], text=False,
            comments=comments).write(str(path))


def sidecar_path(ply_path: str | Path) -> Path:
    """Sidecar for a sub-group PLY: ``scene.ply`` -> ``scene.meta.npz`` (same folder)."""
    return Path(ply_path).with_suffix(SIDECAR_SUFFIX)


def save_subgroup(path: str | Path, structured: np.ndarray, meta: SubgroupMeta) -> Path:
    """Write a sub-group INT8 model: the uint8 PLY plus its ``.meta.npz`` sidecar.

    Returns the sidecar path. Keep the two files together; the PLY alone cannot be
    dequantized.
    """
    side = sidecar_path(path)
    comments = [
        f"quant_subgroup version {SUBGROUP_FORMAT_VERSION}",
        f"quant_subgroup groups {meta.n_groups}",
        f"quant_subgroup sidecar {side.name}",
    ]
    PlyData([PlyElement.describe(structured, "vertex")], text=False,
            comments=comments).write(str(path))
    np.savez_compressed(
        side,
        version=np.int64(SUBGROUP_FORMAT_VERSION),
        n=np.int64(meta.n),
        n_groups=np.int64(meta.n_groups),
        fields=np.array(meta.fields),
        ranges=meta.ranges,
    )
    return side


def load_subgroup(path: str | Path) -> dict[str, np.ndarray]:
    """Read a sub-group INT8 PLY (plus its sidecar) back into float32 fields.

    The fields come back in the stored (Morton) order. Raises ``FileNotFoundError``
    if the sidecar is missing and ``ValueError`` if the two files do not belong together.
    """
    side = sidecar_path(path)
    if not side.exists():
        raise FileNotFoundError(f"sidecar not found: {side} (a sub-group PLY needs it to dequantize)")
    structured = PlyData.read(str(path))["vertex"].data
    with np.load(side, allow_pickle=False) as z:
        version = int(z["version"])
        if version != SUBGROUP_FORMAT_VERSION:
            raise ValueError(f"unsupported sub-group format version {version} in {side}")
        meta = SubgroupMeta(
            n=int(z["n"]),
            n_groups=int(z["n_groups"]),
            fields=tuple(str(f) for f in z["fields"]),
            ranges=np.asarray(z["ranges"], dtype=np.float32),
        )
    if meta.ranges.shape != (len(meta.fields), meta.n_groups, 2):
        raise ValueError(f"malformed range table in {side}: shape {meta.ranges.shape}")
    return dequantize_subgroup(structured, meta)


def filesize_mb(path: str | Path) -> float:
    """File size in megabytes (1 MB = 1e6 bytes, matching the notebook's reporting)."""
    return Path(path).stat().st_size / 1e6
