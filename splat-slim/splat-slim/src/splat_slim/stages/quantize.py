"""Stage 4 - mixed-precision quantization.

Default scheme: FP16 for geometry (position, scale, rotation) and INT8 for
appearance (SH coefficients). Mixed-precision PSNR is empirically identical to
full FP16 across SH levels, so the appearance INT8 is effectively free quality.

IMPORTANT - INT8 method matters:
Naive per-column (per-field) min/max INT8 quantization collapses PSNR to ~14 dB.
That is a real, scene-independent failure of *that method*, not of INT8 itself.
Per-channel / sub-group quantization (many small ranges instead of one global
range per column) recovers quality. ``int8_subgroup`` is the slot for that;
keep both so the README can show the contrast (naive INT8 catastrophe vs
sub-group INT8 recovery).
"""

from __future__ import annotations

import numpy as np

GEOMETRY_FIELDS = ("x", "y", "z", "scale_0", "scale_1", "scale_2",
                   "rot_0", "rot_1", "rot_2", "rot_3")


def to_fp16(arr: np.ndarray) -> np.ndarray:
    return arr.astype(np.float16)


def int8_percolumn(arr: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Naive per-column min/max INT8 quant. Included to reproduce the failure case."""
    lo, hi = float(arr.min()), float(arr.max())
    scale = (hi - lo) / 255.0 if hi > lo else 1.0
    q = np.round((arr - lo) / scale).astype(np.uint8)
    return q, lo, scale


def int8_subgroup(arr: np.ndarray, n_groups: int = 1000):
    """Sub-group INT8: split values into groups, quantize each with its own range.

    This is the method that recovers PSNR. Port your validated grouping
    (e.g. ~1000 ranges) here and return per-group (lo, scale) alongside codes.
    """
    # TODO: implement contiguous/learned grouping + per-group min/max from your
    # working experiment, then return codes + per-group dequant params.
    raise NotImplementedError("Port your sub-group INT8 quantization here.")


def quantize_mixed(fields: dict[str, np.ndarray], int8: str = "subgroup") -> dict[str, np.ndarray]:
    """Apply FP16 to geometry and INT8 to appearance (f_dc_*, f_rest_*).

    Args:
        int8: "percolumn" (reproduces the catastrophe) or "subgroup" (recovers).
    """
    out: dict[str, np.ndarray] = {}
    for name, arr in fields.items():
        if name in GEOMETRY_FIELDS:
            out[name] = to_fp16(arr)
        else:
            out[name] = arr  # appearance handled by the chosen INT8 path
    # TODO: route f_dc_*/f_rest_* through int8_percolumn or int8_subgroup and
    # store dequant params so io.save_splat can persist them.
    return out
