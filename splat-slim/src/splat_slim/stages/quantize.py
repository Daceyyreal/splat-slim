"""Stage 4 - quantization.

Three modes (ported from the validated pipeline):
  * "fp16"  - every field to FP16 (uint16 storage).
  * "int8"  - every field to per-field min/max INT8 (uint8 storage).
  * "mixed" - FP16 geometry + INT8 appearance (the recommended Pareto choice).

IMPORTANT - INT8 method matters:
The "int8"/"mixed" path here uses *naive per-field* min/max ranges. Naive
per-field INT8 on geometry collapses PSNR to ~14 dB (a real, scene-independent
failure of this method, not of INT8 itself). Mixed precision avoids that by
keeping geometry in FP16 and only quantizing appearance, where it's near-free.

Per-field min/max is also the contrast point for future sub-group INT8: many
small ranges instead of one global range per field. ``int8_subgroup`` is the
slot for that; keeping both lets the paper show naive-INT8 catastrophe vs
sub-group recovery.
"""

from __future__ import annotations

import numpy as np

GEOMETRY_FIELDS = frozenset(
    {"x", "y", "z", "scale_0", "scale_1", "scale_2",
     "rot_0", "rot_1", "rot_2", "rot_3"}
)


def quantize(fields: dict[str, np.ndarray], mode: str = "mixed") -> tuple[np.ndarray, list[str]]:
    """Quantize a fields dict.

    Returns a structured numpy array (mixed u1/u2 dtypes) ready for
    ``io.save_quantized``, plus the ``quant_minmax`` PLY comments needed to
    dequantize the int8 fields.
    """
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


def int8_subgroup(arr: np.ndarray, n_groups: int = 1000):
    """Sub-group INT8: split values into groups, each with its own range.

    The method that recovers the PSNR naive per-field INT8 destroys. Not yet
    implemented - this is the planned contribution differentiating from naive
    INT8. Port the grouping + per-group min/max here when built.
    """
    raise NotImplementedError("Sub-group INT8 quantization is on the roadmap.")
