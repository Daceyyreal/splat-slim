"""Read/write Gaussian-splat PLY files.

A Gaussian splat PLY stores, per vertex: position (x,y,z), optional normals,
spherical-harmonic coefficients (f_dc_*, f_rest_*), opacity, scale (scale_*),
and rotation quaternion (rot_*). This module loads those into a plain dict of
numpy arrays so each stage can operate on them without nerfstudio internals.

Quantized output uses smaller dtypes: fp16 stored as uint16 (plyfile has no
'f2' dtype) and int8 stored as uint8 with per-field (min, max) saved in PLY
comments as `quant_minmax <field> <min> <max>` so a reader can dequantize.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement


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
    for name in out:
        structured[name] = out[name]
    PlyData([PlyElement.describe(structured, "vertex")], text=False).write(str(path))


def save_quantized(path: str | Path, structured: np.ndarray, comments: list[str]) -> None:
    """Write a pre-quantized structured array (mixed u1/u2 dtypes) with minmax comments."""
    PlyData([PlyElement.describe(structured, "vertex")], text=False,
            comments=comments).write(str(path))


def filesize_mb(path: str | Path) -> float:
    """File size in megabytes (1 MB = 1e6 bytes, matching the notebook's reporting)."""
    return Path(path).stat().st_size / 1e6
