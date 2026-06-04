"""Read/write Gaussian-splat PLY files.

A Gaussian splat PLY stores, per vertex: position (x,y,z), normals (unused),
spherical-harmonic coefficients (f_dc_*, f_rest_*), opacity, scale (scale_*),
and rotation quaternion (rot_*). This module loads those into a plain dict of
numpy arrays so each pipeline stage can operate on them without depending on
nerfstudio internals.

Note: plyfile has no native float16 ('f2') dtype. To store FP16 geometry we
reinterpret the FP16 bytes as uint16 via ``arr.view(np.uint16)`` on write and
``arr.view(np.float16)`` on read. The PLY header then declares those fields as
'ushort'. Consumers that don't know the convention will read raw uint16, so
keep this reader/writer paired.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement


def load_splat(path: str | Path) -> dict[str, np.ndarray]:
    """Load a Gaussian-splat PLY into ``{field_name: np.ndarray}``.

    Returns one array per scalar property, each of shape (N,).
    """
    ply = PlyData.read(str(path))
    verts = ply["vertex"].data
    return {name: np.asarray(verts[name]) for name in verts.dtype.names}


def save_splat(path: str | Path, fields: dict[str, np.ndarray]) -> None:
    """Write ``{field_name: np.ndarray}`` back to a binary PLY.

    float16 arrays are stored as uint16 (see module docstring).
    """
    out = {}
    dtype = []
    for name, arr in fields.items():
        arr = np.asarray(arr)
        if arr.dtype == np.float16:
            out[name] = arr.view(np.uint16)
            dtype.append((name, "u2"))
        else:
            out[name] = arr
            dtype.append((name, arr.dtype.str))
    n = len(next(iter(out.values())))
    structured = np.empty(n, dtype=dtype)
    for name in out:
        structured[name] = out[name]
    el = PlyElement.describe(structured, "vertex")
    PlyData([el], text=False).write(str(path))


def filesize_mb(path: str | Path) -> float:
    """Size of a file on disk in megabytes (1 MB = 1024*1024 bytes)."""
    return Path(path).stat().st_size / (1024 * 1024)
