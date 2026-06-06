"""Stage 3 - spherical-harmonic (SH) degree reduction.

SH coefficients dominate splat file size. f_rest is stored channel-blocked:
for SH degree 3 there are 15 coefficients per RGB channel laid out as
[ch0: 0..14][ch1: 15..29][ch2: 30..44]. Reducing to a lower degree keeps,
per channel, only the first ``coeffs_per_channel`` coefficients (those
belonging to SH degrees <= target). Naively truncating the flat f_rest array
would mix channels and corrupt color, which is why the indexing is per channel.

Coefficients kept per channel: deg 1 -> 3, deg 2 -> 8, deg 3 -> 15 (no-op).
"""

from __future__ import annotations

import numpy as np

_COEFFS_PER_CHANNEL = {1: 3, 2: 8, 3: 15}

_GEOMETRY_ORDER = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
_TAIL_ORDER = ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]


def _rest_count(fields: dict[str, np.ndarray]) -> int:
    return sum(1 for k in fields if k.startswith("f_rest_"))


def reduce_sh(fields: dict[str, np.ndarray], target_degree: int) -> dict[str, np.ndarray]:
    """Reduce SH to ``target_degree`` (1, 2, or 3); degree 3 is a no-op."""
    if target_degree == 3:
        return dict(fields)
    if target_degree not in _COEFFS_PER_CHANNEL:
        raise ValueError(f"target_degree must be 1, 2, or 3; got {target_degree}")

    per_channel = _rest_count(fields) // 3          # 15 for a standard SH3 model
    coeffs = _COEFFS_PER_CHANNEL[target_degree]
    kept = [c * per_channel + k for c in range(3) for k in range(coeffs)]

    out: dict[str, np.ndarray] = {}
    for f in _GEOMETRY_ORDER:
        if f in fields:
            out[f] = fields[f]
    for new_i, src_i in enumerate(kept):
        out[f"f_rest_{new_i}"] = fields[f"f_rest_{src_i}"]
    for f in _TAIL_ORDER:
        if f in fields:
            out[f] = fields[f]
    return out
