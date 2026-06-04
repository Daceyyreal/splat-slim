"""Stage 3 - spherical-harmonic (SH) degree reduction.

SH coefficients dominate splat file size. Dropping from degree 3 -> 2 -> 1
trims the f_rest_* coefficients while keeping f_dc_* (the base color).

The subtlety is channel-blocked slicing: f_rest is stored interleaved as
[coeff x RGB], so naively truncating the flat f_rest array corrupts color.
The correct reduction keeps, per RGB channel, only the coefficients belonging
to SH degrees <= target. The coefficient counts per degree are:
    deg 0: 1 (this is f_dc),  deg 1: +3,  deg 2: +5,  deg 3: +7
so f_rest holds 3 + 5 + 7 = 15 coeffs per channel at degree 3.

Fill in ``reduce_sh`` with your validated channel-blocked indexing.
"""

from __future__ import annotations

import numpy as np

# Number of f_rest coefficients PER CHANNEL kept for each target degree.
_REST_PER_CHANNEL = {1: 3, 2: 8, 3: 15}


def reduce_sh(fields: dict[str, np.ndarray], target_degree: int) -> dict[str, np.ndarray]:
    """Reduce SH to ``target_degree`` (1, 2, or 3).

    degree 3 = no-op. Lower degrees drop the appropriate f_rest_* fields,
    sliced per RGB channel (not as a flat block).
    """
    if target_degree == 3:
        return dict(fields)
    if target_degree not in _REST_PER_CHANNEL:
        raise ValueError(f"target_degree must be 1, 2, or 3; got {target_degree}")

    # TODO: replace with your validated channel-blocked slice. The list of
    # f_rest_* field names must be regrouped by channel before truncation so
    # that the kept indices correspond to SH degrees <= target_degree.
    raise NotImplementedError(
        "Port the channel-blocked f_rest slicing from your working pipeline here."
    )
