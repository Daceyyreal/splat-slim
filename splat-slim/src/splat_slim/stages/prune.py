"""Stage 1 - adaptive opacity pruning.

Removes near-transparent Gaussians. The threshold is *adaptive*: it is the
Pth-percentile of the per-Gaussian sigmoid opacity (default 5th percentile),
so it self-calibrates to each scene rather than using a fixed cutoff. Across
scenes this lands at distinct values (~0.005 -> ~0.015), which is the empirical
evidence behind the "adaptive" claim.

Drop your validated pruning logic into ``prune_opacity`` below. The percentile
threshold computation is provided; wire it to your field layout.
"""

from __future__ import annotations

import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def adaptive_threshold(opacity_logits: np.ndarray, percentile: float = 5.0) -> float:
    """Return the Pth-percentile sigmoid opacity used as the prune cutoff."""
    return float(np.percentile(_sigmoid(opacity_logits), percentile))


def prune_opacity(fields: dict[str, np.ndarray], percentile: float = 5.0) -> dict[str, np.ndarray]:
    """Drop Gaussians whose sigmoid opacity is below the adaptive threshold.

    Args:
        fields: splat dict from ``io.load_splat``.
        percentile: percentile of sigmoid opacity to use as the cutoff.

    Returns:
        A new fields dict with low-opacity Gaussians removed.
    """
    thr = adaptive_threshold(fields["opacity"], percentile)
    keep = _sigmoid(fields["opacity"]) >= thr
    return {name: arr[keep] for name, arr in fields.items()}
