"""Stage 1 - adaptive opacity pruning (paper section 3.1).

Removes near-transparent Gaussians. Opacities are stored as logits ``o``; the
cutoff is the 5th percentile of their sigmoid,

    tau = percentile_5(sigmoid(o)),

and every Gaussian with ``sigmoid(o) < tau`` is dropped, i.e. the lowest-opacity
5% of the scene (``percentile`` is configurable; the paper uses 5). Because the
threshold is computed per scene it adapts to splat density instead of using a
fixed cutoff such as splatfacto's training-time ``cull-alpha-thresh`` of 0.005.
The paper reports tau = 0.0105 (Bicycle), 0.0138 (Garden) and 0.0148 (Vase).

A Gaussian whose opacity equals tau exactly is kept.
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
