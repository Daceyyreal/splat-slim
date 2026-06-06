"""Quality and size metrics for reporting Pareto points."""

from __future__ import annotations

import numpy as np


def psnr(rendered: np.ndarray, reference: np.ndarray, max_val: float = 1.0) -> float:
    """PSNR (dB) between two image arrays in [0, max_val]."""
    mse = float(np.mean((rendered.astype(np.float64) - reference.astype(np.float64)) ** 2))
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10((max_val ** 2) / mse)


def size_reduction(original_mb: float, optimized_mb: float) -> float:
    """Fraction of size removed, e.g. 0.54 for a 54% reduction."""
    return 1.0 - (optimized_mb / original_mb)
