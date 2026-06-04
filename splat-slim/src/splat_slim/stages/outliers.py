"""Stage 2 - spatial and scale outlier removal.

Removes floaters and over-large Gaussians (ported from the validated pipeline):
  * spatial: drop points outside a percentile box on x/y/z (default 0.5-99.5);
  * scale: drop points whose scale falls outside [p1, min(p99, scale_cap)],
    where scale_cap (default 1.0) hard-caps the upper bound to kill the giant
    blurry blobs that bloat file size and hurt PSNR.

Both are *removal* masks (the Gaussian is dropped), not value clips.
"""

from __future__ import annotations

import numpy as np


def remove_outliers(
    fields: dict[str, np.ndarray],
    spatial_low: float = 0.5,
    spatial_high: float = 99.5,
    scale_low: float = 1.0,
    scale_high: float = 99.0,
    scale_cap: float = 1.0,
) -> dict[str, np.ndarray]:
    """Drop spatial and scale outliers.

    Args:
        fields: splat dict.
        spatial_low/high: percentile bounds for the x/y/z bounding box.
        scale_low/high: percentile bounds for per-axis scale.
        scale_cap: hard upper bound on scale (min'd with the p-high value).
    """
    positions = np.stack([fields["x"], fields["y"], fields["z"]], axis=-1)
    sp_lo = np.percentile(positions, spatial_low, axis=0)
    sp_hi = np.percentile(positions, spatial_high, axis=0)
    spatial_mask = np.all((positions >= sp_lo) & (positions <= sp_hi), axis=-1)

    scales = np.stack([fields["scale_0"], fields["scale_1"], fields["scale_2"]], axis=-1)
    sc_lo = np.percentile(scales, scale_low, axis=0)
    sc_hi = np.minimum(np.percentile(scales, scale_high, axis=0), scale_cap)
    scale_mask = np.all((scales >= sc_lo) & (scales <= sc_hi), axis=-1)

    keep = spatial_mask & scale_mask
    return {name: arr[keep] for name, arr in fields.items()}
