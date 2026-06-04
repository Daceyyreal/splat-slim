"""Stage 2 - spatial and scale outlier removal.

Removes floaters and over-large Gaussians:
  * spatial: drop points outside a percentile box on x/y/z (default 0.5-99.5);
  * scale: clip per-axis log-scale so exp(scale) never exceeds ``scale_cap``
    (default 1.0), which suppresses the giant blurry blobs that bloat size
    and hurt PSNR.

Replace the bodies with your validated clipping logic; the percentile masks
are provided as a starting point.
"""

from __future__ import annotations

import numpy as np


def remove_outliers(
    fields: dict[str, np.ndarray],
    spatial_low: float = 0.5,
    spatial_high: float = 99.5,
    scale_cap: float = 1.0,
) -> dict[str, np.ndarray]:
    """Clip spatial/scale outliers.

    Args:
        fields: splat dict.
        spatial_low/high: percentile bounds for the x/y/z bounding box.
        scale_cap: maximum allowed linear scale (applied to exp(scale_*)).
    """
    xyz = np.stack([fields["x"], fields["y"], fields["z"]], axis=1)
    lo = np.percentile(xyz, spatial_low, axis=0)
    hi = np.percentile(xyz, spatial_high, axis=0)
    keep = np.all((xyz >= lo) & (xyz <= hi), axis=1)
    out = {name: arr[keep] for name, arr in fields.items()}

    # Hard scale cap: scale_* are stored as log-scale, so cap in log space.
    log_cap = np.log(scale_cap)
    for k in ("scale_0", "scale_1", "scale_2"):
        if k in out:
            out[k] = np.minimum(out[k], log_cap)
    return out
