"""Stage 2 - spatial and scale outlier removal.

Removes floaters and over-large Gaussians (paper section 3.2):
  * spatial: drop points outside a percentile box on x/y/z (default 0.5-99.5);
  * scale: drop points whose log-scale falls outside [p1, min(p99, log(scale_cap))],
    where ``scale_cap`` (default 1.0, a *linear* size) hard-caps the upper bound to
    kill the giant blurry blobs that bloat file size and hurt PSNR.

Units: PLY / splatfacto store scales in LOG space. ``scale_cap`` is given in linear
units and is converted with ``log`` before being compared against the stored values,
so the default ``scale_cap=1.0`` is the paper's absolute bound ``log(1.0) = 0`` on the
stored log-scales. (v0.1.0 compared the stored log-scales against 1.0 directly, i.e. a
linear cap of e = 2.718. On our Garden and Bicycle PLYs the 99th-percentile
bound is between -1.9 and -2.4 in log space, far below either cap, so the two readings keep
the same Gaussians there; measured on our Garden/Bicycle PLYs, see examples/verify_subgroup.md.)

Both are *removal* masks (the Gaussian is dropped), not value clips.
"""

from __future__ import annotations

import math

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
        fields: splat dict (scales stored in log space, as in a PLY).
        spatial_low/high: percentile bounds for the x/y/z bounding box.
        scale_low/high: percentile bounds for the per-axis log-scale.
        scale_cap: hard upper bound on Gaussian size in LINEAR units, applied to the
            stored log-scales as ``log_scale <= log(scale_cap)`` (min'd with the
            p-high value). The default 1.0 is the paper's cap, log-scale 0.
            ``float("inf")`` disables the cap. Must be > 0.
    """
    if not scale_cap > 0.0:  # also rejects NaN
        raise ValueError(f"scale_cap is a linear size and must be > 0; got {scale_cap}")
    log_cap = math.log(scale_cap)

    positions = np.stack([fields["x"], fields["y"], fields["z"]], axis=-1)
    sp_lo = np.percentile(positions, spatial_low, axis=0)
    sp_hi = np.percentile(positions, spatial_high, axis=0)
    spatial_mask = np.all((positions >= sp_lo) & (positions <= sp_hi), axis=-1)

    log_scales = np.stack([fields["scale_0"], fields["scale_1"], fields["scale_2"]], axis=-1)
    sc_lo = np.percentile(log_scales, scale_low, axis=0)
    sc_hi = np.minimum(np.percentile(log_scales, scale_high, axis=0), log_cap)
    scale_mask = np.all((log_scales >= sc_lo) & (log_scales <= sc_hi), axis=-1)

    keep = spatial_mask & scale_mask
    return {name: arr[keep] for name, arr in fields.items()}
