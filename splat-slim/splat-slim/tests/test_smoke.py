"""Smoke tests: imports, CLI wiring, and pure-numpy stage logic."""

import numpy as np

from splat_slim.stages import outliers, prune


def test_import_cli():
    from splat_slim.cli import app  # noqa: F401


def test_adaptive_threshold_in_range():
    logits = np.linspace(-5, 5, 1000)
    thr = prune.adaptive_threshold(logits, percentile=5.0)
    assert 0.0 < thr < 1.0


def test_prune_removes_low_opacity():
    fields = {
        "x": np.zeros(100),
        "opacity": np.linspace(-10, 10, 100),
    }
    out = prune.prune_opacity(fields, percentile=5.0)
    assert len(out["opacity"]) < 100


def test_outliers_clip_scale():
    fields = {
        "x": np.zeros(10), "y": np.zeros(10), "z": np.zeros(10),
        "scale_0": np.full(10, 5.0), "scale_1": np.zeros(10), "scale_2": np.zeros(10),
    }
    out = outliers.remove_outliers(fields, scale_cap=1.0)
    assert np.all(out["scale_0"] <= np.log(1.0) + 1e-6)
