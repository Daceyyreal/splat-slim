"""Smoke tests: imports, CLI wiring, and pure-numpy stage logic."""

import numpy as np

from splat_slim.stages import outliers, prune, quantize, sh


def _toy_splat(n=200):
    rng = np.random.default_rng(0)
    f = {"x": rng.normal(0, 1, n), "y": rng.normal(0, 1, n), "z": rng.normal(0, 1, n),
         "f_dc_0": rng.normal(0, 1, n), "f_dc_1": rng.normal(0, 1, n), "f_dc_2": rng.normal(0, 1, n),
         "opacity": rng.normal(0, 3, n),
         "scale_0": rng.normal(-3, 1, n), "scale_1": rng.normal(-3, 1, n), "scale_2": rng.normal(-3, 1, n),
         "rot_0": rng.normal(0, 1, n), "rot_1": rng.normal(0, 1, n),
         "rot_2": rng.normal(0, 1, n), "rot_3": rng.normal(0, 1, n)}
    for i in range(45):
        f[f"f_rest_{i}"] = rng.normal(0, 1, n)
    return {k: v.astype(np.float32) for k, v in f.items()}


def test_import_cli():
    from splat_slim.cli import app  # noqa: F401


def test_adaptive_threshold_in_range():
    thr = prune.adaptive_threshold(np.linspace(-5, 5, 1000), percentile=5.0)
    assert 0.0 < thr < 1.0


def test_prune_removes_low_opacity():
    out = prune.prune_opacity(_toy_splat(), percentile=5.0)
    assert len(out["opacity"]) < 200


def test_outliers_keep_subset():
    out = outliers.remove_outliers(_toy_splat(), scale_cap=1.0)
    assert 0 < len(out["x"]) <= 200


def test_reduce_sh_counts():
    f = _toy_splat()
    assert sum(k.startswith("f_rest_") for k in sh.reduce_sh(f, 3)) == 45
    assert sum(k.startswith("f_rest_") for k in sh.reduce_sh(f, 2)) == 24
    assert sum(k.startswith("f_rest_") for k in sh.reduce_sh(f, 1)) == 9


def test_quantize_mixed_dtypes():
    structured, comments = quantize.quantize(_toy_splat(), "mixed")
    assert structured.dtype["x"].str.endswith("u2")        # geometry -> fp16
    assert structured.dtype["f_dc_0"].str.endswith("u1")   # appearance -> int8
    assert any(c.startswith("quant_minmax f_dc_0") for c in comments)


def test_quantize_rejects_bad_mode():
    import pytest
    with pytest.raises(ValueError):
        quantize.quantize(_toy_splat(), "bogus")
