"""Scale cap semantics (paper section 3.2): cap on stored LOG-scales at log(1.0) = 0."""

import inspect
import math
import re
from pathlib import Path

import numpy as np
import pytest

import splat_slim
from splat_slim.stages.outliers import remove_outliers


def _splat(n=1000, n_big=50, big_values=(0.5,)):
    """All Gaussians at one point (so the spatial box keeps everyone); the first
    ``n_big`` have large log-scales cycling through ``big_values``, the rest ~ -3."""
    rng = np.random.default_rng(0)
    scales = rng.normal(-3.0, 0.05, size=(n, 3))
    scales[:n_big] = np.resize(np.array(big_values), n_big)[:, None]
    f = {"x": np.zeros(n), "y": np.zeros(n), "z": np.zeros(n)}
    for i in range(3):
        f[f"scale_{i}"] = scales[:, i]
    return {k: v.astype(np.float32) for k, v in f.items()}


def test_default_cap_is_log_zero_as_in_the_paper():
    assert inspect.signature(remove_outliers).parameters["scale_cap"].default == 1.0  # linear 1.0
    out = remove_outliers(_splat())  # p99 ~ +0.5, so the cap is what binds
    assert out["scale_0"].max() <= 0.0
    assert len(out["scale_0"]) > 900  # the ordinary Gaussians are untouched


def test_cap_splits_exactly_at_log_scale_zero():
    # 25 Gaussians just under the cap and 25 just over it; p99 sits above both.
    f = _splat(n_big=50, big_values=(-0.001, 0.001))
    out = remove_outliers(f)
    just_under = (out["scale_0"] > -0.01) & (out["scale_0"] <= 0.0)
    assert just_under.sum() == 25
    assert (out["scale_0"] > 0.0).sum() == 0


def test_cap_is_a_linear_size_not_a_log_value():
    f = _splat()
    # linear e == log 1.0, which is how v0.1.0 read scale_cap=1.0: big splats now survive up to p99
    assert (remove_outliers(f, scale_cap=math.e)["scale_0"] > 0.0).any()
    assert (remove_outliers(f, scale_cap=1.0)["scale_0"] > 0.0).sum() == 0


def test_infinite_cap_disables_it():
    f = _splat()
    assert len(remove_outliers(f, scale_cap=float("inf"))["scale_0"]) == len(
        remove_outliers(f, scale_cap=math.e)["scale_0"])


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan")])
def test_cap_must_be_a_positive_size(bad):
    with pytest.raises(ValueError, match="scale_cap"):
        remove_outliers(_splat(), scale_cap=bad)


def test_package_and_pyproject_versions_agree():
    pyproject = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    declared = re.search(r'^version = "([^"]+)"', pyproject, re.MULTILINE).group(1)
    assert splat_slim.__version__ == declared
