"""Sub-group INT8 (paper section 3.5): Morton order, grouping, round trip, sidecar I/O, CLI."""

import numpy as np
import pytest
from typer.testing import CliRunner

from splat_slim import io
from splat_slim.cli import app
from splat_slim.stages import outliers, prune, quantize
from splat_slim.stages.quantize import (
    dequantize_subgroup,
    group_starts,
    int8_subgroup,
    morton_order,
)

runner = CliRunner()
F32_EPS = float(np.finfo(np.float32).eps)


def _clustered_splat(n=4000, seed=1):
    """Tight clusters plus a few far-away floaters: the heavy-tailed shape that breaks one-range INT8."""
    rng = np.random.default_rng(seed)
    centers = rng.uniform(-5, 5, size=(20, 3))
    xyz = centers[rng.integers(0, 20, n)] + rng.normal(0, 0.05, (n, 3))
    xyz[:5] = rng.uniform(-500, 500, size=(5, 3))
    f = {"x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2], "opacity": rng.normal(0, 3, n)}
    for i in range(3):
        f[f"f_dc_{i}"] = rng.normal(0, 1, n)
        f[f"scale_{i}"] = rng.normal(-3, 0.5, n)
    for i in range(4):
        f[f"rot_{i}"] = rng.normal(0, 1, n)
    return {k: v.astype(np.float32) for k, v in f.items()}


# --- Morton ordering ---------------------------------------------------------


def _spread_ref(v, bits=21):
    return sum(((v >> i) & 1) << (3 * i) for i in range(bits))


def test_bit_spreading_matches_reference_interleave():
    rng = np.random.default_rng(0)
    xs, ys, zs = (rng.integers(0, 1 << 21, 300) for _ in range(3))
    got = (quantize._spread3(xs.astype(np.uint64))
           | (quantize._spread3(ys.astype(np.uint64)) << np.uint64(1))
           | (quantize._spread3(zs.astype(np.uint64)) << np.uint64(2)))
    want = [_spread_ref(int(x)) | (_spread_ref(int(y)) << 1) | (_spread_ref(int(z)) << 2)
            for x, y, z in zip(xs, ys, zs)]
    assert [int(c) for c in got] == want


def test_morton_order_walks_a_cube_in_z_order():
    # x is the least-significant axis, z the most: the order is lexicographic in (z, y, x).
    corners = [(1, 1, 1), (0, 0, 0), (0, 1, 0), (1, 0, 1), (1, 0, 0), (0, 1, 1), (1, 1, 0), (0, 0, 1)]
    f = {k: np.array([c[i] for c in corners], dtype=np.float32) for i, k in enumerate("xyz")}
    order = morton_order(f)
    walked = [corners[i] for i in order]
    assert walked == [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0),
                      (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1)]


def test_morton_order_is_stable_and_handles_a_flat_axis():
    f = {"x": np.zeros(6, np.float32), "y": np.zeros(6, np.float32), "z": np.zeros(6, np.float32)}
    assert list(morton_order(f)) == list(range(6))  # all identical -> input order kept


# --- grouping ---------------------------------------------------------------


@pytest.mark.parametrize("n,g", [(1000, 1000), (1001, 1000), (1999, 1000), (12345, 1000), (10, 3), (7, 7)])
def test_groups_are_balanced_and_non_empty(n, g):
    starts = group_starts(n, g)
    sizes = np.diff(np.append(starts, n))
    assert len(starts) == g and starts[0] == 0
    assert sizes.sum() == n and sizes.min() >= 1 and sizes.max() - sizes.min() <= 1


def test_more_groups_than_points_is_rejected_by_group_starts():
    with pytest.raises(ValueError):
        group_starts(5, 6)


# --- round trip ---------------------------------------------------------------


def test_roundtrip_error_is_bounded_by_half_a_step_per_group():
    fields = _clustered_splat()
    q, meta = int8_subgroup(fields, n_groups=100)
    assert meta.n == 4000 and meta.n_groups == 100 and meta.fields == tuple(fields)
    assert all(q.dtype[name] == np.uint8 for name in meta.fields)

    back = dequantize_subgroup(q, meta)
    assert tuple(back) == tuple(fields) and all(v.dtype == np.float32 for v in back.values())

    order = morton_order(fields)
    starts = group_starts(meta.n, meta.n_groups)
    sizes = np.diff(np.append(starts, meta.n))
    for i, name in enumerate(meta.fields):
        orig = fields[name][order].astype(np.float64)
        mn = meta.ranges[i, :, 0].astype(np.float64)
        mx = meta.ranges[i, :, 1].astype(np.float64)
        # the stored range is the true group min/max
        assert np.array_equal(mn, np.minimum.reduceat(orig, starts))
        assert np.array_equal(mx, np.maximum.reduceat(orig, starts))
        # half a quantization step per group, plus float32 rounding of the output
        slack = 4 * F32_EPS * np.maximum(np.abs(mn), np.abs(mx))
        bound = np.repeat((mx - mn) / 510.0 + slack, sizes)
        err = np.abs(back[name].astype(np.float64) - orig)
        assert (err <= bound).all(), f"{name}: worst error {float((err - bound).max())} over the bound"


def test_constant_group_is_reconstructed_exactly():
    fields = _clustered_splat(500)
    fields["opacity"] = np.full(500, 2.5, dtype=np.float32)
    q, meta = int8_subgroup(fields, 20)
    assert (dequantize_subgroup(q, meta)["opacity"] == 2.5).all()


def test_group_count_is_clamped_to_the_number_of_gaussians():
    fields = _clustered_splat(10)
    q, meta = int8_subgroup(fields, n_groups=1000)
    assert meta.n_groups == 10
    back = dequantize_subgroup(q, meta)  # one Gaussian per group: min == max, so exact
    order = morton_order(fields)
    for name in fields:
        assert np.array_equal(back[name], fields[name][order])


def test_subgroup_recovers_geometry_that_a_single_range_destroys():
    """The paper's point, in miniature: floaters stretch one global range; local ranges don't care."""
    fields = _clustered_splat()
    naive_q, comments = quantize.quantize(fields, "int8")
    lo, hi = next((float(c.split()[2]), float(c.split()[3])) for c in comments
                  if c.startswith("quant_minmax x "))
    naive_x = lo + naive_q["x"].astype(np.float64) / 255.0 * (hi - lo)
    naive_err = np.median(np.abs(naive_x - fields["x"]))

    q, meta = int8_subgroup(fields, n_groups=100)
    sub_x = dequantize_subgroup(q, meta)["x"].astype(np.float64)
    sub_err = np.median(np.abs(sub_x - fields["x"][morton_order(fields)]))
    assert sub_err < naive_err / 20


def test_rejects_nonfinite_values():
    fields = _clustered_splat(100)
    fields["f_dc_0"][3] = np.nan
    with pytest.raises(ValueError, match="f_dc_0"):
        int8_subgroup(fields, 10)


def test_rejects_missing_position_and_bad_group_count():
    fields = _clustered_splat(100)
    with pytest.raises(ValueError, match="'z'"):
        int8_subgroup({k: v for k, v in fields.items() if k != "z"}, 10)
    with pytest.raises(ValueError):
        int8_subgroup(fields, 0)


def test_dequantize_rejects_a_payload_that_does_not_match_the_metadata():
    q, meta = int8_subgroup(_clustered_splat(100), 10)
    with pytest.raises(ValueError):
        dequantize_subgroup(q[:50], meta)


def test_quantize_points_callers_at_int8_subgroup():
    with pytest.raises(ValueError, match="int8_subgroup"):
        quantize.quantize(_clustered_splat(50), "int8-subgroup")


# --- sidecar I/O ------------------------------------------------------------


def test_sidecar_roundtrip(tmp_path):
    q, meta = int8_subgroup(_clustered_splat(600), 30)
    ply = tmp_path / "scene.ply"
    side = io.save_subgroup(ply, q, meta)
    assert side == tmp_path / "scene.meta.npz" == io.sidecar_path(ply)
    assert b"comment quant_subgroup groups 30" in ply.read_bytes()[:4000]
    assert meta.nbytes == len(meta.fields) * 30 * 2 * 4  # float32 min/max per field per group

    back, ref = io.load_subgroup(ply), dequantize_subgroup(q, meta)
    assert all(np.array_equal(back[k], ref[k]) for k in ref)


def test_load_requires_the_sidecar(tmp_path):
    q, meta = int8_subgroup(_clustered_splat(100), 10)
    ply = tmp_path / "scene.ply"
    io.save_subgroup(ply, q, meta).unlink()
    with pytest.raises(FileNotFoundError, match="sidecar"):
        io.load_subgroup(ply)


def test_load_rejects_a_sidecar_from_a_different_model(tmp_path):
    qa, ma = int8_subgroup(_clustered_splat(600), 30)
    qb, mb = int8_subgroup(_clustered_splat(300, seed=2), 30)
    io.save_subgroup(tmp_path / "a.ply", qa, ma)
    io.save_subgroup(tmp_path / "b.ply", qb, mb)
    io.sidecar_path(tmp_path / "b.ply").replace(io.sidecar_path(tmp_path / "a.ply"))
    with pytest.raises(ValueError):
        io.load_subgroup(tmp_path / "a.ply")


# --- CLI --------------------------------------------------------------------


def test_cli_run_int8_subgroup_roundtrips_through_dequantize(tmp_path):
    src, out, restored = tmp_path / "in.ply", tmp_path / "slim.ply", tmp_path / "restored.ply"
    io.save_splat(src, _clustered_splat(1000))

    r = runner.invoke(app, ["run", str(src), str(out), "--quant", "int8-subgroup", "--groups", "40"])
    assert r.exit_code == 0, r.output
    assert out.exists() and (tmp_path / "slim.meta.npz").exists()

    r = runner.invoke(app, ["dequantize", str(out), str(restored)])
    assert r.exit_code == 0, r.output
    back = io.load_splat(restored)

    ref = outliers.remove_outliers(prune.prune_opacity(io.load_splat(src), 5.0), scale_cap=1.0)
    assert len(back["x"]) == len(ref["x"])  # same Gaussians as the plain pipeline
    atol = float(ref["x"].max() - ref["x"].min()) / 510.0 + 1e-4  # a group is never wider than the scene
    assert np.allclose(np.sort(back["x"]), np.sort(ref["x"]), atol=atol)


def test_cli_default_quant_is_still_mixed_without_a_sidecar(tmp_path):
    src, out = tmp_path / "in.ply", tmp_path / "slim.ply"
    io.save_splat(src, _clustered_splat(500))
    r = runner.invoke(app, ["run", str(src), str(out)])
    assert r.exit_code == 0, r.output
    assert out.exists() and not (tmp_path / "slim.meta.npz").exists()


def test_cli_rejects_an_unknown_quant_mode_instead_of_writing_float32(tmp_path):
    src, out = tmp_path / "in.ply", tmp_path / "slim.ply"
    io.save_splat(src, _clustered_splat(200))
    r = runner.invoke(app, ["run", str(src), str(out), "--quant", "int8_subgroup"])  # underscore typo
    assert r.exit_code != 0
    assert not out.exists()


def test_cli_dequantize_without_a_sidecar_fails_cleanly(tmp_path):
    q, meta = int8_subgroup(_clustered_splat(100), 10)
    ply = tmp_path / "scene.ply"
    io.save_subgroup(ply, q, meta).unlink()
    r = runner.invoke(app, ["dequantize", str(ply), str(tmp_path / "out.ply")])
    assert r.exit_code == 1
    assert "sidecar" in r.output
