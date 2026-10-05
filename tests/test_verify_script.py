"""examples/verify_subgroup.py runs end to end and reports what the docs say it reports."""

import importlib.util
import re
from pathlib import Path

import numpy as np

from splat_slim import io

SCRIPT = Path(__file__).resolve().parents[1] / "examples" / "verify_subgroup.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("verify_subgroup", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _scene(n=3000, seed=3):
    """Clustered positions plus far floaters, so one-range INT8 is visibly worse."""
    rng = np.random.default_rng(seed)
    centers = rng.uniform(-5, 5, size=(20, 3))
    xyz = centers[rng.integers(0, 20, n)] + rng.normal(0, 0.05, (n, 3))
    xyz[:5] = rng.uniform(-300, 300, size=(5, 3))
    f = {"x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2], "opacity": rng.normal(0, 3, n)}
    for i in range(3):
        f[f"scale_{i}"] = rng.normal(-3, 0.5, n)
        f[f"f_dc_{i}"] = rng.normal(0, 1, n)
    return {k: v.astype(np.float32) for k, v in f.items()}


def test_script_reports_the_three_checks(tmp_path, capsys):
    ply = tmp_path / "toy.ply"
    io.save_splat(ply, _scene())
    code = _load_script().main([str(ply), "--groups", "40"])
    out = capsys.readouterr().out
    assert code == 0
    assert "values over (max-min)/510     : 0" in out
    assert "[2] mean position error" in out and "[3] scale cap" in out
    assert "cap removes anything on its own     : False" in out
    ratio = float(re.search(r"pooled .*: ([0-9.]+)x", out).group(1))
    assert ratio > 2.0  # sub-group ranges beat one range per field on floater-stretched data


def test_script_rejects_a_ply_without_the_required_fields(tmp_path, capsys):
    ply = tmp_path / "bad.ply"
    io.save_splat(ply, {k: v for k, v in _scene(200).items() if k != "opacity"})
    assert _load_script().main([str(ply)]) == 2
    assert "opacity" in capsys.readouterr().err


def _size_rows(out):
    return {m.group(1): int(m.group(2).replace(",", ""))
            for m in re.finditer(r"^\s+(fp16|mixed|int8-subgroup|int8)\s+([\d,]+)\s", out, re.MULTILINE)}


def test_sizes_section_runs_the_cli_in_every_mode_and_cleans_up(tmp_path, capsys):
    ply, work = tmp_path / "toy.ply", tmp_path / "work"
    work.mkdir()
    io.save_splat(ply, _scene())
    code = _load_script().main([str(ply), "--groups", "40", "--sizes", "--tmp-dir", str(work)])
    out = capsys.readouterr().out
    assert code == 0
    sizes = _size_rows(out)
    assert set(sizes) == {"fp16", "mixed", "int8", "int8-subgroup"}
    assert sizes["fp16"] > sizes["mixed"] > sizes["int8"]  # 2 B, mixed, 1 B per value
    assert "no paper column" in out
    assert list(work.iterdir()) == []  # every temporary output was deleted


def test_sizes_flag_a_mismatch_with_the_paper_without_failing(tmp_path, capsys):
    ply = tmp_path / "toy.ply"
    io.save_splat(ply, _scene())
    mod = _load_script()
    # toy files are kilobytes; give the "paper" values that cannot match so the flagging path runs
    mod.PAPER = {"garden": {"n0": 1, "pruned": 1.0, "fp16": 50.0, "mixed": 50.0, "int8": 50.0,
                            "int8-subgroup": 50.0}}
    code = mod.main([str(ply), "--groups", "40", "--sizes", "--paper", "garden"])
    out = capsys.readouterr().out
    assert code == 0  # the size comparison is informational
    assert out.count("DIFFERS") == 4
    assert "Nothing is hard-coded to pass or fail on it" in out
    assert "paper column: garden" in out
