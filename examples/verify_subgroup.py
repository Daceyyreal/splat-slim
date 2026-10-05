#!/usr/bin/env python
"""Re-check the splat-slim sub-group INT8 and scale-cap claims on a Gaussian-splat PLY.

    python examples/verify_subgroup.py scene.ply --groups 1000

Runs the same stages as ``splat-slim run`` (prune -> clean -> reduce-SH) and then, with the
package's own code:

  1. sub-group INT8 error bound: for every field and every group, the reconstruction error
     of every value is at most half a quantization step, ``(max - min) / 510``, of that group;
  2. mean position error (x, y, z), one range per field vs sub-group ranges, as a ratio;
  3. the scale cap: per-axis 99th-percentile log-scale (what ``clean`` compares against
     ``log(scale_cap)``), and whether the cap removes anything the percentile bound does not.

Exit status is 1 if the error bound is violated, 0 otherwise. Needs nothing beyond the
package's own dependencies.
"""

from __future__ import annotations

import argparse
import inspect
import math
import sys
import time

import numpy as np

from splat_slim import io
from splat_slim.stages import outliers, prune, quantize, sh

F32_EPS = float(np.finfo(np.float32).eps)
REQUIRED = ("x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2")


def check_subgroup(fields, groups):
    """Quantize with ``int8_subgroup`` and check bound + position error, field by field.

    Fields are dequantized one at a time (through the library's ``dequantize_subgroup``),
    which keeps memory low on multi-million-Gaussian scenes.
    """
    q, meta = quantize.int8_subgroup(fields, groups)
    order = quantize.morton_order(fields)
    starts = quantize.group_starts(meta.n, meta.n_groups)
    sizes = np.diff(np.append(starts, meta.n))

    violations, worst, constant_groups = 0, 0.0, 0
    sub_err = {}
    for i, name in enumerate(meta.fields):
        one = quantize.SubgroupMeta(n=meta.n, n_groups=meta.n_groups, fields=(name,),
                                    ranges=meta.ranges[i:i + 1])
        back = quantize.dequantize_subgroup(q[[name]], one)[name].astype(np.float64)
        orig = np.asarray(fields[name], dtype=np.float32)[order].astype(np.float64)
        mn = meta.ranges[i, :, 0].astype(np.float64)
        mx = meta.ranges[i, :, 1].astype(np.float64)
        half_step = (mx - mn) / 510.0
        slack = 4.0 * F32_EPS * np.maximum(np.abs(mn), np.abs(mx))  # float32 rounding of the output
        err = np.abs(back - orig)
        violations += int((err > np.repeat(half_step + slack, sizes)).sum())
        moving = half_step > 0.0
        constant_groups += int((~moving).sum())
        if moving.any():
            worst = max(worst, float((np.maximum.reduceat(err, starts)[moving] / half_step[moving]).max()))
        if name in ("x", "y", "z"):
            sub_err[name] = err

    # one range per field ("int8"), only for the position axes
    xyz = {a: fields[a] for a in "xyz"}
    one_q, comments = quantize.quantize(xyz, "int8")
    minmax = {c.split()[1]: (float(c.split()[2]), float(c.split()[3])) for c in comments}
    rows = []
    for a in "xyz":
        lo, hi = minmax[a]
        naive = lo + one_q[a].astype(np.float64) / 255.0 * (hi - lo)
        e_naive = np.abs(naive - np.asarray(fields[a], dtype=np.float64))
        rows.append((a, hi - lo, float(e_naive.mean()), float(sub_err[a].mean())))
    return meta, violations, worst, constant_groups, rows


def check_scale_cap(pruned, scale_cap):
    """Per-axis p99 log-scale, and whether ``log(scale_cap)`` removes anything extra.

    ``clean`` only reads x, y, z and the three scales, so it is run on those six columns
    (same function, same masks, without copying every field).
    """
    cols = {k: pruned[k] for k in ("x", "y", "z", "scale_0", "scale_1", "scale_2")}
    logs = np.stack([cols[f"scale_{i}"] for i in range(3)], axis=-1)
    scale_high = inspect.signature(outliers.remove_outliers).parameters["scale_high"].default
    p99 = np.percentile(logs, scale_high, axis=0)
    log_cap = math.log(scale_cap)

    above = int((logs > log_cap).any(axis=1).sum())
    with_cap = outliers.remove_outliers(cols, scale_cap=scale_cap)
    without_cap = outliers.remove_outliers(cols, scale_cap=math.inf)
    survivors_above = int((np.stack([without_cap[f"scale_{i}"] for i in range(3)], axis=-1)
                           > log_cap).any(axis=1).sum())
    return scale_high, p99, log_cap, above, len(with_cap["x"]), len(without_cap["x"]), survivors_above


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ply", help="Gaussian-splat PLY (e.g. an exported splatfacto checkpoint)")
    ap.add_argument("--groups", type=int, default=quantize.DEFAULT_GROUPS, help="sub-group count G (default 1000)")
    ap.add_argument("--degree", type=int, default=3, help="SH degree to reduce to (default 3)")
    ap.add_argument("--percentile", type=float, default=5.0, help="prune percentile (default 5)")
    ap.add_argument("--scale-cap", type=float, default=1.0, help="linear scale cap for clean (default 1.0)")
    args = ap.parse_args(argv)

    t0 = time.time()
    fields = io.load_splat(args.ply)
    missing = [k for k in REQUIRED if k not in fields]
    if missing:
        print(f"error: {args.ply} lacks fields {missing}", file=sys.stderr)
        return 2
    n0, n_fields = len(fields["x"]), len(fields)
    fields = prune.prune_opacity(fields, args.percentile)
    n_pruned = len(fields["x"])

    cap = check_scale_cap(fields, args.scale_cap)

    fields = outliers.remove_outliers(fields, scale_cap=args.scale_cap)
    fields = sh.reduce_sh(fields, args.degree)
    n = len(fields["x"])
    meta, violations, worst, constant_groups, pos_rows = check_subgroup(fields, args.groups)

    print(f"{args.ply}")
    print(f"  {n0:,} Gaussians, {n_fields} fields | prune {args.percentile:g}th pct -> {n_pruned:,} | "
          f"clean (scale cap {args.scale_cap:g}) -> {n:,} | SH degree {args.degree} | {len(fields)} fields")

    print(f"\n[1] sub-group INT8 error bound, G = {meta.n_groups}")
    print(f"    fields x groups checked       : {len(meta.fields)} x {meta.n_groups} "
          f"({len(meta.fields) * meta.n_groups:,} ranges, {constant_groups:,} constant)")
    print(f"    values over (max-min)/510     : {violations}")
    print(f"    worst error / half-step       : {worst:.4f}")
    print(f"    range table (float32)         : {meta.nbytes / 1e6:.3f} MB")

    print("\n[2] mean position error, one range per field vs sub-group (scene units)")
    print("    axis   extent   one-range   sub-group   ratio")
    for a, extent, naive, sub in pos_rows:
        ratio = naive / sub if sub > 0 else math.inf
        print(f"    {a:>4}  {extent:7.2f}  {naive:10.5f}  {sub:10.5f}  {ratio:5.1f}x")
    pooled_naive = sum(r[2] for r in pos_rows)
    pooled_sub = sum(r[3] for r in pos_rows)
    print(f"    pooled (sum of per-axis means): {pooled_naive / pooled_sub if pooled_sub > 0 else math.inf:.1f}x")

    scale_high, p99, log_cap, above, with_cap, without_cap, survivors_above = cap
    print(f"\n[3] scale cap (applied to stored log-scales as <= log({args.scale_cap:g}) = {log_cap:g})")
    print(f"    per-axis {scale_high:g}th-percentile log-scale : "
          + "  ".join(f"{v:+.3f}" for v in p99) + f"   (max {p99.max():+.3f})")
    print(f"    Gaussians above the cap after prune : {above:,} ({above / n_pruned:.3%})")
    print(f"    ...of which the percentile bound keeps : {survivors_above:,}")
    print(f"    clean keeps with cap / without cap  : {with_cap:,} / {without_cap:,}")
    print(f"    cap removes anything on its own     : {with_cap != without_cap}")
    print(f"\n({time.time() - t0:.1f} s)")
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
