# Verifying the sub-group INT8 and scale-cap claims

`examples/verify_subgroup.py` re-checks, on any Gaussian-splat PLY, the numbers quoted in the v0.2.0
release notes and in the README's "Known differences" section. It runs the same stages as
`splat-slim run` (prune, clean, reduce-SH) and then uses the package's own code:

```bash
python examples/verify_subgroup.py scene.ply --groups 1000
```

1. **Error bound.** For every field and every group, every value is reconstructed to within half a
   quantization step of that group's range, `(max - min) / 510`.
2. **Position error.** Mean absolute error on x, y, z with one range per field (`--quant int8`)
   versus sub-group ranges, as a ratio. This is a quantization-error statistic in scene units, not a
   PSNR.
3. **Scale cap.** The per-axis 99th-percentile log-scale (the bound `clean` applies together with
   `log(scale_cap)`), and whether the cap at log-scale 0 removes anything the percentile bound does
   not.

Exit status is 1 if the error bound is violated. Run with `CPython 3.13.14`,
numpy 2.2.5; the script is deterministic.

## Garden

The paper's Garden export: 1,572,747 Gaussians, 390.0 MB.

```text
garden.ply
  1,572,747 Gaussians, 62 fields | prune 5th pct -> 1,494,109 | clean (scale cap 1) -> 1,402,751 | SH degree 3 | 62 fields

[1] sub-group INT8 error bound, G = 1000
    fields x groups checked       : 62 x 1000 (62,000 ranges, 3,000 constant)
    values over (max-min)/510     : 0
    worst error / half-step       : 1.0003
    range table (float32)         : 0.496 MB

[2] mean position error, one range per field vs sub-group (scene units)
    axis   extent   one-range   sub-group   ratio
       x    22.02     0.02167     0.00102   21.3x
       y    23.37     0.02289     0.00087   26.3x
       z    10.10     0.00931     0.00030   31.4x
    pooled (sum of per-axis means): 24.7x

[3] scale cap (applied to stored log-scales as <= log(1) = 0)
    per-axis 99th-percentile log-scale : -2.126  -2.398  -1.863   (max -1.863)
    Gaussians above the cap after prune : 1,504 (0.101%)
    ...of which the percentile bound keeps : 0
    clean keeps with cap / without cap  : 1,402,751 / 1,402,751
    cap removes anything on its own     : False

(14.6 s)
```

## Bicycle

The local Bicycle baseline PLY has **2,648,386** Gaussians; the paper reports 2,649,045 for its
Bicycle baseline (659 more), so this is not byte-for-byte the paper's checkpoint export and counts
after pruning and cleaning can differ slightly from the paper's.

```text
bicycle.ply
  2,648,386 Gaussians, 62 fields | prune 5th pct -> 2,515,966 | clean (scale cap 1) -> 2,355,251 | SH degree 3 | 62 fields

[1] sub-group INT8 error bound, G = 1000
    fields x groups checked       : 62 x 1000 (62,000 ranges, 3,000 constant)
    values over (max-min)/510     : 0
    worst error / half-step       : 1.0002
    range table (float32)         : 0.496 MB

[2] mean position error, one range per field vs sub-group (scene units)
    axis   extent   one-range   sub-group   ratio
       x    23.54     0.02301     0.00175   13.1x
       y    23.57     0.02311     0.00146   15.8x
       z     8.36     0.00821     0.00041   19.8x
    pooled (sum of per-axis means): 15.0x

[3] scale cap (applied to stored log-scales as <= log(1) = 0)
    per-axis 99th-percentile log-scale : -2.059  -2.441  -2.360   (max -2.059)
    Gaussians above the cap after prune : 720 (0.029%)
    ...of which the percentile bound keeps : 0
    clean keeps with cap / without cap  : 2,355,251 / 2,355,251
    cap removes anything on its own     : False

(27.3 s)
```

## What this shows

- **Bound:** 0 values over `(max - min) / 510` on both scenes, for all 62 fields x 1000 groups. The
  check allows a float32 rounding slack of `4 * eps * |value|` for the stored output, which is why
  the worst error is 1.0003 and 1.0002 half-steps and not exactly 1. The 3,000 constant ranges are
  `nx`, `ny`, `nz` (all zeros); they reconstruct exactly.
- **Position error:** sub-group ranges lower the mean error by 21.3x / 26.3x / 31.4x (x / y / z) on
  Garden and by 13.1x / 15.8x / 19.8x on Bicycle. The v0.2.0 release note's "21-31x" described
  Garden only; Bicycle is lower.
- **Scale cap:** the 99th-percentile log-scale is between -1.86 and -2.44 on both scenes, far below
  log(1.0) = 0. The 1,504 Garden and 720 Bicycle Gaussians above the cap are all already removed by
  the percentile bound, so the cap changes nothing on its own.
- **Not shown:** PSNR/SSIM/LPIPS. That needs the checkpoint round trip and `ns-eval`; see
  [`reproduce.md`](reproduce.md).
