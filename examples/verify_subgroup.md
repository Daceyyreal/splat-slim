# Verifying the sub-group INT8, scale-cap and file-size claims

`examples/verify_subgroup.py` re-checks, on any Gaussian-splat PLY, the numbers quoted in the v0.2.0
release notes and in the README's "Known differences" section. It runs the same stages as
`splat-slim run` (prune, clean, reduce-SH) and then uses the package's own code:

```bash
python examples/verify_subgroup.py scene.ply --groups 1000 --sizes --tmp-dir /path/with/space
```

1. **Error bound.** For every field and every group, every value is reconstructed to within half a
   quantization step of that group's range, `(max - min) / 510`.
2. **Position error.** Mean absolute error on x, y, z with one range per field (`--quant int8`)
   versus sub-group ranges, as a ratio. This is a quantization-error statistic in scene units, not a
   PSNR.
3. **Scale cap.** The per-axis 99th-percentile log-scale (the bound `clean` applies together with
   `log(scale_cap)`), and whether the cap at log-scale 0 removes anything the percentile bound does
   not.
4. **File sizes** (`--sizes`). The real `splat-slim run --degree 3 --quant MODE` output in each of
   fp16, mixed, int8 and int8-subgroup, in MB (10^6 bytes) and MiB, next to the paper's value for
   that scene. The paper values are constants in the script, copied from the camera-ready's
   per-scene tables; the script flags a difference but does not fail on one. `--paper
   garden|bicycle|vase` picks the table (Garden is detected from its exact Gaussian count).
   Outputs go to `--tmp-dir` and are deleted right after being measured.

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

[4] file sizes of `splat-slim run --degree 3 --quant MODE` (prune 5th pct, scale cap 1, G = 1000); MB = 10^6 bytes
    paper column: garden. Gaussians after clean: 1,402,751 | paper-implied (pruned_sh3 347.9 MB / 248 B): 1,402,823 (-0.01%)
    mode              bytes          MB      MiB   paper MB  paper@our N  sidecar MB  status
    fp16              173,942,718     173.9    165.9      173.9        173.9           -  match
    mixed             101,003,103     101.0     96.3      101.0        101.0           -  match
    int8               86,976,232      87.0     82.9       87.0         87.0           -  match
    int8-subgroup      86,972,216      87.0     82.9       85.7         85.7        0.43  DIFFERS (+1.3 MB)
    note: int8-subgroup differs from the paper's measurement; see the README's 'Known differences from the paper'. Nothing is hard-coded to pass or fail on it.

(28.9 s)
```

## Bicycle

The local Bicycle baseline PLY has **2,648,386** Gaussians; the paper reports 2,649,045 for its
Bicycle baseline (659 more), so this is not byte-for-byte the paper's checkpoint export. After
pruning and cleaning it keeps 0.27% fewer Gaussians than the paper's file sizes imply, and the
"paper@our N" column rescales the paper's values to our count. Run with `--paper bicycle`.

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

[4] file sizes of `splat-slim run --degree 3 --quant MODE` (prune 5th pct, scale cap 1, G = 1000); MB = 10^6 bytes
    paper column: bicycle. Gaussians after clean: 2,355,251 | paper-implied (pruned_sh3 585.7 MB / 248 B): 2,361,694 (-0.27%)
    mode              bytes          MB      MiB   paper MB  paper@our N  sidecar MB  status
    fp16              292,052,718     292.1    278.5      292.9        292.1           -  match at our Gaussian count
    mixed             169,583,108     169.6    161.7      170.1        169.6           -  match at our Gaussian count
    int8              146,031,240     146.0    139.3      146.4        146.0           -  match at our Gaussian count
    int8-subgroup     146,027,216     146.0    139.3      146.4        146.0        0.43  match at our Gaussian count

(56.4 s)
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
- **Sizes, convention:** the paper's values are in 10^6-byte megabytes. In MiB the same Garden files
  would be 165.9 MB (fp16) and 96.3 MB (mixed), not the paper's 173.9 and 101.0.
- **Sizes, Garden:** fp16 (173.9 MB), mixed (101.0 MB) and int8 (87.0 MB) match the paper. The
  int8-subgroup PLY is 87.0 MB plus a 0.43 MB sidecar, against the paper's 85.7 MB: a difference of
  1.3 MB that this script cannot explain.
- **Sizes, Bicycle:** all four modes match the paper once rescaled to our Gaussian count, including
  int8-subgroup (146.0 MB for the PLY at our count vs 146.4 MB in the paper) plus the 0.43 MB
  sidecar. In the paper's own tables the int8 and int8-subgroup sizes are equal for Bicycle (146.4 /
  146.4) and Vase (46.8 / 46.8); Garden's pair (87.0 / 85.7) is the only one that differs.
- **Not shown:** PSNR/SSIM/LPIPS. That needs the checkpoint round trip and `ns-eval`; see
  [`reproduce.md`](reproduce.md).
