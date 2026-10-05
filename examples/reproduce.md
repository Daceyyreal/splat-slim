# Reproducing the results

Environment: Kaggle T4 GPU, Nerfstudio `splatfacto`, MipNeRF-360.

## What this repo covers, and what it does not

The paper evaluated **`splatfacto` checkpoints** with Nerfstudio's `ns-eval`: each compressed model
was written back ("brain-swapped") into a copy of the trained checkpoint, and PSNR/SSIM/LPIPS were
computed on held-out views. `splat-slim` works on an **exported `.ply`** and has no renderer.

| Step in the paper's protocol | Where it happens |
|---|---|
| Train `splatfacto`, 30k iterations, `--downscale-factor 4` | Nerfstudio (install sequence below) |
| Export the checkpoint to a PLY | Nerfstudio |
| Prune, clean, reduce SH, quantize | `splat-slim run` |
| Write the compressed model back into a copy of the checkpoint | **not included**; see "Getting quality numbers" |
| PSNR / SSIM / LPIPS | Nerfstudio `ns-eval` |

So `splat-slim` reproduces the **file sizes** of the paper's pipeline. The quality columns need the
checkpoint round trip.

## Verified install sequence (Kaggle)

```bash
pip install nerfstudio
pip install numpy==1.26.4 --force-reinstall
pip install plyfile pytorch-msssim tensorboard
# patch eval_utils.py + trainer.py for weights_only=False (see note below)
MAX_JOBS=2 pip install gsplat   # pre-compile to avoid OOM during build
```

Notes:
- Install `tensorboard` explicitly, otherwise the jax/tensorflow dependency
  chain re-breaks numpy when nerfstudio dataparsers are imported.
- The current nerfstudio never sets `weights_only` explicitly, so a regex
  injection (not `sed`) is needed to force `weights_only=False` on torch.load.
- COLMAP GPU matching is unavailable on Kaggle (software OpenGL only); use CPU.
- For MipNeRF-360's non-standard layout, pass `--colmap-path sparse/0` and hide
  any pre-downscaled image folders so nerfstudio re-derives them.

## Per-scene commands

Train the baseline (30k iterations, `--downscale-factor 4`), export it to a PLY, then:

```bash
# Headline operating point: mixed precision, SH degree 3 (~74% smaller)
splat-slim run garden.ply  garden_mixed.ply  --degree 3 --quant mixed
splat-slim run bicycle.ply bicycle_mixed.ply --degree 3 --quant mixed
splat-slim run vase.ply    vase_mixed.ply    --degree 3 --quant mixed

# Conservative: every field FP16 (~55% smaller)
splat-slim run garden.ply  garden_fp16.ply   --degree 3 --quant fp16

# Sub-group INT8: writes garden_sub.ply and garden_sub.meta.npz (keep them together)
splat-slim run garden.ply  garden_sub.ply    --degree 3 --quant int8-subgroup
```

The paper's SH ablations are `--degree 2` and `--degree 1`. The default `--percentile 5` and
`--scale-cap 1.0` are the paper's settings (5th-percentile opacity threshold; stored log-scales
capped at log 1.0 = 0).

Sizes on the Garden scene (1,572,747 Gaussians, 390.0 MB), measured from the files this CLI writes:

| Command | Output size |
|---|---|
| `--degree 3 --quant mixed` | 101.0 MB |
| `--degree 3 --quant fp16` | 173.9 MB |
| `--degree 3 --quant int8-subgroup` | 87.0 MB + 0.43 MB sidecar |

The first two match the paper's Garden table. After pruning and cleaning, 1,402,751 Gaussians remain.

## Getting quality numbers

The checkpoint stores the model as tensors under `_model.gauss_params`: `means` (n x 3),
`features_dc` (n x 3), `features_rest` (n x 15 x 3), `opacities` (n x 1, logit space), `scales`
(n x 3, log space) and `quats` (n x 4). In the PLY, `f_rest_0` to `f_rest_44` are channel-blocked:
15 R coefficients, then 15 G, then 15 B.

To score a `splat-slim` output the way the paper did:

1. Turn the output back into float values: nothing to do for `--quant none`; for
   `--quant int8-subgroup` run `splat-slim dequantize in.ply out.ply`; for `fp16`, `int8` and
   `mixed`, decode with the `quant_minmax` header comments and reinterpret the `uint16` fields as
   `float16`. (`dequantize` only covers `int8-subgroup` today.)
2. Write the tensors into a copy of the checkpoint, keeping every tensor in the same Gaussian order
   (`int8-subgroup` output is in Morton order). If you reduced the SH degree, zero-pad
   `features_rest` back to (n, 15, 3) so the model accepts it. The paper round-trips FP16 values
   through `float16` so the evaluated model matches what a viewer would see.
3. Run `ns-eval` on that checkpoint.

The injection step is not part of this package.
