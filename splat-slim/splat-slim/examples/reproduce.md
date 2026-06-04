# Reproducing the results

Environment: Kaggle T4 GPU, Nerfstudio `splatfacto`, MipNeRF-360.

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

```bash
# Train baseline (30k iters), then:
splat-slim run garden.ply  garden_slim.ply  --degree 3 --quant fp16
splat-slim run bicycle.ply bicycle_slim.ply --degree 1 --quant mixed
splat-slim run vase.ply    vase_slim.ply    --degree 3 --quant mixed
```

Fill in exact eval commands / render scripts once stages are ported.
