# splat-slim

**Lightweight, training-free post-processing to shrink 3D Gaussian Splatting models.**

[![CI](https://github.com/Daceyyreal/splat-slim/actions/workflows/ci.yml/badge.svg)](https://github.com/Daceyyreal/splat-slim/actions)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

`splat-slim` takes a trained 3D Gaussian Splatting (3DGS) model and shrinks it
**without retraining**. It runs as a post-processing pass — prune, clean,
reduce spherical harmonics, quantize — and is framework-agnostic: it operates on
the exported `.ply`, so it works regardless of the trainer that produced it.

> Companion code for *"[paper title]"* (ICCA 2026). See [Citation](#citation).

---

## Results

A single tool, three scenes, large size cuts at near-lossless quality. Best Pareto
point per scene:

| Scene   | Config              | Size (MB) | Size ↓ | ΔPSNR (dB) |
|---------|---------------------|-----------|--------|------------|
| Garden  | pruned + SH3 + FP16 | 170       | 54%    | −0.54      |
| Bicycle | pruned + SH1 + mixed| 82        | 87%    | −2.32      |
| Vase    | pruned + SH3 + mixed| 55.5      | 74.3%  | −1.13      |

Key findings:
- The opacity threshold **self-calibrates** per scene (≈0.005 → 0.015), confirming the *adaptive* design empirically.
- Mixed-precision (FP16 geometry + INT8 appearance) matches full-FP16 PSNR across all SH levels.
- Indoor matte scenes tolerate more aggressive SH reduction than outdoor scenes.

> ⚠️ The numbers above are scaffolded from the ablation tables — re-confirm against your final results before tagging a release.

<p align="center"><img src="assets/garden_before_after.png" width="80%" alt="Garden before/after"></p>

---

## Install

```bash
pip install splat-slim          # core CLI (numpy, plyfile, typer)
pip install "splat-slim[torch]" # add torch for tensor-based stages/metrics
```

From source:

```bash
git clone https://github.com/Daceyyreal/splat-slim
cd splat-slim
pip install -e ".[dev]"
```

## Quickstart

```bash
# Inspect a model
splat-slim info scene.ply

# Run the full pipeline
splat-slim run scene.ply slim.ply --degree 3 --quant mixed

# Or run individual stages
splat-slim prune scene.ply a.ply --percentile 5
splat-slim clean a.ply b.ply --scale-cap 1.0
splat-slim reduce-sh b.ply c.ply --degree 2
```

## How it works

| Stage | What it does |
|-------|--------------|
| 1. Prune    | Drop near-transparent Gaussians below an adaptive opacity percentile. |
| 2. Clean    | Remove spatial floaters and cap over-large scales. |
| 3. Reduce SH| Lower spherical-harmonic degree (3→2→1) with channel-blocked slicing. |
| 4. Quantize | FP16 geometry + INT8 appearance (mixed precision). |
| 5. Report   | Measure size reduction and PSNR to find the Pareto point. |

## Reproducing the paper results

Tested on Kaggle T4 (Nerfstudio `splatfacto`, MipNeRF-360). See
[`examples/reproduce.md`](examples/reproduce.md) for the verified install
sequence and per-scene commands.

## Roadmap

- [ ] Sub-group INT8 quantization (recovers the PSNR that naive per-column INT8 destroys).
- [ ] Interactive web viewer with a live size/quality slider.
- [ ] Direct nerfstudio checkpoint (`.ckpt`) input, not just `.ply`.

## Citation

```bibtex
@inproceedings{splatslim2026,
  title     = {PLACEHOLDER},
  author    = {Dace and others},
  booktitle = {ICCA},
  year      = {2026}
}
```

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgements

Built on [Nerfstudio](https://github.com/nerfstudio-project/nerfstudio) and
[gsplat](https://github.com/nerfstudio-project/gsplat); evaluated on the
[MipNeRF-360](https://jonbarron.info/mipnerf360/) dataset.
