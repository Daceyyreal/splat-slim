# splat-slim — build plan

## What it is (1 line)
A training-free CLI/library that shrinks any trained 3DGS `.ply` via prune → clean →
reduce-SH → quantize, with reproducible benchmarks. Companion code to your ICCA paper.

## Why this is the flagship
- It's real research, not a tutorial clone — almost no applicant has this.
- It's mostly *packaging* work you've already done, so high payoff per hour.
- A code link strengthens the paper; the paper makes the repo credible. Loop.
- It doubles as the engine for Project #2 (the web viewer).

## Definition of done
- [ ] `pip install -e .` works; `splat-slim run in.ply out.ply` runs end to end on a real scene.
- [ ] README results table matches your final numbers; one before/after render image.
- [ ] CI green; `v0.1.0` tagged release.
- [ ] Repo link added to the paper (even "code coming soon" counts before the deadline).

## Day-by-day

**Day 1 — DONE (this scaffold).** Repo structure, packaging, CLI, tests, CI, README,
license, reproduce guide. Your move: create the GitHub repo, push this, replace
`your-username` everywhere, fix the LICENSE name.

**Day 2 — Port your real logic.** Drop your validated functions into the stage
stubs (`src/splat_slim/stages/`). Three have working numpy starting points
(prune, outliers, fp16); two are marked `NotImplementedError` on purpose:
- `sh.reduce_sh` → your channel-blocked f_rest slicing.
- `quantize.int8_subgroup` → your sub-group INT8 (the FlexGaussian-killer).
Goal: one real scene runs through `splat-slim run` and matches your notebook output.

**Day 3 — Polish + ship.** Final README numbers, add `assets/garden_before_after.png`,
finish `examples/reproduce.md`, confirm CI green, tag `v0.1.0`, paste the link into the paper.

## What I need from you to do Day 2
Your current pipeline code — notebook (`garden_retrain.ipynb` / the ablation
notebook) or the scripts. Paste it or upload it and I'll port each stage into the
package and reconcile it with the stubs.

---

## Reusable project prompt (use for all 5 projects)
Paste this to me or Claude Code at the start of each project:

> I'm building **[PROJECT NAME]** for my GitHub. Context: final-year CS student,
> targeting CS masters in Canada/Australia + SWE jobs — the repo has to read as
> "real engineer," not "student following a tutorial."
> Standards: clean package layout, a README with real results/screenshots,
> tests, green CI, reproducible install. Honest assessment over flattery; give
> me working code and concrete next steps, recommend the higher-effort option
> when it's genuinely better.
> Current state: **[paste code / describe what exists]**.
> Today's task: **[the specific thing]**. Constraints: **[GPU/time/etc.]**.
