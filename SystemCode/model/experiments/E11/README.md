# E11: Difficult-background pixel weighting

[Experiment index](../../README.md) | [Protocol and evidence](../../docs/EXPERIMENT_STATUS.md)

## Research question

Add detached probability-weighted background BCE to the existing loss while retaining positive-only B data and sampling.

## Code packages

- [e11_background_windows](../../e11_background_windows/)

## Recorded result

The available local run record still reports TRAINING; it does not establish a completed three-seed result. No final improvement is asserted here.

## Interpretation and next decision

Preserve the hypothesis and source; separate background pixels within positive images from genuinely negative full frames.

## Reproduction prerequisites

Run from `SystemCode/model`, not this navigation directory. Restore the authorised dataset, signed split manifests, exact upstream checkpoints and cached outputs required by this stage. Read [research reproduction requirements](../../docs/REPRODUCIBILITY.md) before invoking entrypoints. Later stages depend on earlier frozen run artifacts; a source-only clone is not a complete experiment environment. Do not override a failed gate or resume a run against changed source fingerprints.

## Original Windows entrypoints

The list documents the available entrypoints; it is not an unconditional batch to execute. Read the package instructions, run checks first, and stop when a selection or protection condition fails. Status, verification and optional analysis commands do not necessarily initiate training.

### e11_background_windows

```powershell
.\e11_background_windows\00_review_cached_errors.cmd
.\e11_background_windows\01_test_and_prepare.cmd
.\e11_background_windows\02_smoke.cmd
.\e11_background_windows\03_train_all.cmd
.\e11_background_windows\04_summarize.cmd
.\e11_background_windows\05_status.cmd
.\e11_background_windows\06_verify_summary.cmd
```
