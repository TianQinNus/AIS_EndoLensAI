# E9: Original/expanded source-balance revision

[Experiment index](../../README.md) | [Protocol and evidence](../../docs/EXPERIMENT_STATUS.md)

## Research question

Change only original/new source sampling mass from 50/50 to 70/30 and retrain three seeds from the same ImageNet initialisation.

## Code packages

- [e9_balance_windows](../../e9_balance_windows/)

## Recorded result

NO_PASS_RETAIN_B: the 70/30 revision failed the prespecified multi-seed protection criteria.

## Interpretation and next decision

Keep the existing 50/50-source B recipe; do not select a favourable seed or change the historical E8 decision.

## Reproduction prerequisites

Run from `SystemCode/model`, not this navigation directory. Restore the authorised dataset, signed split manifests, exact upstream checkpoints and cached outputs required by this stage. Read [research reproduction requirements](../../docs/REPRODUCIBILITY.md) before invoking entrypoints. Later stages depend on earlier frozen run artifacts; a source-only clone is not a complete experiment environment. Do not override a failed gate or resume a run against changed source fingerprints.

## Original Windows entrypoints

The list documents the available entrypoints; it is not an unconditional batch to execute. Read the package instructions, run checks first, and stop when a selection or protection condition fails. Status, verification and optional analysis commands do not necessarily initiate training.

### e9_balance_windows

```powershell
.\e9_balance_windows\01_check_and_smoke.cmd
.\e9_balance_windows\02_train_all.cmd
.\e9_balance_windows\03_summarize.cmd
.\e9_balance_windows\04_status.cmd
```
