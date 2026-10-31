# E5: Fixed E0/E4 equal-weight fusion

[Experiment index](../../README.md) | [Protocol and evidence](../../docs/EXPERIMENT_STATUS.md)

## Research question

Freeze both branches and average native probabilities equally, with pixel threshold 0.5 and no weight search.

## Code packages

- [e5_windows](../../e5_windows/)

## Recorded result

Historical Dice/IoU: 0.837386/0.765085. Dice rose by only 0.000604 versus E0; MAE increased from 0.021855 to 0.033442.

## Interpretation and next decision

A small Dice gain did not establish a uniformly better configuration; investigate calibrated fusion on separate development data.

## Reproduction prerequisites

Run from `SystemCode/model`, not this navigation directory. Restore the authorised dataset, signed split manifests, exact upstream checkpoints and cached outputs required by this stage. Read [research reproduction requirements](../../docs/REPRODUCIBILITY.md) before invoking entrypoints. Later stages depend on earlier frozen run artifacts; a source-only clone is not a complete experiment environment. Do not override a failed gate or resume a run against changed source fingerprints.

## Original Windows entrypoints

The list documents the available entrypoints; it is not an unconditional batch to execute. Read the package instructions, run checks first, and stop when a selection or protection condition fails. Status, verification and optional analysis commands do not necessarily initiate training.

### e5_windows

```powershell
.\e5_windows\01_check_and_smoke.cmd
.\e5_windows\02_run_full.cmd
```
