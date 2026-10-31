# E3: Fixed E0/E2 probability fusion

[Experiment index](../../README.md) | [Protocol and evidence](../../docs/EXPERIMENT_STATUS.md)

## Research question

Test whether fixed equal-weight probabilities from complementary networks improve the reference.

## Code packages

- [v3_windows](../../v3_windows/README.md)

## Recorded result

Historical Dice/IoU: 0.83290/0.76058, below frozen E0 (0.83678/0.76300).

## Interpretation and next decision

Fusion is not automatically better than its strongest component. The original E3 implementation directory is named v3_windows.

## Reproduction prerequisites

Run from `SystemCode/model`, not this navigation directory. Restore the authorised dataset, signed split manifests, exact upstream checkpoints and cached outputs required by this stage. Read [research reproduction requirements](../../docs/REPRODUCIBILITY.md) before invoking entrypoints. Later stages depend on earlier frozen run artifacts; a source-only clone is not a complete experiment environment. Do not override a failed gate or resume a run against changed source fingerprints.

## Original Windows entrypoints

The list documents the available entrypoints; it is not an unconditional batch to execute. Read the package instructions, run checks first, and stop when a selection or protection condition fails. Status, verification and optional analysis commands do not necessarily initiate training.

### v3_windows

```powershell
.\v3_windows\01_check_and_smoke.cmd
.\v3_windows\02_run_full.cmd
```
