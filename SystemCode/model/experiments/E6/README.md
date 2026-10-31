# E6: External development and calibrated fusion

[Experiment index](../../README.md) | [Protocol and evidence](../../docs/EXPERIMENT_STATUS.md)

## Research question

Prepare FAR-POLYP-SEG, preserve patient-grouped calibration/selection roles, and test temperature-calibrated fixed fusion.

## Code packages

- [e6_windows](../../e6_windows/)

## Recorded result

Selection status NO_PASS. The recorded nearest candidate had Dice 0.770947 versus E0 0.767970, but MAE increased by 0.031665.

## Interpretation and next decision

Do not run the formal-test application command when selection gates fail. FAR means the external FAR-POLYP-SEG development dataset, not a false-alarm-rate metric.

## Reproduction prerequisites

Run from `SystemCode/model`, not this navigation directory. Restore the authorised dataset, signed split manifests, exact upstream checkpoints and cached outputs required by this stage. Read [research reproduction requirements](../../docs/REPRODUCIBILITY.md) before invoking entrypoints. Later stages depend on earlier frozen run artifacts; a source-only clone is not a complete experiment environment. Do not override a failed gate or resume a run against changed source fingerprints.

## Original Windows entrypoints

The list documents the available entrypoints; it is not an unconditional batch to execute. Read the package instructions, run checks first, and stop when a selection or protection condition fails. Status, verification and optional analysis commands do not necessarily initiate training.

### e6_windows

```powershell
.\e6_windows\01_test_and_prepare.cmd
.\e6_windows\02_smoke_inference.cmd
.\e6_windows\03_infer_development.cmd
.\e6_windows\04_calibrate_and_select.cmd
.\e6_windows\05_apply_locked_test_once.cmd
```
