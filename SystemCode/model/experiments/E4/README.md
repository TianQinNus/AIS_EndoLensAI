# E4: Stronger augmentation control

[Experiment index](../../README.md) | [Protocol and evidence](../../docs/EXPERIMENT_STATUS.md)

## Research question

Change augmentation while retaining the pretrained ResNet34 U-Net development setup.

## Code packages

- [e4_windows](../../e4_windows/README.md)

## Recorded result

Historical Dice/IoU: 0.76091/0.69031. Zero-Dice cases fell from 51 to 19 versus E2, but Dice<0.5 cases rose from 149 to 151.

## Interpretation and next decision

Examine foreground errors and complementarity; final E2/E4 training lengths differ, limiting single-factor attribution.

## Reproduction prerequisites

Run from `SystemCode/model`, not this navigation directory. Restore the authorised dataset, signed split manifests, exact upstream checkpoints and cached outputs required by this stage. Read [research reproduction requirements](../../docs/REPRODUCIBILITY.md) before invoking entrypoints. Later stages depend on earlier frozen run artifacts; a source-only clone is not a complete experiment environment. Do not override a failed gate or resume a run against changed source fingerprints.

## Original Windows entrypoints

The list documents the available entrypoints; it is not an unconditional batch to execute. Read the package instructions, run checks first, and stop when a selection or protection condition fails. Status, verification and optional analysis commands do not necessarily initiate training.

### e4_windows

```powershell
.\e4_windows\01_check_and_smoke.cmd
.\e4_windows\02_train_development.cmd
.\e4_windows\03_compare_validation.cmd
```
