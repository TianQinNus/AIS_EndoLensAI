# E2: Encoder initialisation controls

[Experiment index](../../README.md) | [Protocol and evidence](../../docs/EXPERIMENT_STATUS.md)

## Research question

Compare random U-Net and random/pretrained ResNet34 encoder configurations under the development protocol.

## Code packages

- [e2_windows](../../e2_windows/README.md)

## Recorded result

The selected pretrained ResNet34 U-Net used 67 full-data training rounds; historical evaluation Dice was 0.7549.

## Interpretation and next decision

Use the pretrained ResNet34 encoder route for subsequent controlled experiments.

## Reproduction prerequisites

Run from `SystemCode/model`, not this navigation directory. Restore the authorised dataset, signed split manifests, exact upstream checkpoints and cached outputs required by this stage. Read [research reproduction requirements](../../docs/REPRODUCIBILITY.md) before invoking entrypoints. Later stages depend on earlier frozen run artifacts; a source-only clone is not a complete experiment environment. Do not override a failed gate or resume a run against changed source fingerprints.

## Original Windows entrypoints

The list documents the available entrypoints; it is not an unconditional batch to execute. Read the package instructions, run checks first, and stop when a selection or protection condition fails. Status, verification and optional analysis commands do not necessarily initiate training.

### e2_windows

```powershell
.\e2_windows\01_check_and_smoke.cmd
.\e2_windows\02_train_development.cmd
.\e2_windows\03_select_and_train_final.cmd
.\e2_windows\04_evaluate_cuda.cmd
```
