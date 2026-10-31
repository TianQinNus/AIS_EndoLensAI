# E12: Verified negative frames and matched continuations

[Experiment index](../../README.md) | [Protocol and evidence](../../docs/EXPERIMENT_STATUS.md)

## Research question

Compare three frozen baselines and four matched 30-round U-Net continuations: R34/R50 positive-only controls and 75/25 positive/negative arms.

## Code packages

- [e12_negative_windows](../../e12_negative_windows/README.md)

## Recorded result

Recorded validation selection: VALIDATION_SELECTION_LOCKED_EXPLORATORY, selected r50_positive. Neither mixed continuation passed its declared negative-training-gain criteria.

## Interpretation and next decision

Late negative fine-tuning did not establish the desired protected improvement. Test positive/negative supervision from the beginning with a fresh segmentation decoder.

## Reproduction prerequisites

Run from `SystemCode/model`, not this navigation directory. Restore the authorised dataset, signed split manifests, exact upstream checkpoints and cached outputs required by this stage. Read [research reproduction requirements](../../docs/REPRODUCIBILITY.md) before invoking entrypoints. Later stages depend on earlier frozen run artifacts; a source-only clone is not a complete experiment environment. Do not override a failed gate or resume a run against changed source fingerprints.

## Original Windows entrypoints

The list documents the available entrypoints; it is not an unconditional batch to execute. Read the package instructions, run checks first, and stop when a selection or protection condition fails. Status, verification and optional analysis commands do not necessarily initiate training.

### e12_negative_windows

```powershell
.\e12_negative_windows\00_prepare_data.cmd
.\e12_negative_windows\01_verify_and_lock.cmd
.\e12_negative_windows\02_evaluate_frozen_validation.cmd
.\e12_negative_windows\03_train_all.cmd
.\e12_negative_windows\03a_train_r34_positive.cmd
.\e12_negative_windows\03b_train_r34_mixed.cmd
.\e12_negative_windows\03c_train_r50_positive.cmd
.\e12_negative_windows\03d_train_r50_mixed.cmd
.\e12_negative_windows\04_select_on_validation.cmd
.\e12_negative_windows\05_evaluate_common_test.cmd
.\e12_negative_windows\06_summarize.cmd
.\e12_negative_windows\07_optional_rotate4.cmd
.\e12_negative_windows\08_status.cmd
```
