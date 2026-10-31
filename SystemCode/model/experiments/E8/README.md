# E8: Expanded data and difficult-sample ablation

[Experiment index](../../README.md) | [Protocol and evidence](../../docs/EXPERIMENT_STATUS.md)

## Research question

Cross original/expanded training data with ordinary/difficult-image sampling and repeat seeds 42, 17 and 2024.

## Code packages

- [e8_windows](../../e8_windows/)
- [e8_repeat_windows](../../e8_repeat_windows/)

## Recorded result

B source-equal validation Dice: 0.80584 +/- 0.00163 versus A 0.75098 +/- 0.00556. Historical C6 B Dice/IoU: 0.84709/0.78151 (352, identity). B_GATES_NOT_REPRODUCED_ALL_SEEDS: seed 2024 failed the original-source low-Dice guard.

## Interpretation and next decision

Retain expanded-data B as the research reference, with the failed all-seed protection explicitly recorded; do not call it an all-gates-passing result.

## Reproduction prerequisites

Run from `SystemCode/model`, not this navigation directory. Restore the authorised dataset, signed split manifests, exact upstream checkpoints and cached outputs required by this stage. Read [research reproduction requirements](../../docs/REPRODUCIBILITY.md) before invoking entrypoints. Later stages depend on earlier frozen run artifacts; a source-only clone is not a complete experiment environment. Do not override a failed gate or resume a run against changed source fingerprints.

## Original Windows entrypoints

The list documents the available entrypoints; it is not an unconditional batch to execute. Read the package instructions, run checks first, and stop when a selection or protection condition fails. Status, verification and optional analysis commands do not necessarily initiate training.

### e8_windows

```powershell
.\e8_windows\00_download_and_prepare.cmd
.\e8_windows\01_check_and_smoke.cmd
.\e8_windows\02_train_baseline.cmd
.\e8_windows\03_train_expanded.cmd
.\e8_windows\04_train_hard_controls.cmd
.\e8_windows\05_compare_validation.cmd
.\e8_windows\06_lock_holdout.cmd
.\e8_windows\07_evaluate_locked_holdout.cmd
.\e8_windows\08_analyze_holdout.cmd
```

### e8_repeat_windows

```powershell
.\e8_repeat_windows\01_check_and_prepare.cmd
.\e8_repeat_windows\02_train_seed17.cmd
.\e8_repeat_windows\03_train_seed2024.cmd
.\e8_repeat_windows\04_summarize_seeds.cmd
.\e8_repeat_windows\05_status.cmd
```
