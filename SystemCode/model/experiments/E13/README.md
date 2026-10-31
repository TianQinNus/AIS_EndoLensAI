# E13: Joint positive/negative ResNet50 and final output rule

[Experiment index](../../README.md) | [Protocol and evidence](../../docs/EXPERIMENT_STATUS.md)

## Research question

Start from ImageNet ResNet50 V2 and a random GroupNorm decoder; use 50/50 positive/negative sampling from epoch one, then study fixed-weight output rules.

## Code packages

- [e13_joint_windows](../../e13_joint_windows/README.md)
- [e14_postprocess_windows](../../e14_postprocess_windows/README.md)
- [endolens_e13](../../endolens_e13/README.md)

## Recorded result

Epoch 94 is LOCKED_BEST_POSITIVE_ONLY_NO_PROTECTED_TRADEOFF. Matched final C6 Dice/IoU: 0.85607/0.79299; negative-frame FPR: 20.97%. Unfiltered E13: 0.85883/0.79540 and 30.80%.

## Interpretation and next decision

Selected for exploratory application integration at a lower-false-alarm operating point, not as a checkpoint passing the original validation criteria. The original postprocessing package is named E14 and is documented here as the E13 output-rule extension.

## Reproduction prerequisites

Run from `SystemCode/model`, not this navigation directory. Restore the authorised dataset, signed split manifests, exact upstream checkpoints and cached outputs required by this stage. Read [research reproduction requirements](../../docs/REPRODUCIBILITY.md) before invoking entrypoints. Later stages depend on earlier frozen run artifacts; a source-only clone is not a complete experiment environment. Do not override a failed gate or resume a run against changed source fingerprints.

## Original Windows entrypoints

The list documents the available entrypoints; it is not an unconditional batch to execute. Read the package instructions, run checks first, and stop when a selection or protection condition fails. Status, verification and optional analysis commands do not necessarily initiate training.

### e13_joint_windows

```powershell
.\e13_joint_windows\00_prepare.cmd
.\e13_joint_windows\01_smoke_test.cmd
.\e13_joint_windows\02_train_joint.cmd
.\e13_joint_windows\03_select_on_validation.cmd
.\e13_joint_windows\04_evaluate_common_c6.cmd
.\e13_joint_windows\05_summarize.cmd
.\e13_joint_windows\06_status.cmd
```

### e14_postprocess_windows

```powershell
.\e14_postprocess_windows\01_run_postprocessing.cmd
.\e14_postprocess_windows\02_rebuild_analysis.cmd
```

## Final inference and deployment

RGB -> 320-square resize -> embedded ImageNet normalisation once -> four rotations -> inverse-align and average logits -> native logit restoration -> sigmoid -> pixel threshold 0.5 -> native 8-connected components -> keep unrounded region mean >=0.8. The filtered mask defines regions, contours, overlay and candidate status. Empty filtered masks are not candidates. No area threshold or temporal confirmation is part of the final rule.

Use the [final release record](../../docs/FINAL_MODEL_RELEASE.md) for the exact architecture, data split, validation guard failures and controlled timing. Use [main](https://github.com/TianQinNus/AIS_EndoLensAI/tree/main) for the maintained web application. The E14 folder name is retained to preserve executed source identities; it is not a second trained model.
