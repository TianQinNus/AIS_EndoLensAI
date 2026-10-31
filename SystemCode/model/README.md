# Model research: E0-E13

This branch records how EndoLens AI progressed from a frozen Polyp-PVT reference and basic U-Net to positive/negative joint training and the final E13 output rule. Use [main](https://github.com/TianQinNus/AIS_EndoLensAI/tree/main) for application deployment; use this branch to follow the research, inspect source and understand both successful and unsuccessful experiments.

## Read the development in order

| Stage | Research focus | Packages |
|---|---|---|
| [E0](experiments/E0/README.md) | Frozen Polyp-PVT reference | `e0_mac` |
| [E1](experiments/E1/README.md) | Random-initialised U-Net baseline | `e1_mac`, `e1_windows` |
| [E2](experiments/E2/README.md) | Encoder initialisation controls | `e2_windows` |
| [E3](experiments/E3/README.md) | Fixed E0/E2 probability fusion | `v3_windows` |
| [E4](experiments/E4/README.md) | Stronger augmentation control | `e4_windows` |
| [E5](experiments/E5/README.md) | Fixed E0/E4 equal-weight fusion | `e5_windows` |
| [E6](experiments/E6/README.md) | External development and calibrated fusion | `e6_windows` |
| [E7](experiments/E7/README.md) | Constrained calibration and routing diagnosis | `e7a_windows`, `e7a2_windows`, `e7b0_windows`, `e7b1_windows`, `e7b2_windows` |
| [E8](experiments/E8/README.md) | Expanded data and difficult-sample ablation | `e8_windows`, `e8_repeat_windows` |
| [E9](experiments/E9/README.md) | Original/expanded source-balance revision | `e9_balance_windows` |
| [E10](experiments/E10/README.md) | Frozen B/E0 fusion feasibility | `e10a_windows`, `e10b_windows`, `e10c_windows`, `e10d_windows` |
| [E11](experiments/E11/README.md) | Difficult-background pixel weighting | `e11_background_windows` |
| [E12](experiments/E12/README.md) | Verified negative frames and matched continuations | `e12_negative_windows` |
| [E13](experiments/E13/README.md) | Joint positive/negative ResNet50 and final output rule | `e13_joint_windows`, `e14_postprocess_windows`, `endolens_e13` |

The main line is **establish baselines -> test pretraining and fusion -> expand positive data -> diagnose false foreground -> introduce verified negatives -> train jointly from initialisation -> select a region-filtered integration configuration**. The collaborator's GroupNorm/ResNet experiments form a parallel architecture-development track and supply the positive-only ResNet50 reference used in E12/E13. Their existing `unet*` folders remain intact and are not reassigned invented E-numbers.

## Key outcomes and interpretation

- E5 improved historical mean Dice only slightly while worsening MAE; subsequent E6/E7 fusion candidates failed their declared conditions.
- E8 expansion improved three-seed mean validation Dice, but B did not reproduce every protection criterion in all seeds. E9's 70/30 revision failed, so the original B recipe remained the reference.
- E10 constrained fusion/rescue did not qualify. E11's available record does not establish completed three-seed performance; it is not reported as a proven improvement.
- E12's matched negative fine-tuning arms did not pass the declared negative-training-gain checks. E13 instead introduced positives and negatives from epoch one.
- Final E13 uses epoch 94, rotate4 logit averaging, pixel threshold 0.5 and native region mean >=0.8. Matched C6 positive Dice/IoU are 0.85607/0.79299 and negative-frame FPR is 20.97%. Positive-only R50 has higher positive Dice (0.88389), but higher negative-frame FPR (89.72%).
- No E13 checkpoint passed all prespecified positive-protection criteria. It was selected for exploratory lower-false-alarm application integration. Reused C6 results do not retrospectively establish validation compliance or independent generalisation.

## Reproduction and evidence

- [Requirements, data roles and external artifacts](docs/REPRODUCIBILITY.md)
- [Development record and comparison boundaries](docs/EXPERIMENT_STATUS.md)
- [Final architecture, training settings, data counts and selection limits](docs/FINAL_MODEL_RELEASE.md)
- [Portable E13 SDK and Google Drive weight installation](endolens_e13/README.md)
- [Compact experiment evidence](experiment_evidence/README.md)
- [Final presentation](https://youtu.be/qXlUmzXPsoE)

The `experiments/E0` through `experiments/E13` folders are navigation guides. Executable package paths remain unchanged because they are part of the historical import and fingerprint contract. E14 is the original name of the E13 postprocessing study, not a replacement network. No training images, new checkpoints or native probability caches are added in this update.
