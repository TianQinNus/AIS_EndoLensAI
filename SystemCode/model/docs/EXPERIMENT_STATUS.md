# Model development and experiment record

Updated 31 October 2026. This branch preserves experimental source snapshots; `main` contains the final E13 application release. Historical models are research references, not alternative production defaults.

## Research questions and progression

| Stage | Question and design | Outcome and next decision |
|---|---|---|
| E0 | Freeze the supplied Polyp-PVT as the reference. | Establish a reproducible external baseline before changing our own U-Net. |
| E1-E2 | Compare random U-Net training with an ImageNet-pretrained ResNet34 encoder. | Encoder pretraining improved generalisation; retain the pretrained U-Net route. |
| E3-E4 | Examine equal-weight fusion and stronger augmentation. | E3 did not exceed E0 on the historical benchmark; E4 reduced complete failures but increased false foreground. |
| E5-E7 | Investigate fusion, calibration and frozen shallow decision rules. | Development constraints and repeated NO_PASS outcomes did not support unrestricted fusion. Keep the reference fixed and improve the U-Net training data. |
| E8 | Cross original/expanded data with ordinary/difficult-sample training; repeat three seeds. | Data expansion gave the clearest validation benefit. Retain expanded-data variant B. |
| E9-E11 | Investigate source sampling, constrained fusion and background-error behaviour. | Continue from B rather than assume fusion is beneficial. Large erroneous foreground regions motivated explicit negative-image training. |
| Collaborator series | Develop GroupNorm U-Net variants, pretrained ResNet encoders and positive-data joint training. | Positive-only ResNet50 achieved the highest matched positive C6 Dice, but produced frequent foreground predictions on negative frames. |
| E12 | Add verified negative frames during fine-tuning and compare frozen models under a common protocol. | Negative training exposed a segmentation/false-alarm trade-off; aggregate positive Dice alone was insufficient for selection. |
| E13 | Start with ImageNet ResNet50 V2 plus a random GroupNorm U-Net decoder; train with positives and negatives from epoch one. | Retain epoch 94 as the best-positive fallback. No checkpoint met every prespecified validation protection criterion. |
| E13 output study | Keep weights fixed; investigate pixel thresholds and connected-region filtering. | Integrate native 8-connected regions with mean score >=0.8 after pixel threshold 0.5, with rotate4 logit averaging. |

This sequence describes the development rationale, not a set of strict single-factor comparisons. Changes in architecture, training history, data and inference protocol must be considered when comparing stages.

## Historical E0-E4 benchmark

The historical benchmark contains 798 entries. At 352 input under the historical protocol, E0 Dice/IoU were 0.8368/0.7630; random U-Net E1 achieved 0.5707/0.4876; E2 Dice was 0.7549; E3 Dice/IoU were 0.8329/0.7606; E4 Dice/IoU were 0.7609/0.6903. E4 reduced zero-Dice cases from 51 to 19 relative to E2, but low-Dice cases increased from 149 to 151. These numbers must not be substituted for the later matched C6 results.

## E8 data and difficulty ablation

Validation values are source-equal Dice means and sample standard deviations across three seeds. C6 uses seed 42, 352 input and historical identity inference.

| Variant | Validation Dice | C6 Dice | C6 IoU |
|---|---:|---:|---:|
| A: original | 0.75098 +/- 0.00556 | 0.77083 | 0.70568 |
| B: expanded | 0.80584 +/- 0.00163 | 0.84709 | 0.78151 |
| C: original + difficult | 0.75030 +/- 0.00546 | 0.79115 | 0.72907 |
| D: expanded + difficult | 0.80306 +/- 0.00361 | 0.81952 | 0.76212 |
| Frozen E0 reference | Not applicable | 0.83046 | 0.77155 |

The expansion benefit was more consistent than difficult-sample weighting. Variant B remained the reference for subsequent U-Net investigations.

## Final training and evaluation

See [the final model record](FINAL_MODEL_RELEASE.md) for exact architecture, 100-epoch training settings, replacement sampling, data accounting, eligibility limits, validation failure, and full timing boundaries. The training pool contains 2,269 positive and 1,907 verified negative images; each epoch draws 1,160 samples with a 50/50 positive/negative probability. Positive PolypGen centres C1-C4 train, C5 validates, and C6 evaluates. Negative sequences are partitioned 1-10 / 11-13 / 14-23 for training / validation / evaluation.

The matched development comparison uses 320 input, FP32, four-rotation logit averaging and native-resolution scoring on 83 positive and 1,760 negative C6 images.

| Configuration | Positive Dice | Positive IoU | Negative-frame FPR |
|---|---:|---:|---:|
| Frozen Polyp-PVT | 0.84357 | 0.78978 | 66.19% |
| Frozen ResNet34 B | 0.84647 | 0.78440 | 56.02% |
| Positive-only ResNet50 | 0.88389 | 0.82727 | 89.72% |
| E13, unfiltered | 0.85883 | 0.79540 | 30.80% |
| E13, region mean >=0.8 | 0.85607 | 0.79299 | 20.97% |

The region filter removes predictions on 173 additional negative frames (542 to 369) while positive Dice decreases by 0.00275. Seven positive images have Dice below 0.5, including three with zero Dice. The positive-only ResNet50 remains the strongest positive-overlap model; final E13 is the team's lower-false-alarm integration choice.

## Selection limitations

E13 permitted at most 0.02 overall positive Dice loss, 0.03 loss in either source, five additional low-Dice positives and three additional zero-Dice positives relative to the positive-only ResNet50 validation reference. Epoch 94 instead lost 0.05537 Dice, with low-Dice counts increasing from 39 to 62 and zero-Dice counts from 11 to 34. It therefore remained a fallback, not a guard-passing configuration. No new quantitative acceptance margin was validated. Later C6 results do not reverse the failed validation criteria.

C6 informed output-rule selection and was reused during development. Results are descriptive, not an independent final test. Comparator-specific optimal filtering and an otherwise identical fresh-start positive-only control were not completed. A new patient/video-grouped positive/negative evaluation is required with the model and rule frozen in advance. Candidate frames are observations, not distinct lesions; region scores are not calibrated disease probabilities.

## Code and evidence map

- `e0_mac`, `e1_mac`, `e1_windows`, `e2_windows`, `e4_windows`, `v3_windows`, `lib`, `scripts`: preserved E0-E4 code and supporting tools.
- `endolens_b`: historical ResNet34 B handover, retained for traceability.
- Other U-Net exploration directories: collaborator experiment sources retained from the existing model branch.
- [endolens_e13](../endolens_e13/README.md): final portable inference SDK, external weight installer and contract checks.
- [Full-precision final metrics](../../../Miscellaneous/model_results/2026-10-31/README.md): matched comparison and controlled latency CSVs.
- [Project presentation](https://youtu.be/qXlUmzXPsoE): final team presentation and application demonstration.

Historical source snapshots are preserved without rewriting executed training recipes. The newly imported E5-E14 Python/CMD/configuration files are indexed in [the source manifest](IMPORTED_EXPERIMENT_SOURCES.json). Use [the E0-E13 index](../README.md) and [reproduction requirements](REPRODUCIBILITY.md) for package entrypoints and required external artifacts. E14 is the historical postprocessing folder name for the final E13 output-rule study, not a new trained model.

The available E11 run record still reports TRAINING and does not establish completion of all three seeds. E8's all-seed guard status remains B_GATES_NOT_REPRODUCED_ALL_SEEDS; E12's recorded exploratory selection retained r50_positive and neither mixed continuation passed its negative-training-gain criteria. These outcomes are retained rather than replaced by later E13 development results.
