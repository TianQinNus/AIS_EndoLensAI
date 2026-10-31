# e8_repeat_windows: Expanded data and difficult-sample ablation

[Complete E8 stage guide](../experiments/E8/README.md) | [Reproduction requirements](../docs/REPRODUCIBILITY.md)

## Purpose

Cross original/expanded training data with ordinary/difficult-image sampling and repeat seeds 42, 17 and 2024.

## Recorded outcome

B source-equal validation Dice: 0.80584 +/- 0.00163 versus A 0.75098 +/- 0.00556. Historical C6 B Dice/IoU: 0.84709/0.78151 (352, identity). B_GATES_NOT_REPRODUCED_ALL_SEEDS: seed 2024 failed the original-source low-Dice guard.

Retain expanded-data B as the research reference, with the failed all-seed protection explicitly recorded; do not call it an all-gates-passing result.

## Entrypoints and prerequisites

Start in `SystemCode/model` with the historical `endolens-e2-win` environment and restored stage inputs. The following are original command files, ordered by filename. Check the stage guide before executing them; do not bypass NO_PASS or run test-application commands without an eligible locked configuration.

```text
01_check_and_prepare.cmd
02_train_seed17.cmd
03_train_seed2024.cmd
04_summarize_seeds.cmd
05_status.cmd
```

Python, test and command-file bytes are preserved in the imported-source manifest. This English README is a navigation document, not part of the executed training recipe. Dataset images, checkpoints and full run/probability caches are not distributed in this source archive.
