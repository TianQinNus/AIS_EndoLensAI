# e9_balance_windows: Original/expanded source-balance revision

[Complete E9 stage guide](../experiments/E9/README.md) | [Reproduction requirements](../docs/REPRODUCIBILITY.md)

## Purpose

Change only original/new source sampling mass from 50/50 to 70/30 and retrain three seeds from the same ImageNet initialisation.

## Recorded outcome

NO_PASS_RETAIN_B: the 70/30 revision failed the prespecified multi-seed protection criteria.

Keep the existing 50/50-source B recipe; do not select a favourable seed or change the historical E8 decision.

## Entrypoints and prerequisites

Start in `SystemCode/model` with the historical `endolens-e2-win` environment and restored stage inputs. The following are original command files, ordered by filename. Check the stage guide before executing them; do not bypass NO_PASS or run test-application commands without an eligible locked configuration.

```text
01_check_and_smoke.cmd
02_train_all.cmd
03_summarize.cmd
04_status.cmd
```

Python, test and command-file bytes are preserved in the imported-source manifest. This English README is a navigation document, not part of the executed training recipe. Dataset images, checkpoints and full run/probability caches are not distributed in this source archive.
