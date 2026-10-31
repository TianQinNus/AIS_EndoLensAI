# e5_windows: Fixed E0/E4 equal-weight fusion

[Complete E5 stage guide](../experiments/E5/README.md) | [Reproduction requirements](../docs/REPRODUCIBILITY.md)

## Purpose

Freeze both branches and average native probabilities equally, with pixel threshold 0.5 and no weight search.

## Recorded outcome

Historical Dice/IoU: 0.837386/0.765085. Dice rose by only 0.000604 versus E0; MAE increased from 0.021855 to 0.033442.

A small Dice gain did not establish a uniformly better configuration; investigate calibrated fusion on separate development data.

## Entrypoints and prerequisites

Start in `SystemCode/model` with the historical `endolens-e2-win` environment and restored stage inputs. The following are original command files, ordered by filename. Check the stage guide before executing them; do not bypass NO_PASS or run test-application commands without an eligible locked configuration.

```text
01_check_and_smoke.cmd
02_run_full.cmd
```

Python, test and command-file bytes are preserved in the imported-source manifest. This English README is a navigation document, not part of the executed training recipe. Dataset images, checkpoints and full run/probability caches are not distributed in this source archive.
