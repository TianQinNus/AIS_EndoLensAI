# e6_windows: External development and calibrated fusion

[Complete E6 stage guide](../experiments/E6/README.md) | [Reproduction requirements](../docs/REPRODUCIBILITY.md)

## Purpose

Prepare FAR-POLYP-SEG, preserve patient-grouped calibration/selection roles, and test temperature-calibrated fixed fusion.

## Recorded outcome

Selection status NO_PASS. The recorded nearest candidate had Dice 0.770947 versus E0 0.767970, but MAE increased by 0.031665.

Do not run the formal-test application command when selection gates fail. FAR means the external FAR-POLYP-SEG development dataset, not a false-alarm-rate metric.

## Entrypoints and prerequisites

Start in `SystemCode/model` with the historical `endolens-e2-win` environment and restored stage inputs. The following are original command files, ordered by filename. Check the stage guide before executing them; do not bypass NO_PASS or run test-application commands without an eligible locked configuration.

```text
01_test_and_prepare.cmd
02_smoke_inference.cmd
03_infer_development.cmd
04_calibrate_and_select.cmd
05_apply_locked_test_once.cmd
```

Python, test and command-file bytes are preserved in the imported-source manifest. This English README is a navigation document, not part of the executed training recipe. Dataset images, checkpoints and full run/probability caches are not distributed in this source archive.
