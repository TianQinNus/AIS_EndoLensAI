# e7b1_windows: Constrained calibration and routing diagnosis

[Complete E7 stage guide](../experiments/E7/README.md) | [Reproduction requirements](../docs/REPRODUCIBILITY.md)

## Purpose

Test affine calibration, MAE-constrained calibration, grouped cross-validation, and prediction-only shallow routing.

## Recorded outcome

E7-A: NO_PASS; E7-A2: NO_PASS; E7-B0: diagnostic headroom; E7-B1: OOF_NO_PASS; E7-B2 selection: NO_PASS.

Do not promote oracle routing or a failed candidate. Turn to training-data expansion while keeping E0 frozen.

## Entrypoints and prerequisites

Start in `SystemCode/model` with the historical `endolens-e2-win` environment and restored stage inputs. The following are original command files, ordered by filename. Check the stage guide before executing them; do not bypass NO_PASS or run test-application commands without an eligible locked configuration.

```text
01_test_e7b1.cmd
02_run_gate_cv.cmd
03_verify_outputs.cmd
```

Python, test and command-file bytes are preserved in the imported-source manifest. This English README is a navigation document, not part of the executed training recipe. Dataset images, checkpoints and full run/probability caches are not distributed in this source archive.
