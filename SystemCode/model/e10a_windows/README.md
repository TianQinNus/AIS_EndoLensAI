# e10a_windows: Frozen B/E0 fusion feasibility

[Complete E10 stage guide](../experiments/E10/README.md) | [Reproduction requirements](../docs/REPRODUCIBILITY.md)

## Purpose

Diagnose complementarity on FAR; test three fixed fusion weights, constrained temperature calibration, and local rescue feasibility.

## Recorded outcome

E10-B: NO_CALIBRATION_PASS_SELECTION_BLOCKED. E10-C: NO_OOF_PASS_RETAIN_B. E10-D: NO_FEASIBILITY_PASS_RETAIN_B.

The constrained fusion sequence did not justify replacing B. Investigate background errors directly instead of unrestricted fusion searches.

## Entrypoints and prerequisites

Start in `SystemCode/model` with the historical `endolens-e2-win` environment and restored stage inputs. The following are original command files, ordered by filename. Check the stage guide before executing them; do not bypass NO_PASS or run test-application commands without an eligible locked configuration.

```text
01_check_and_freeze.cmd
02_compare_frozen_models.cmd
03_verify.cmd
```

Python, test and command-file bytes are preserved in the imported-source manifest. This English README is a navigation document, not part of the executed training recipe. Dataset images, checkpoints and full run/probability caches are not distributed in this source archive.
