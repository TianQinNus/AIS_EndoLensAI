# e11_background_windows: Difficult-background pixel weighting

[Complete E11 stage guide](../experiments/E11/README.md) | [Reproduction requirements](../docs/REPRODUCIBILITY.md)

## Purpose

Add detached probability-weighted background BCE to the existing loss while retaining positive-only B data and sampling.

## Recorded outcome

The available local run record still reports TRAINING; it does not establish a completed three-seed result. No final improvement is asserted here.

Preserve the hypothesis and source; separate background pixels within positive images from genuinely negative full frames.

## Entrypoints and prerequisites

Start in `SystemCode/model` with the historical `endolens-e2-win` environment and restored stage inputs. The following are original command files, ordered by filename. Check the stage guide before executing them; do not bypass NO_PASS or run test-application commands without an eligible locked configuration.

```text
00_review_cached_errors.cmd
01_test_and_prepare.cmd
02_smoke.cmd
03_train_all.cmd
04_summarize.cmd
05_status.cmd
06_verify_summary.cmd
```

Python, test and command-file bytes are preserved in the imported-source manifest. This English README is a navigation document, not part of the executed training recipe. Dataset images, checkpoints and full run/probability caches are not distributed in this source archive.
