# E0: Frozen Polyp-PVT reference

[Experiment index](../../README.md) | [Protocol and evidence](../../docs/EXPERIMENT_STATUS.md)

## Research question

Establish a frozen external segmentation reference before changing the in-house U-Net.

## Code packages

- [e0_mac](../../e0_mac/)

## Recorded result

Historical 798-entry Dice/IoU: 0.83678/0.76300. Matched C6 rotate4 values are different (0.84357/0.78978).

## Interpretation and next decision

Retain E0 as a reference, not a model retrained by this project.

## Reproduction prerequisites

Run from `SystemCode/model`, not this navigation directory. Restore the authorised dataset, signed split manifests, exact upstream checkpoints and cached outputs required by this stage. Read [research reproduction requirements](../../docs/REPRODUCIBILITY.md) before invoking entrypoints. Later stages depend on earlier frozen run artifacts; a source-only clone is not a complete experiment environment. Do not override a failed gate or resume a run against changed source fingerprints.

## Original entrypoints

Use the package-specific guide above for the original Python/macOS commands.
