# E1: Random-initialised U-Net baseline

[Experiment index](../../README.md) | [Protocol and evidence](../../docs/EXPERIMENT_STATUS.md)

## Research question

Measure what a basic U-Net learns without an ImageNet-pretrained encoder.

## Code packages

- [e1_mac](../../e1_mac/)
- [e1_windows](../../e1_windows/)

## Recorded result

Historical 798-entry Dice/IoU: 0.5707/0.4876.

## Interpretation and next decision

Investigate encoder initialisation rather than assume the decoder alone resolves generalisation.

## Reproduction prerequisites

Run from `SystemCode/model`, not this navigation directory. Restore the authorised dataset, signed split manifests, exact upstream checkpoints and cached outputs required by this stage. Read [research reproduction requirements](../../docs/REPRODUCIBILITY.md) before invoking entrypoints. Later stages depend on earlier frozen run artifacts; a source-only clone is not a complete experiment environment. Do not override a failed gate or resume a run against changed source fingerprints.

## Original entrypoints

Use the package-specific guide above for the original Python/macOS commands.
