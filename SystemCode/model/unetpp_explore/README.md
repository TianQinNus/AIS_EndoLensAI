# Experiment 17: validation-selected inference and matched continuation

Target: arithmetic mean of per-image binary Dice on all 798 known evaluation
images, at their native mask resolutions, at least 0.85. This set has already
been inspected repeatedly and is not an untouched confirmation set.

## Recovered baseline

- Checkpoint: `Desktop/EndoLensAI/SystemCode/model/unetpp_resolution/unetpp_context_long_20261022/best.pt`.
- SHA256: `7edabffc66b300cee08eeaa9caa5479d5781eaaf2c056a69d9a9aac9744d4020`.
- Epoch 130 of 150; stored validation Dice 0.9023209629387691.
- Reproduced validation Dice: 0.9023209666992134 (290 images).
- Architecture: U-Net++ with GroupNorm and residual dilated bottleneck context.
- Input 320; validation target 256; RGB /255; threshold 0.5.
- The workspace's historical `unetpp_resolution/train.py` does **not** contain
  the context block. `model.py` is copied from the adjacent project's frozen
  `polyp_outline/model.py`, loads all baseline checkpoint keys strictly, and
  reproduces the baseline. This is not a recovered byte-identical training source.

## Inference experiment

`evaluate_tta.py` verifies image/mask hashes and decoded RGB separation, compares
identity, four flips and four quarter-turns on validation, writes a locked
selection, then evaluates that one selection on the known set. It averages
aligned logits before native-resolution interpolation and sigmoid. It never
selects a transform, threshold, or checkpoint using evaluation labels.

All images have equal weight in the primary mean. Per-source scores and counts
of Dice below 0.5 / exactly zero are also saved. Validation uses the historical
256 target resolution for comparable checkpoint selection.

Run from the repository root:

```sh
python3 SystemCode/model/unetpp_explore/evaluate_tta.py \
  --checkpoint /Users/tianqin-gg/Desktop/EndoLensAI/SystemCode/model/unetpp_resolution/unetpp_context_long_20261022/best.pt \
  --output outputs/experiment17_inference --device mps
```

## Training experiment

`finetune.py` supports two matched 20-epoch arms initialized from the same
checkpoint, with a fresh AdamW optimizer, LR 1e-5 decreasing to 1e-6, seed 42,
and unchanged geometry, loss, input resolution and data splits.

- Control: original augmentation only.
- Photometric arm: additionally apply saturation U(0.5, 1.5) with p=0.5 and
  gamma U(0.7, 1.5) with p=0.5. This tests an imaging-appearance hypothesis;
  improvement is not assumed.

The unchanged initialization participates in checkpoint selection. Training
loads only the 1,160 training and 290 validation images. Each run saves source,
data and initialization hashes, history, best/last checkpoints and RNG states.
Use `--resume` with identical arguments to resume after a completed epoch.

```sh
python3 SystemCode/model/unetpp_explore/finetune.py \
  --checkpoint /Users/tianqin-gg/Desktop/EndoLensAI/SystemCode/model/unetpp_resolution/unetpp_context_long_20261022/best.pt \
  --output outputs/experiment17_control --device mps

python3 SystemCode/model/unetpp_explore/finetune.py \
  --checkpoint /Users/tianqin-gg/Desktop/EndoLensAI/SystemCode/model/unetpp_resolution/unetpp_context_long_20261022/best.pt \
  --output outputs/experiment17_photometric --device mps --photometric
```

Select the training arm using validation only, then lock its inference procedure
using validation before evaluating the known set. Do not describe a comparison
of continuation and photometric arms as a clean single-variable effect of
individual saturation or gamma changes. They form one combined policy.

## Verification and limits

- Inverse transform alignment checked with a pixelwise equivariant model.
- Empty-mask metric conventions checked.
- Both training arms completed a two-epoch synthetic CPU smoke, including
  checkpoint loading and completed-run resume. This does not test interrupted
  GPU numerical reproducibility or establish accuracy.
- All 34 historical imported Python sources pass the integrity check.
- Historical E0 and E1 tests pass; the repository regression runner stops at E2
  because this Python environment lacks torchvision.
- An initial inference run failed because validation manifests lack `dataset`;
  its failed protocol is preserved. The v2 run handles this schema.
- Full training is multi-hour on this Mac (historical mean approximately 459
  seconds per epoch); no training improvement or 0.85 achievement is claimed
  until actual completed evaluations establish it.
