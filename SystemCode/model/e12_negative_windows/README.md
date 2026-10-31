# E12: negative-data training and common model comparison

Research question: can verified empty-frame supervision reduce false foreground
on no-polyp images while preserving segmentation quality on positive images?

The Windows package uses the existing `endolens-e2-win` CUDA environment. It needs
PyTorch, torchvision, NumPy and Pillow, which are already installed. No OpenCV,
SciPy, new encoder download, package downgrade, or modification of earlier runs is
required. All new source files, logs and documentation are in English.

## What is compared

The three delivered starting models are frozen and evaluated with the same
validation list and output contract:

1. Original Polyp-PVT (E0), never trained in E12.
2. Original retained E8 B: ResNet34 U-Net, seed42 round98.
3. Teammate ResNet50/GN U-Net, supplied best.pt, historical round90.

The two U-Nets each receive two **matched 30-round continuations**:

| Entry | Training images drawn | Purpose |
|---|---|---|
| r34_positive | 100% positive | Control for extra training and the new common recipe |
| r34_mixed | 75% positive / 25% negative | Negative supervision effect in R34 |
| r50_positive | 100% positive | Control for extra training and the new common recipe |
| r50_mixed | 75% positive / 25% negative | Negative supervision effect in R50 |

Both arms of a model start from exactly the same supplied weights, with new
AdamW states. The final table has seven entries: three frozen baselines and four
continuations. This is not a clean architecture-only ablation: the supplied R34
and R50 weights have different training histories. A common dataset does not
erase inherited training exposure.

The first complete screen uses continuation seed42. CLI support for seeds17 and
2024 is provided for later matched continuation repeats. Those are not independent
retraining of the original ImageNet-to-final histories. Do not claim three-seed
evidence from the initial screen.

## Verified data on this computer

All E8 positive identities and file hashes are retained. The teammate manifests
were checked against the same positive identities and hashes, despite their Mac
paths. The original image/mask files are not renamed or edited.

| Split | Positive | Negative sequence | Negative single | Negative total | Total |
|---|---:|---:|---:|---:|---:|
| Train | 2,269 | 1,789 | 118 | 1,907 | 4,176 |
| Validation | 494 | 731 | 2 | 733 | 1,227 |
| Historical C6 evaluation | 83 | 1,755 | 5 | 1,760 | 1,843 |

Positive training remains original1151 + PolypGen C1/C2/C3/C4
251/267/456/144. Positive validation remains original290 + C5_204. The original
83-positive C6 list stays fixed and its Dice is reported separately.

Negative labels have two independent forms of evidence:

- **125 single frames:** their present author mask is empty at the existing
  grayscale threshold >127.5, and their author bounding-box file is also empty.
  The bbox filename convention varies by center; both documented original forms
  are paired explicitly. C1-C4 train, C5 validate, C6 evaluate.
- **4,275 frames in 23 authored `sequenceData/negativeOnly/seq*_neg` folders:**
  the original README and paper identify these as no-polyp sequences. Their
  native-size all-zero PNG masks are generated in the new run directory. A
  missing mask in an arbitrary folder is never interpreted as a negative label.

Official Figure7 assigns seq1-10 to training centers C1/C4, seq11-13 to C5, and
seq14-23 to C6. Whole sequences stay in one role. The figure labels seq7_neg C1
while Figure2's aggregate C1/C4 totals differ by its 207 frames. This is recorded
instead of silently inferring a new center identity; both possible centers are
training centers, and negative sampling is image-uniform, so role membership and
sampling are unaffected.

No positive-sequence frames are added in this first experiment. Original
quarantined near duplicates are not restored to the positive training list.
Protected positives (including historical798 and FAR images) and all negative
candidates are checked for decoded RGB equality and the frozen E8 near-image
heuristic before inference. The active final audit has zero negative quarantines.
These image/sequence checks do not prove complete patient independence.

Active data: `runs/e12_negative_data_seed42/`. Use its train.json,
validation.json and test.json, not recursive directory scanning. A preserved
preliminary data draft is not an experiment input.

## Common train and inference protocol

- Primary input: **352 x 352 for every model**, RGB Pillow bilinear resize,
  /255 and ImageNet normalization exactly once. R50 embeds normalization in its
  network; R34 and PVT receive it through the common adapter.
- Training: 30 rounds, 1,160 replacement draws per round, microbatch2,
  accumulation4, 145 planned updates per round. Actual updates and AMP skips
  are logged. Both U-Net encoders and decoders are trainable; encoder BN running
  statistics are frozen for both, while BN affine parameters remain trainable.
- AdamW LR1e-5 cosine to1e-6, weight decay1e-4, CUDA AMP forward and FP32 loss.
- Common training augmentation: independent paired H/V flips p0.5, rotation
  U(-15,15), image brightness/contrast U(.85,1.15). No shrink/pad, saturation or
  gamma experiment is mixed into this comparison. Nearest-neighbor mask resize.
- Per-image loss: **BCEWithLogits + I(nonempty augmented GT) x soft Dice**,
  averaged over all images; soft Dice smooth1. Empty targets get BCE only. There
  is no boundary term in either arm. Weighted gradient accumulation reproduces
  the same effective-batch loss even when a microbatch is entirely negative.
- Primary evaluation: **FP32, no TTA, threshold0.5, no mask-component filter**.
  Resize logits to native GT dimensions before sigmoid. PVT uses the sum of its
  two original output logits. No per-image min-max and no repeated sigmoid.
- An optional secondary comparison applies the same rotate4 logit averaging to
  all seven entries at352. It cannot select a new winner or replace the primary
  table. The archived teammate score0.883889 used320+rotate4 and is a different
  protocol.

The common loss, resolution, augmentation and R34 BN policy differ from parts of
the historical recipes. Positive-only controls capture those continuation
changes; compare mixed to both its own frozen model and its positive control.

## Sampling precisely

For a positive-only control, source probability is original0.5 and each C1-C4
0.125, uniform among images within each source.

For the mixed arm, first assign 0.75 probability to the positive pool and0.25 to
the audited training-negative pool. Conditional positive proportions stay the
same: overall original0.375 and each C1-C4 positive source0.09375. The negative
pool is uniform over its1,907 eligible images. It is not center-balanced or
sequence-balanced. Sequence-equal evaluation is reported separately so long
sequences do not dominate the primary negative comparison.

Per-round expected draws are870 positive and290 negative, not fixed quotas for
each round or batch. Both arms use34,800 scheduled image draws over30 rounds;
the mixed arm replaces part of the positive exposure with negatives rather than
adding extra compute. `history.csv` records actual positive/negative counts and
sampling-index SHA256. The same arm/seed uses the same indices for both U-Nets.

## Checkpoint and final model selection

Every round evaluates **all1,227 validation images** without training augmentation.
Positive source macro Dice gives equal weight to original290 and C5_204. Negative
primary FPR is the arithmetic mean of the per-sequence any-foreground FPR over
the three C5 negative sequences. Negative single frames are reported separately.

An eligible checkpoint must protect its own frozen baseline under the common
protocol: positive macro Dice drop <=0.01, each positive source drop <=0.02,
low-Dice count increase <=2, zero-Dice increase <=1. Rank eligible checkpoints by
lowest negative sequence macro any-FPR, then lowest negative macro FP area, then
highest positive macro Dice. Ties retain the earlier checkpoint. **Epoch0 is
eligible**; training need not replace a better starting model.

After all four runs, final selection applies the same positive protection against
the strongest frozen **U-Net** positive macro-Dice baseline, then the same negative ranking.
PVT remains a comparison entry, not the protection anchor: original290 is not
verified unseen by its public training history. This is preparation revision
v1.1, made after initial common validation was viewed and before any formal
continuation or new C6 inference. The previous plan and baseline metadata are
preserved in pre_formal_protocol_revision/. Numeric gates, data, common inference
and training settings did not change. This is transparent development, not blind
preregistration.
The chosen entry is locked **on validation**, before test inference. Declaring a
negative-training gain additionally requires the mixed arm to reduce negative
any-FPR by at least0.10 absolute vs BOTH its own frozen baseline and positive
control, with no FP-area increase and with positive protection against both.
These are fixed research gates, not established clinical thresholds. A gate
failure is a valid result; do not change the gates after seeing C6.

## Run in Anaconda Prompt / CMD

```bat
conda activate endolens-e2-win
cd /d F:\Semester1\ISY5002-PracticeModule\Polyp-PVT\Polyp-PVT
e12_negative_windows\00_prepare_data.cmd
e12_negative_windows\01_verify_and_lock.cmd
e12_negative_windows\02_evaluate_frozen_validation.cmd
e12_negative_windows\03_train_all.cmd
e12_negative_windows\04_select_on_validation.cmd
e12_negative_windows\05_evaluate_common_test.cmd
e12_negative_windows\06_summarize.cmd
```

Preparation is idempotent: existing splits are verified, not silently rebuilt.
Verification runs19 unit tests plus real GPU forward/backward, empty-microbatch,
checkpoint roundtrip, intentional-interruption/resume and completed-run reuse
checks. All smoke results are clearly isolated and use train/validation only.
Preflight locks code, data, model hashes, runtime and protocol in plan.json.

Do not edit Python sources, plan.json or manifests after locking. The exact
restart command resumes the last committed round, including epoch0 recovery.
A same-host lock is reclaimed only after its recorded process is proven absent;
an active process blocks a second training. Completed runs are checked and skipped
without rewriting selected weights. Do not delete locks or launch parallel runs
on the single8GB GPU.

If running manually one experiment at a time, use03a,03b,03c,03d instead of03:

```bat
e12_negative_windows\03a_train_r34_positive.cmd
e12_negative_windows\03b_train_r34_mixed.cmd
e12_negative_windows\03c_train_r50_positive.cmd
e12_negative_windows\03d_train_r50_mixed.cmd
```

The status command is safe in another terminal:

```bat
e12_negative_windows\08_status.cmd
```

If Windows worker startup is troublesome, rerun an individual command with
`--workers 0`; the deterministic per-draw augmentation does not depend on worker
order. Use the exact same model/arm/seed. Do not change the locked batch budget.

Optional secondary TTA, only after the primary test and summary:

```bat
e12_negative_windows\07_optional_rotate4.cmd
```

## Where to follow progress and retrieve results

`runs/e12_negative_comparison_seed42/` contains:

- plan.json: locked experimental plan and source/environment fingerprints.
- baseline_validation/{pvt,r34,r50}/: original common validation results.
- training/{r34_positive,r34_mixed,r50_positive,r50_mixed}_seed42/:
  console.log, history.csv, run_info.json, last.pth and best.pth,
  best_validation.csv and best_validation_summary.json.
- selection.json and selected_model.json: validation-only locked candidate.
- test_identity/: seven identically evaluated historical C6 entries.
- report/EXPERIMENT_SUMMARY.md: readable experiment summary.
- report/comparison.csv: full positive/negative comparison table.
- report/test_negative_groups.csv: sequence-level negative outcomes.
- report/*.png: raw, unsmoothed validation trajectories.

Terminal output prints training progress every80 images, validation every100
images, positive Dice, negative FPR, FP area, selected round and elapsed time.
Checkpoints are complete states, not only weights; allow several GB of free disk
space. A best round of0 means the original checkpoint remained preferable.

To exercise the selected model on a new image after validation selection:

```bat
python -u e12_negative_windows\run_e12.py infer --image path_to_image.jpg --output runs\e12_preview\mask.png
```

The native binary PNG and adjacent JSON record the selected model and operating
point. This does not automatically overwrite the existing web release.

## Reading the results correctly

Report positive Dice/IoU/MAE, low/near-zero/zero Dice, negative any-foreground FPR,
FP image-area fraction, and fixed0.001/0.01 area-event FPR separately. An image
can have tiny FP area and still be a candidate frame under an any-foreground
rule. Area reduction alone does not solve candidate-count saturation.

Mixed Dice is supplementary: correctly empty negative images get Dice1 and can
inflate a pooled average. Do not substitute mixed Dice for the fixed83-positive
C6 Dice. The0.001/0.01 metrics are descriptive and do not remove mask pixels.

C6 has been repeatedly inspected, and R50 was supplied as the historical C6
winner. R50 inherited earlier contact with nine later-quarantined images. The
original290 internal validation split is not guaranteed unseen by E0 or all
inherited histories. These facts are retained in the plan/report; no new
independent-test or architecture-superiority claim is made. Validate a chosen
candidate on genuinely new complete videos before a final deployment claim.

## Sources

- [User-provided dataset overview](https://github.com/cloneiq/Awesome-GI-Endoscopy-Datasets/blob/main/datasets/2023-PolypGen/PolypGen_zh.md)
- [Author repository](https://github.com/DebeshJha/PolypGen)
- Ali et al., [PolypGen dataset paper](https://doi.org/10.1038/s41597-023-01981-y), Scientific Data10,75(2023).
- Original archive readme.md: negative sequence folders contain original images
  only; binary masks exist for single frames.
- Original paper Figure7, downloaded and visually checked; retained in
  resources/polypgen_fig7.png with its source URL in the data descriptor.

The archive's license files and previous provenance record remain unchanged.
The delivered source ZIP is preserved and checked against its own file manifest;
its machine-specific historical training scripts are not executed.
