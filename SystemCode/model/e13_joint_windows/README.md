# E13: ResNet50 U-Net trained jointly on positive and negative data from initialization

## Research question and scope

Can an ImageNet-initialized encoder and randomly initialized decoder learn polyp
segmentation and empty-frame rejection together, without inheriting a positive-only
segmentation training history?

The default command runs **one model**, `mixed50`, for 100 epochs. It does not run
the four E12 continuations. Both positive and verified negative images are eligible
from epoch 1. No previously trained segmentation checkpoint is loaded into training.

The teammate's delivered model is the historical positive-only pipeline comparator.
Its multiple training stages and historical C6 selection differ from this one-stage
run. This comparison cannot isolate a pure negative-data effect or prove superiority
of early joint training over later fine-tuning. An **optional** `positive` arm uses
the identical E13 initial weight, recipe and budget and gives a matched intervention
control. It is never started automatically.

## What stays the same as the teammate's recipe

| Item | E13 setting |
|---|---|
| Network | Supplied `UNet` implementation: ResNet50 encoder, U-Net decoder |
| Encoder initialization | Official torchvision ImageNet1K V2 |
| Decoder initialization | Random, seed 42 |
| Parameters | 32,520,977 |
| Encoder features | 64, 256, 512, 1024, 2048 |
| Decoder widths | 256, 128, 64, 32, 16 |
| Decoder blocks | Bilinear upsampling, skip concatenation, two Conv3x3/GN8/ReLU operations |
| Output | One-channel logits; single sigmoid at inference |
| Input | 320 x 320 RGB, /255, embedded ImageNet normalization exactly once |
| Trainability | Encoder and decoder train; encoder BN running statistics frozen, BN affine trainable |
| Optimizer | Fresh AdamW; weight decay 1e-4 |
| Effective batch | 8, implemented as microbatch 2 and four weighted accumulations |
| Training length | 100 epochs |
| Learning rate | 1e-4 -> 1e-5, cosine, endpoints included |
| Draws per epoch | 1,160 with replacement; not a full pass over 4,176 images |
| Validation | Input320, mask256, identity, threshold0.5 |
| Main C6 comparison | Input320, original mask dimensions, rotate4 logit averaging, threshold0.5 |

The learning rate follows the teammate's **initial ImageNet-to-segmentation** stage,
not her final 3e-5 continuation or E12's 1e-5 fine-tuning. The new experiment is one
100-epoch joint stage; it does not silently reproduce the previous four-stage history.
The full later-stage augmentation set is active from epoch 1, as requested for a
single joint recipe. Windows CUDA FP32 replaces the teammate's Mac MPS FP32;
no AMP, boundary-loss removal, network change or new image augmentation is added.

### Augmentation order

1. Resize RGB and target to 320 first (bilinear RGB, nearest-neighbor target).
2. Independent horizontal/vertical flips, probability 0.5 each.
3. Random 0/90/180/270-degree rotation.
4. Brightness and contrast factors U(0.8,1.2).
5. Shrink + black padding, probability 0.5, scale U(0.4,0.8), random placement.
6. Saturation, probability 0.5, factor U(0.5,1.5).
7. Gamma, probability 0.5, U(0.7,1.5).

Geometric operations are paired. Color operations affect the image only.
The original geometric operations and original loss are imported from the supplied
reference code and checked for numerical equivalence. The unrelated historical
U-Net++ class in that reference file is **not** instantiated; the ResNet50 `UNet`
is instantiated from `model/resnet50_model.py`.

### Original loss, including empty targets

```text
BCEWithLogits + per-image soft Dice(smooth=1)
              + 0.2 * inner morphological boundary Dice(smooth=1)
```

The boundary uses 3x3 erosion with explicit zero outside the image. All terms are
computed with the **original formula on all images**, including negative images.
The smoothing makes all-empty microbatches finite. This deliberately preserves the
teammate's mathematics. It differs from E12, which removed boundary loss and
disabled Dice on empty targets. Therefore E13 versus E12 is a pipeline comparison,
not an isolated initialization-only or ratio-only ablation.

## Fixed data membership

Reuse the audited E12 manifests; no additional images are downloaded or moved.

| Role | Positive | Negative sequences | Negative singles | Total |
|---|---:|---:|---:|---:|
| Training | 2,269 | 1,789 | 118 | 4,176 |
| Validation | 494 | 731 | 2 | 1,227 |
| Historical C6 evaluation | 83 | 1,755 | 5 | 1,843 |

Positive training: original1151 + C1 251 + C2 267 + C3 456 + C4 144.
Positive validation: original290 + C5 204. Positive C6: the same fixed83 images.
Training negatives are authored seq1-10 and verified empty singles C1-C4;
validation negatives are seq11-13 and C5 singles; C6 contains seq14-23 and C6 singles.

Manifests: `runs/e12_negative_data_seed42/{train,validation,test}.json`.
Negative labels require present empty mask plus empty author bbox, or explicit
author `negativeOnly` sequence provenance; missing labels are never used as negatives.
Preparation verifies train/validation image/mask/bbox byte hashes and all split
manifest signatures. C6 images and metrics are not opened by preparation or training.
The prior image similarity audit remains applicable; it does not prove patient
independence. The original nine quarantined positives are not restored, and the
new training initialization has not inherited the teammate's earlier exposure to them.

## Sampling: 50/50 means positive versus negative

The default arm has **50% probability for positive images and 50% for negative
images**. This is not the old original-versus-expanded-data 50/50 terminology.
Conditional on a positive draw, retain original-source50% and each C1-C4 12.5%.

| Stratum | Unconditional probability |
|---|---:|
| Original positive training | 25.00% |
| Each positive center C1-C4 | 6.25% each |
| All audited training negatives | 50.00% |

Negative sampling is image-uniform over 1,907 negatives, preserving E12's policy;
it is not newly center- or sequence-balanced. Adjacent frames remain correlated.
Counts are stochastic, not a per-microbatch quota. Expect about 580 positive and
580 negative draws each epoch, 116,000 total draws and 14,500 updates over 100
epochs. Actual class counts and per-epoch draw-index hashes are logged.
This is a **fixed-compute** experiment; it halves expected positive exposure versus
the positive-only control. Keeping positive exposure fixed would require a separate
larger-budget experiment. Fifty percent is a false-positive-oriented experimental
choice, not a proven optimal mixture.

## Validation selection: preserve both views of the result

The supplied R50 is evaluated at **input320 / mask256 / identity** on the 494 positive
and 733 negative validation images before training. Those scores set a frozen
reference; neither historical C6 numbers nor E12's 352 results set the thresholds.

Two checkpoint tracks are saved:

1. `best_positive.pth`: maximum positive **per-image mean Dice**, matching the
   teammate's original positive selection criterion. Negative images do not inflate it.
2. `best_tradeoff.pth`: minimum negative **sequence-equal any-foreground FPR**, then
   minimum sequence-equal FP area, then maximum positive Dice, among candidates with:
   - Positive mean Dice no more than **0.02** below the supplied R50 reference.
   - Neither original290 nor C5_204 Dice more than **0.03** below its reference.
   - No more than **5 additional** Dice<0.5 positive cases.
   - No more than **3 additional** Dice=0 positive cases.

These are new E13 negative-priority exploratory tolerances, chosen before E13
training. They are not unchanged E12 gates. They allow modest mean loss in exchange
for reduced false foreground, while recording severe failures explicitly. They are
not clinically established operating criteria. A zero-Dice output can be nonempty
and spatially wrong; it is not necessarily an empty prediction.

`03_select_on_validation.cmd` chooses the tradeoff checkpoint if one exists. If none
exists, it explicitly labels the result `NO_PROTECTED_TRADEOFF` and retains the
best-positive model only as a fallback comparison. There is no random epoch-zero
fallback. The run is still summarized/evaluated; the fallback is not called a
successful negative-priority model. Never select using mixed positive/negative Dice.

## Commands to run in the existing terminal

```bat
conda activate endolens-e2-win
cd /d F:\Semester1\ISY5002-PracticeModule\Polyp-PVT\Polyp-PVT

.\e13_joint_windows\00_prepare.cmd
.\e13_joint_windows\01_smoke_test.cmd
.\e13_joint_windows\02_train_joint.cmd
.\e13_joint_windows\03_select_on_validation.cmd
.\e13_joint_windows\04_evaluate_common_c6.cmd
.\e13_joint_windows\05_summarize.cmd
```

In PowerShell, use `Set-Location` instead of `cd /d` when changing directories.
Run commands sequentially; wait for the preceding command to exit successfully.
Preparation and a disposable smoke test may already be completed by Codex. Their
repeat is safe: preparation retains the signed plan, and smoke only touches its
own disposable outputs. **The 100-epoch training is left for the user to start.**

The encoder has been downloaded separately from the official URL:
https://download.pytorch.org/models/resnet50-11ad3fa6.pth

Full SHA256 (verified before initialization):

```text
11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca
```

The run uses the local encoder file and does not depend on a later automatic
torchvision download. `initial.pth` preserves the exact full initial network for
the main arm and optional matched positive control.

### What progress looks like

```text
EPOCH 1/100 arm=mixed50 lr=0.0001; ImageNet initialization; FP32
  train 80/1160 loss=... updates=10/145
...
[eval e13/identity] 1227/1227 ...s
EPOCH_DONE {"epoch": 1, "val_positive_dice": ..., ...}
```

Useful log fields: `val_positive_dice`, `val_positive_iou`, `val_zero_dice`,
`val_negative_frame_fpr`, `val_negative_seq_macro_fpr`, `eligible_tradeoff`,
`best_positive_epoch`, `best_tradeoff_epoch`. A null tradeoff epoch early in
training is normal: a new decoder may not yet reach the positive reference floor.

Inspect progress in a second terminal with `06_status.cmd`. For live logs:

```powershell
Get-Content .\runs\e13_joint_seed42\training\mixed50\console.log -Tail 20 -Wait
```

If interrupted, wait for the previous process to exit and run **the same**
`02_train_joint.cmd` again. The last completely committed epoch, optimizer and RNG
state are restored; an interrupted epoch is replayed. Active same-host process
locks block duplicate jobs; do not manually delete a live lock. If Windows worker
startup causes problems, use `python -u e13_joint_windows/run_e13.py train --arm mixed50 --workers 0`;
the deterministic per-sample augmentation/draw policy is unchanged. Completed
training commands verify the best hashes and return without modifying weights.

Reserve approximately a few GB for initial, last, best and smoke checkpoints.
No previous E8/E12 checkpoint or web model is overwritten.

## Evaluation and interpretation

`04_evaluate_common_c6.cmd` evaluates frozen PVT, frozen R34 B, supplied R50 and
the available E13 best-positive/best-tradeoff weights under the **same320 rotate4
native-mask protocol**. It may take substantially longer than identity inference.
Each completed output is hash-verified and reused if the command is repeated.
The optional identity command provides a secondary same-weight comparison;
it does not change the locked selection.

This evaluation includes the same 83 C6 positive images and 1,760 negative images.
Always report positive Dice/IoU and low/near-zero/zero counts separately from
negative any-FPR and FP-area fraction. Report frame-weighted and sequence-equal
negative outcomes separately. Correctly empty negatives must not inflate the
headline segmentation Dice. This experiment does not test temporal tracking,
clinical diagnostic performance or processing FPS.

Results are exploratory because C6 has already been inspected. The historical
supplied R50 score0.88389 used320/rotate4 and came from a different training history;
recomputed common evaluation is the comparison evidence. E12's352/identity numbers
must not be copied directly into the E13 comparison. Keep C5/C6 out of training.

## Optional matched positive-only control

After or before the main run, execute `optional_train_positive_control.cmd` if a
strict same-start comparison is wanted. It uses `initial.pth`, 100 epochs, the same
loss, augmentation, optimizer, draw budget and validation policy; only negative
sampling mass changes from0.5 to0. It has its own output folder.

```bat
python -u e13_joint_windows\run_e13.py select --arm positive
python -u e13_joint_windows\run_e13.py evaluate --arm positive --tta rotate4
python -u e13_joint_windows\run_e13.py report --arm positive
```

## Output locations and checks

- `runs/e13_joint_seed42/plan.json`: signed recipe, data/code/encoder/initial hashes,
  validation reference and comparison disclosure.
- `runs/e13_joint_seed42/initial.pth`: official ImageNet encoder + random decoder.
- `runs/e13_joint_seed42/reference_validation/`: supplied R50 validation evidence.
- `runs/e13_joint_seed42/training/mixed50/`: best weights, last recovery state,
  history.csv, console.log, run_info.json, selection.json and validation evidence.
- `runs/e13_joint_seed42/test/mixed50/rotate4/`: common C6 outputs for every entry.
- `runs/e13_joint_seed42/report/mixed50/`: comparison.csv and EXPERIMENT_SUMMARY.md.
- `runs/e13_joint_seed42/smoke/`: disposable checks, not formal experiment results.

Validation tests:

```bat
python e13_joint_windows\test_e13.py
```

Tests check exact original loss/augmentation parity, finite all-negative gradients,
effective-batch accumulation, 50/50 empirical sampling and source proportions,
fixed split roles, deterministic real negative augmentation, exact official encoder
loading, frozen BN statistics and tradeoff ranking. The separate CUDA smoke checks
real positive/negative/mixed updates and checkpoint roundtrip without touching formal
training checkpoints or C6 data.

After the plan is locked, editing locked Python sources or the training recipe is
rejected rather than silently continuing a changed experiment. Use a new stage/run
for later revisions.
