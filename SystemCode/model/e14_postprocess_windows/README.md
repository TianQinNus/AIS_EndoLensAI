# E14: postprocessing exploration on the E13 model

## Scope

The user explicitly requested an exploratory sweep on the already-inspected C6
test set. This package freezes E13 `mixed50` epoch94 and changes only output rules.
There is no retraining and no automatic web deployment. Test-informed operating
points must not be presented as independent validation or a generally optimal
deployment setting.

The original E13 files and model weights remain unchanged. Outputs are written
under `runs/e14_postprocess_seed42/`.

## Frozen inference contract

- Weight: `runs/e13_joint_seed42/training/mixed50/best_positive.pth`.
- SHA256: `1dadc0706f69069104f1395a3959ade36beca8be4be0b2c4b2cf821471a59088`.
- RGB /255, embedded ImageNet normalization exactly once.
- Input320, rotate4 inverse-aligned mean logits, FP32, TF32 disabled.
- Resize logits to native mask dimensions, sigmoid once, then output processing.
- Fixed83 positive and1,760 negative images from E12's signed C6 manifest.
- Every unfiltered prediction must exactly reproduce its E13 TP, FP, FN and
  predicted-pixel counts before it enters the analysis.

## Spatial rules and fixed parameter grid

| Rule | Values |
|---|---|
| Pixel threshold | 0.50, 0.60, 0.70, 0.80, 0.90 |
| Minimum connected-region area / image area | 0, 0.0001, 0.0005, 0.001, 0.002, 0.005 |
| Minimum region mean model probability | disabled, 0.70, 0.80, 0.90, 0.95 |

The complete Cartesian grid has150 settings, including baseline and all one-factor
comparisons. Regions use8-connectivity at the native image size. Area fractions
are percentages of the **whole image**, not percentages of the predicted mask;
0.001 means0.1% of the image. A region equal to the cutoff is retained.
All qualifying regions are retained; this is not a largest-component-only rule.

Region confidence is the mean original probability over that region's pixels
after pixel thresholding. It is an uncalibrated model score, not a probability
that the region is truly a polyp. The rules use no ground-truth information at
inference. Ground truth is used only to evaluate and illustrate tradeoffs.

Primary outputs are positive Dice/IoU, low/near-zero/zero-Dice counts, positive
empty-mask count and overlap-hit rate, negative any-foreground frame FPR, negative
FP image-area fraction and equal-sequence negative summaries. Small positive
targets (<1% of native image area) are reported separately. Correctly empty
negative masks never inflate the positive Dice headline.

Three descriptive operating points minimize negative frame any-FPR with positive
Dice drops no larger than0.005,0.01 or0.02 relative to E13, and at most one extra
zero-Dice positive. These are **test-informed illustrations**, not locked deployment
choices. All150 rows and the Dice/FPR Pareto flag are retained to show the tradeoff.

## Causal consecutive-frame confirmation

The C6 list has1,755 images in10 negative-only sequences and5 empty single images.
The83 positive images are static examples. Consequently, temporal confirmation
can be assessed for negative false foreground only. Temporal positive sensitivity,
positive Dice and detection delay in seconds cannot be estimated with this list.

For baseline and the illustrative spatial operating points:

1. Sort each sequence by the integer suffix of its filename. Plain lexical sorting
   is wrong for names such as `..._001`, `..._0010`, `..._00100`.
2. Reset tracks at sequence boundaries and at every missing numeric frame ID.
3. Match current and previous regions greedily, one-to-one, at bounding-box IoU
   >=0.1. This is a simple position-consistency heuristic, not an optical-flow tracker.
4. Require2 or3 consecutive observed frames before retaining a current-frame region.
5. Use only past/current information. Do not backfill earlier frames or look ahead.

Minimum confirmation delay is1 or2 observed frames; FPS/timestamps are not supplied,
so this is not converted to seconds. Results include both all-frame and post-warmup
negative FPR to expose the automatic empty startup frames. The negative-only
temporal denominator is1,755, not the1,760 used in spatial tables. Do not pair
unchanged static-positive Dice with temporal-negative FPR and claim both came from
a complete video evaluation.

## Run / resume

Use the existing `endolens-e2-win` environment. OpenCV4.10 is already available in
the local `.handoff_test_deps` directory and is loaded only if it is absent from the
environment. No package installation or environment change is required.

```powershell
conda activate endolens-e2-win
Set-Location 'F:\Semester1\ISY5002-PracticeModule\Polyp-PVT\Polyp-PVT'
.\e14_postprocess_windows\01_run_postprocessing.cmd
```

The command runs frozen inference once, caches native connected-component
sufficient statistics and FP32 mean logits, then evaluates all rules. It can resume
from completed, plan-bound samples. Cache size is approximately0.8GB for logits
plus small component-statistic records. No formal training process is started.

After the cache is complete, rebuild the analysis without new inference:

```powershell
.\e14_postprocess_windows\02_rebuild_analysis.cmd
```

Do not edit the locked Python sources or grids and then reuse the same run.
Use a new experiment/revision directory for a changed protocol.

## Result files

- `plan.json`: signed grid, original weight/data identities and exploratory scope.
- `cache_status.json`: completion and exact-baseline reproduction count.
- `comparison.csv` / `comparison.json`: all150 spatial settings.
- `EXPERIMENT_SUMMARY.md`: one-factor results, combined operating points and
  temporal feasibility results in readable tables.
- `exploratory_operating_points.json`: descriptive choices and explicit test-use warning.
- `operating_points/*/per_image.csv`: paired baseline/operating-point counts.
- `temporal_comparison.csv`: negative-sequence confirmation outcomes.
- `temporal_per_sequence.csv` / `temporal_per_frame.csv`: complete causal decisions.
- `sequence_order_audit.json`: numeric ranges and missing-ID boundaries.

## Verification

```powershell
python e14_postprocess_windows/test_postprocess.py
```

Tests cover exact baseline reconstruction from region counts,8-connectivity,
native-area and mean-score decisions, empty masks, Dice gain by removal of false
foreground, causal startup, missing-frame resets, different-location rejection and
one-to-one track matching. The real inference pass additionally checks all1,843
unfiltered predictions against the original E13 records.
