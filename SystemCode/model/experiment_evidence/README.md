# Compact experiment evidence

These tables and selected decision fields were extracted from completed local records on 31 October 2026; no new evaluation was run. CSVs contain aggregate values, not per-patient images or predictions. Machine-specific checkpoint path fields, where present, are made relative or replaced with an external-artifact marker. Values are otherwise retained. `recorded_statuses.json` contains selected fields and original artifact hashes, not complete signed run artifacts and not inputs suitable for resuming experiments.

| Evidence | Interpretation |
|---|---|
| E5_summary.csv | Historical 798-entry and subset comparisons; 352 input |
| E8_three_seed_mean_sd.csv | Original/expanded x ordinary/difficult sampling, three-seed validation |
| E9_three_seed_mean_sd.csv | 70/30 revision versus retained controls; does not supersede failed gates |
| E10B_calibration_summary.csv | Fixed-weight probability fusion, calibration only |
| E10C_cv_summary.csv | Patient-grouped temperature CV; adaptively reused development data |
| E10D_diagnosis_summary.csv | Rescue feasibility; not a deployable oracle |
| E12_comparison.csv | Original common-protocol continuation report; not interchangeable with E13's matched 320 rotate4 comparison |
| recorded_statuses.json | Selected original status and protection fields, including failed and incomplete stages |

For final E13 matched metrics and controlled timing, use [the published full-precision comparison](../../../Miscellaneous/model_results/2026-10-31/README.md). E0-E5 historical metrics, E8 C6 identity metrics, E12 common-protocol metrics and final matched rotate4 metrics use different settings. Do not combine them into a single architecture ranking.
