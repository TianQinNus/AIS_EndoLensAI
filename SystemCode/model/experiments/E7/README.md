# E7: Constrained calibration and routing diagnosis

[Experiment index](../../README.md) | [Protocol and evidence](../../docs/EXPERIMENT_STATUS.md)

## Research question

Test affine calibration, MAE-constrained calibration, grouped cross-validation, and prediction-only shallow routing.

## Code packages

- [e7a_windows](../../e7a_windows/)
- [e7a2_windows](../../e7a2_windows/)
- [e7b0_windows](../../e7b0_windows/)
- [e7b1_windows](../../e7b1_windows/)
- [e7b2_windows](../../e7b2_windows/)

## Recorded result

E7-A: NO_PASS; E7-A2: NO_PASS; E7-B0: diagnostic headroom; E7-B1: OOF_NO_PASS; E7-B2 selection: NO_PASS.

## Interpretation and next decision

Do not promote oracle routing or a failed candidate. Turn to training-data expansion while keeping E0 frozen.

## Reproduction prerequisites

Run from `SystemCode/model`, not this navigation directory. Restore the authorised dataset, signed split manifests, exact upstream checkpoints and cached outputs required by this stage. Read [research reproduction requirements](../../docs/REPRODUCIBILITY.md) before invoking entrypoints. Later stages depend on earlier frozen run artifacts; a source-only clone is not a complete experiment environment. Do not override a failed gate or resume a run against changed source fingerprints.

## Original Windows entrypoints

The list documents the available entrypoints; it is not an unconditional batch to execute. Read the package instructions, run checks first, and stop when a selection or protection condition fails. Status, verification and optional analysis commands do not necessarily initiate training.

### e7a_windows

```powershell
.\e7a_windows\01_test_e7a.cmd
.\e7a_windows\02_select_e7a.cmd
.\e7a_windows\03_apply_locked_test_once.cmd
```

### e7a2_windows

```powershell
.\e7a2_windows\01_test_e7a2.cmd
.\e7a2_windows\02_select_e7a2.cmd
.\e7a2_windows\03_apply_locked_test_once.cmd
```

### e7b0_windows

```powershell
.\e7b0_windows\01_test_e7b0.cmd
.\e7b0_windows\02_run_diagnostics.cmd
```

### e7b1_windows

```powershell
.\e7b1_windows\01_test_e7b1.cmd
.\e7b1_windows\02_run_gate_cv.cmd
.\e7b1_windows\03_verify_outputs.cmd
```

### e7b2_windows

```powershell
.\e7b2_windows\01_test_e7b2.cmd
.\e7b2_windows\02_freeze_rule.cmd
.\e7b2_windows\03_evaluate_selection_once.cmd
.\e7b2_windows\04_verify_outputs.cmd
```

## Substage roles

| Substage | Role | Recorded outcome |
|---|---|---|
| E7-A | Post-fusion affine calibration | NO_PASS |
| E7-A2 | MAE-constrained revision | NO_PASS |
| E7-B0 | Error diagnosis and grouped calibration CV | Diagnostic headroom; oracle uses ground truth |
| E7-B1 | Prediction-only logistic gate CV | OOF_NO_PASS |
| E7-B2 | Frozen median-based shallow rule | Selection NO_PASS |
