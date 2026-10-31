# E10: Frozen B/E0 fusion feasibility

[Experiment index](../../README.md) | [Protocol and evidence](../../docs/EXPERIMENT_STATUS.md)

## Research question

Diagnose complementarity on FAR; test three fixed fusion weights, constrained temperature calibration, and local rescue feasibility.

## Code packages

- [e10a_windows](../../e10a_windows/)
- [e10b_windows](../../e10b_windows/)
- [e10c_windows](../../e10c_windows/)
- [e10d_windows](../../e10d_windows/)

## Recorded result

E10-B: NO_CALIBRATION_PASS_SELECTION_BLOCKED. E10-C: NO_OOF_PASS_RETAIN_B. E10-D: NO_FEASIBILITY_PASS_RETAIN_B.

## Interpretation and next decision

The constrained fusion sequence did not justify replacing B. Investigate background errors directly instead of unrestricted fusion searches.

## Reproduction prerequisites

Run from `SystemCode/model`, not this navigation directory. Restore the authorised dataset, signed split manifests, exact upstream checkpoints and cached outputs required by this stage. Read [research reproduction requirements](../../docs/REPRODUCIBILITY.md) before invoking entrypoints. Later stages depend on earlier frozen run artifacts; a source-only clone is not a complete experiment environment. Do not override a failed gate or resume a run against changed source fingerprints.

## Original Windows entrypoints

The list documents the available entrypoints; it is not an unconditional batch to execute. Read the package instructions, run checks first, and stop when a selection or protection condition fails. Status, verification and optional analysis commands do not necessarily initiate training.

### e10a_windows

```powershell
.\e10a_windows\01_check_and_freeze.cmd
.\e10a_windows\02_compare_frozen_models.cmd
.\e10a_windows\03_verify.cmd
```

### e10b_windows

```powershell
.\e10b_windows\01_test_and_freeze.cmd
.\e10b_windows\02_calibrate_and_lock.cmd
.\e10b_windows\03_evaluate_locked_selection.cmd
.\e10b_windows\04_verify.cmd
```

### e10c_windows

```powershell
.\e10c_windows\01_test_and_freeze.cmd
.\e10c_windows\02_patient_cv.cmd
.\e10c_windows\03_refit_if_pass.cmd
.\e10c_windows\04_locked_selection_if_pass.cmd
.\e10c_windows\05_verify.cmd
```

### e10d_windows

```powershell
.\e10d_windows\01_test_and_freeze.cmd
.\e10d_windows\02_infer_stability.cmd
.\e10d_windows\03_diagnose_patient_cv.cmd
.\e10d_windows\04_verify.cmd
```

## Substage roles

| Substage | Role | Recorded outcome |
|---|---|---|
| E10-A | Frozen B/E0 FAR comparison | Complementarity diagnosis only |
| E10-B | Three fixed E0 weights: 0.25, 0.50, 0.75 | Calibration NO_PASS; selection blocked |
| E10-C | Bounded temperature calibration, grouped CV | OOF NO_PASS |
| E10-D | Local rescue feasibility and stability features | Feasibility NO_PASS; retain B |
