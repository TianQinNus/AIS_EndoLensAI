# Team collaboration

1. `git pull --ff-only` before work; use a branch such as `feature/video-reader`, `feature/web-review` or `experiment/e5-equal-fusion`.
2. Keep existing E0–E4 Python code frozen: checkpoint recovery/preflight checks validate source hashes. For a new model or protocol, create a new experiment directory and record differences instead of silently changing old files.
3. Run from `SystemCode/model/`. After installation run `python scripts/verify_imported_code.py` and `python scripts/run_checks.py`; GPU changes also require Windows CUDA probe and smoke.
4. Commit code, dependency declarations, fixed configs/manifests and compact summaries. Never use `git add -f` to upload ignored data, weights, full `runs/`, credentials or private clinical uploads.
5. Each experiment needs a new output directory, model/source/data hashes, seed, initialization, training budget, selection rule and full metrics. Store complete runs separately; copy reviewed compact summaries to `Miscellaneous/model_results/` with provenance.
6. Open a pull request describing problem, behavior, commands checked and remaining uncertainty; a teammate reviews before merging. Do not force-push shared `master`.
7. The GitHub repository is for shared source and experiment documentation. Recover full artifact directories locally according to the asset guide; a CSV summary is not a reproducible checkpoint.
8. Video/UI work should use the fixed metric/model interfaces documented in `MODEL_PROTOCOL.md`. Label planned or unverified features explicitly; candidate frames are not unique lesions and reports describe system outputs.
