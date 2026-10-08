#!/usr/bin/env python3
"""E2: validation-guided U-Net transfer learning, CUDA-only formal training.

All paths supplied to this CLI are resolved relative to --project when relative.
The package is self-contained. It never imports the original PVT model.
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
from data import local_path, prepare_bundle, load_bundle, signature
from engine import (PACKAGE, PROTOCOL, code_hashes, environment, require_cuda, execute_training,
                    evaluate_records, compare_reference, load_checkpoint)
from models import WEIGHT_FILENAME, WEIGHT_URL, make_model, read_pretrained
from e0_protocol import file_hash, group_metrics, write_csv, write_json


def resolve_project(value):
    candidates = [Path(value).expanduser().resolve()] if value else [Path.cwd(), PACKAGE.parent]
    for p in candidates:
        if (p / "dataset/TrainDataset/images").is_dir():
            return p.resolve()
    raise FileNotFoundError("Run in the project root or supply --project with TrainDataset/images")


def make_config(args):
    return {"experiment": args.experiment, "phase": "development", "epochs": 1 if args.smoke else args.epochs,
            "schedule_horizon": args.epochs, "warmup_epochs": min(5, max(0, args.epochs - 1)),
            "input_size": 352, "batch_size": args.batch_size, "accumulation": args.accumulation,
            "lr": 1e-4, "weight_decay": 1e-4, "seed": args.seed, "amp": not args.no_amp,
            "augmentation": args.augmentation, "base_channels": 64, "smoke": args.smoke,
            "encoder_weights": args.encoder_weights.replace("\\", "/") if args.experiment == "pretrained" else None,
            "loss": "BCE + per-image soft Dice, coefficients 1+1, smooth=1",
            "scheduler": "linear warmup then cosine decay to 0.1 initial LR; full horizon preserved in final training",
            "encoder_norm": "none" if args.experiment == "original" else "torchvision BatchNorm, trainable affine and running stats",
            "decoder_norm": "none" if args.experiment == "original" else "GroupNorm(8)",
            "validation_selection": "maximum original-size binary Dice, threshold 0.5, earliest tie",
            "precision": "AMP float16 forward, FP32 loss and validation" if not args.no_amp else "float32"}


def select_recipe(project, runs, output):
    candidates = []
    snapshots = []
    for value in runs:
        path = local_path(project, value)
        info = json.loads((path / "run_info.json").read_text(encoding="utf-8"))
        config = info["config"]
        if info["status"] != "TRAINED" or info["run_kind"] != "E2_DEVELOPMENT" or config["smoke"]:
            raise ValueError("Selection requires completed formal development runs, never smoke/final/test runs")
        if info["source_hashes"] != code_hashes():
            raise ValueError("Development source differs from current E2 package")
        best_path = path / "checkpoints/best.pth"
        if file_hash(best_path) != info["best_checkpoint_sha256"]:
            raise ValueError("Best development checkpoint changed")
        best = load_checkpoint(best_path)
        if best["epoch"] != info["best_epoch"] or best["bundle_signature"] != info["bundle_signature"] or best["config"] != config:
            raise ValueError("Development checkpoint provenance mismatch")
        candidates.append({"run": path.relative_to(project).as_posix(), "experiment": config["experiment"],
                           "augmentation": config["augmentation"], "best_epoch": info["best_epoch"],
                           "validation_dice": info["best_validation_dice"], "config": config,
                           "best_checkpoint_sha256": info["best_checkpoint_sha256"],
                           "initial_encoder_sha256": info.get("initial_encoder_sha256")})
        snapshots.append(info["bundle_signature"])
    if len(set(snapshots)) != 1:
        raise ValueError("All development runs must use the same internal data split")
    if len({r["config"]["seed"] for r in candidates}) != 1:
        raise ValueError("Compare matching seeds first")
    chosen = sorted(candidates, key=lambda r: (-r["validation_dice"], r["run"]))[0]
    final_config = dict(chosen["config"], phase="final", epochs=chosen["best_epoch"])
    # Keep development schedule_horizon, do NOT compress cosine to the selected
    # shorter epoch budget. No continuation from the development best weights.
    recipe = {"format_version": 1, "selection_basis": "internal validation Dice only; no test results",
              "bundle_signature": snapshots[0], "source_hashes": code_hashes(), "candidates": candidates,
              "selected_run": chosen["run"], "config": final_config,
              "initial_encoder_sha256": chosen["initial_encoder_sha256"],
              "final_training_rule": "reinitialize and train all 1450 pairs for the preselected number of epochs"}
    recipe["recipe_signature"] = signature(recipe)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError("Selection output already exists; preserve it and choose another filename")
    write_json(output, recipe)
    write_csv(output.with_suffix(".csv"), [{k: v for k, v in c.items() if k not in ("config", "initial_encoder_sha256")} for c in candidates])
    print(f"Selected {chosen['experiment']}, validation Dice={chosen['validation_dice']:.4f}, final epoch budget={chosen['best_epoch']}. Recipe: {output}")


def evaluate_final(project, run, device_name, e0_run, e1_evaluation):
    if device_name == "cuda":
        device = require_cuda()
    elif device_name == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS unavailable; explicitly select CPU for analysis")
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    info = json.loads((run / "run_info.json").read_text(encoding="utf-8"))
    if info["run_kind"] != "E2_FINAL" or info["status"] != "TRAINED":
        raise ValueError("Test evaluation requires a completed final full-training run. Development/smoke test evaluation is blocked.")
    if info["source_hashes"] != code_hashes():
        raise ValueError("Evaluation code differs from the immutable training snapshot")
    cp_path = run / "checkpoints/final.pth"
    if file_hash(cp_path) != info["final_checkpoint_sha256"]:
        raise ValueError("Final model checkpoint changed")
    cp = load_checkpoint(cp_path)
    if cp["config"] != info["config"] or cp["epoch"] != info["config"]["epochs"]:
        raise ValueError("Invalid final epoch/configuration")
    test = json.loads((run / "test_manifest.json").read_text(encoding="utf-8"))
    bundle = json.loads((run / "bundle_snapshot.json").read_text(encoding="utf-8"))
    if signature(test) != bundle["e0_test_signature"] or cp["bundle_signature"] != bundle["bundle_signature"]:
        raise ValueError("Frozen final manifest/checkpoint mismatch")
    if len(test) != 798:
        raise ValueError("Final test set must contain all 798 entries")
    for r in test:
        for kind in ("image", "mask"):
            if file_hash(project / r[kind]) != r[kind + "_sha256"]:
                raise ValueError("Test file changed: " + r["sample_id"])
    e0 = local_path(project, e0_run)
    e0_info = json.loads((e0 / "run_info.json").read_text(encoding="utf-8"))
    if (e0_info.get("status") != "COMPLETED" or e0_info.get("run_kind") != "FULL_E0" or
            file_hash(e0 / "manifest.json") != e0_info.get("manifest_sha256") or
            signature(json.loads((e0 / "manifest.json").read_text(encoding="utf-8"))) != signature(test)):
        raise ValueError("Local E0 reference incompatible with final test manifest")
    e1_reference = None
    if e1_evaluation:
        e1_reference = local_path(project, e1_evaluation)
        e1_info = json.loads((e1_reference / "run_info.json").read_text(encoding="utf-8"))
        if e1_info.get("status") != "COMPLETED" or e1_info.get("processed_count") != 798 or e1_info.get("test_signature") != signature(test):
            raise ValueError("Supply the completed 798-entry E1 evaluation directory")
    output = run / ("evaluation_" + device_name + "_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    output.mkdir(exist_ok=False)
    record = {"status": "RUNNING", "expected_count": 798, "checkpoint_sha256": file_hash(cp_path),
              "protocol": PROTOCOL, "environment": environment(device), "batch_size": 1, "warmup": 3,
              "training_environment": info["environment"], "bundle_signature": bundle["bundle_signature"],
              "timing": "FP32 forward synchronized, excludes pre/postprocessing, metrics and file writes"}
    write_json(output / "run_info.json", record)
    try:
        model = make_model(cp["config"], initialization=False)
        model.load_state_dict(cp["model_state"], strict=True)
        model.to(device).eval()
        del cp
        summary, rows = evaluate_records(model, project, test, device, info["config"]["input_size"], output, warmup=3)
        groups = group_metrics(rows, include_legacy=False)
        write_csv(output / "summary.csv", groups)
        write_json(output / "metrics.json", {"protocol": PROTOCOL, "processed_count": len(rows), "groups": groups})
        compare_reference(rows, e0 / "per_image.csv", "e0", output)
        if e1_reference:
            compare_reference(rows, e1_reference / "per_image.csv", "e1", output)
        record.update(status="COMPLETED", processed_count=len(rows), summary=summary,
                      analysis_loop_timing="decode, preprocess, forward, postprocess, metrics and writes; excludes model load/audit/warmup")
        write_json(output / "run_info.json", record)
        for r in groups:
            print(f"{r['dataset']:20s} n={r['count']:3d} Dice={r['mean_dice']:.6f} IoU={r['mean_iou']:.6f}")
        print("Evaluation:", output)
    except BaseException as error:
        record.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        write_json(output / "run_info.json", record)
        raise


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("prepare", "download-weights", "train", "resume", "select", "train-final", "evaluate"):
        s = sub.add_parser(name)
        s.add_argument("--project")
        s.add_argument("--threads", type=int, default=4)
        if name == "prepare":
            s.add_argument("--e0-run", required=True)
            s.add_argument("--output", default="runs/e2_data_seed42")
            s.add_argument("--seed", type=int, default=42)
            s.add_argument("--metadata-csv", help="Optional complete sample_id,source,group_id table")
        elif name == "download-weights":
            s.add_argument("--output", default="e2_weights/" + WEIGHT_FILENAME)
        elif name == "train":
            s.add_argument("--experiment", choices=("original", "random", "pretrained"), required=True)
            s.add_argument("--bundle", default="runs/e2_data_seed42")
            s.add_argument("--output", required=True)
            s.add_argument("--device", choices=("cuda",), default="cuda")
            s.add_argument("--epochs", type=int, default=100)
            s.add_argument("--batch-size", type=int, default=2)
            s.add_argument("--accumulation", type=int, default=4)
            s.add_argument("--workers", type=int, default=2)
            s.add_argument("--seed", type=int, default=42)
            s.add_argument("--augmentation", choices=("flips", "strong"), default="flips")
            s.add_argument("--encoder-weights", default="e2_weights/" + WEIGHT_FILENAME)
            s.add_argument("--smoke", action="store_true")
            s.add_argument("--no-amp", action="store_true")
        elif name == "resume":
            s.add_argument("--run-dir", required=True)
            s.add_argument("--bundle", default="runs/e2_data_seed42")
            s.add_argument("--workers", type=int, default=2)
            s.add_argument("--device", choices=("cuda",), default="cuda")
        elif name == "select":
            s.add_argument("--runs", nargs="+", required=True)
            s.add_argument("--output", default="runs/e2_selection_seed42.json")
        elif name == "train-final":
            s.add_argument("--recipe", default="runs/e2_selection_seed42.json")
            s.add_argument("--bundle", default="runs/e2_data_seed42")
            s.add_argument("--output", default="runs/e2_final_seed42")
            s.add_argument("--device", choices=("cuda",), default="cuda")
            s.add_argument("--workers", type=int, default=2)
        elif name == "evaluate":
            s.add_argument("--run-dir", default="runs/e2_final_seed42")
            s.add_argument("--device", choices=("cuda", "cpu", "mps"), default="cuda")
            s.add_argument("--e0-run", required=True)
            s.add_argument("--e1-evaluation")
    a = p.parse_args()
    for field in ("threads", "epochs", "batch_size", "accumulation"):
        if hasattr(a, field) and getattr(a, field) < 1:
            p.error(field + " must be positive")
    if hasattr(a, "workers") and a.workers < 0 or hasattr(a, "seed") and a.seed < 0:
        p.error("workers/seed must be nonnegative")
    return a


def main():
    a = parse_args()
    torch.set_num_threads(a.threads)
    project = resolve_project(a.project)
    if a.command == "prepare":
        prepare_bundle(project, local_path(project, a.output), a.e0_run, a.seed, a.metadata_csv)
    elif a.command == "download-weights":
        path = local_path(project, a.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            torch.hub.download_url_to_file(WEIGHT_URL, str(path), hash_prefix="b627a593", progress=True)
        _, digest = read_pretrained(path)
        print("Official ImageNet encoder weights:", path, "SHA256:", digest)
    elif a.command == "select":
        select_recipe(project, a.runs, local_path(project, a.output))
    elif a.command == "evaluate":
        evaluate_final(project, local_path(project, a.run_dir), a.device, a.e0_run, a.e1_evaluation)
    else:
        # CUDA fail-fast comes BEFORE audits, model creation or weight downloads.
        device = require_cuda()
        bundle_info, records = load_bundle(local_path(project, a.bundle), project)
        if a.command == "train":
            config = make_config(a)
            if Path(config["encoder_weights"] or "placeholder").is_absolute():
                raise ValueError("Use a project-relative encoder weight path for portability")
            if config["experiment"] == "pretrained":
                read_pretrained(project / config["encoder_weights"])
            execute_training(project, local_path(project, a.output), config, bundle_info, records, device, a.workers)
        elif a.command == "resume":
            run = local_path(project, a.run_dir)
            info = json.loads((run / "run_info.json").read_text(encoding="utf-8"))
            execute_training(project, run, info["config"], bundle_info, records, device, a.workers, resume=True)
        else:
            recipe = json.loads(local_path(project, a.recipe).read_text(encoding="utf-8"))
            checksum = recipe.pop("recipe_signature")
            if signature(recipe) != checksum or recipe["source_hashes"] != code_hashes() or recipe["bundle_signature"] != bundle_info["bundle_signature"]:
                raise ValueError("Final recipe/data/code changed")
            if recipe["config"]["phase"] != "final" or recipe["config"]["smoke"]:
                raise ValueError("Invalid final training recipe")
            if recipe["config"]["experiment"] == "pretrained":
                _, digest = read_pretrained(project / recipe["config"]["encoder_weights"])
                if digest != recipe["initial_encoder_sha256"]:
                    raise ValueError("Pretraining weight source changed since development")
            execute_training(project, local_path(project, a.output), recipe["config"], bundle_info, records, device, a.workers)


if __name__ == "__main__":
    main()  # Required guard for Windows DataLoader multiprocessing.
