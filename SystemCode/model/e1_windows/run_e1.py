#!/usr/bin/env python3
"""E1: randomly initialized U-Net trained on the existing 1450 pairs.

Uses a frozen E0 test manifest, last-epoch selection, MPS/CPU, and epoch resume.
No original Polyp-PVT model imports and no test-based checkpoint selection.
"""
from __future__ import annotations

import argparse
import gc
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import time

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from e0_protocol import (
    DATASETS, audit_dataset, binary_metrics, file_hash, group_metrics,
    paired_index, read_mask, read_rgb, write_csv, write_json,
)
from unet import UNet, segmentation_loss

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
PROTOCOL = {
    "id": "endolens-e1-probability-v1",
    "input": "RGB; Pillow bilinear resize; float32 /255; ImageNet normalization",
    "output": "sigmoid(bilinear_resize(UNet_logits, original H,W, align_corners=False))",
    "probability_storage": "compressed NPZ key probability; float32 original HxW; NO per-image minmax",
    "binary_prediction": "probability >= 0.5; uint8 mask 0/255",
    "metrics": "per-image binary Dice/IoU then arithmetic mean; ALL is image-weighted",
    "ground_truth": "L-mode; max<=1 uses >0, otherwise >127.5",
    "empty_masks": "both empty => 1; exactly one empty => 0",
}


def utc():
    return datetime.now(timezone.utc).isoformat()


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def code_hashes():
    base = Path(__file__).resolve().parent
    return {name: file_hash(base / name) for name in ("run_e1.py", "unet.py", "e0_protocol.py")}


def environment(device):
    return {"python": sys.version, "platform": platform.platform(), "machine": platform.machine(),
            "versions": {n: importlib.metadata.version(n) for n in ("torch", "numpy", "Pillow")},
            "device": str(device), "mps_built": torch.backends.mps.is_built(),
            "mps_available": torch.backends.mps.is_available(),
            "mps_fallback": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "unset"),
            "precision": "float32", "torch_cpu_threads": torch.get_num_threads()}


def choose_device(requested):
    if requested == "auto":
        requested = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable. Run check_device.py in your local Terminal with native arm64 Python/current stable PyTorch, or explicitly use --device cpu.")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; Apple GPU uses --device mps.")
    return torch.device(requested)


def sync(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def resolve_project(value):
    candidates = [Path(value).expanduser().resolve()] if value else [Path.cwd(), Path(__file__).resolve().parent.parent]
    for project in candidates:
        if (project / "dataset/TrainDataset/images").is_dir():
            return project.resolve()
    raise FileNotFoundError("Use --project '/absolute/path/Polyp-PVT' or run in its root directory.")


def resolve_e0(project, value):
    if value:
        candidates = [Path(value).expanduser().resolve()]
    else:
        runs = project / "runs"
        candidates = sorted(runs.glob("e0_*"), reverse=True) if runs.is_dir() else []
    for directory in candidates:
        path = directory / "run_info.json"
        if not path.is_file():
            continue
        info = json.loads(path.read_text())
        if info.get("status") == "COMPLETED" and info.get("run_kind") == "FULL_E0" and info.get("processed_count") == 798:
            manifest_path = directory / "manifest.json"
            if file_hash(manifest_path) != info.get("manifest_sha256"):
                raise ValueError("E0 manifest checksum changed: " + str(directory))
            if info.get("protocol", {}).get("id") != "endolens-e0-probability-v1":
                raise ValueError("Unsupported E0 evaluation protocol")
            return directory, json.loads(manifest_path.read_text())
    raise ValueError("No completed FULL_E0 with 798 samples. Supply --e0-run '/path/to/e0_...'.")


def prepare_data(project, e0_directory, reference):
    # Temporary output keeps audits separate from immutable training snapshots.
    with tempfile.TemporaryDirectory(prefix="endolens-e1-audit-") as tmp:
        tmp = Path(tmp)
        test = audit_dataset(project, tmp)
        test_audit = json.loads((tmp / "audit.json").read_text())
    if signature(test) != signature(reference):
        raise ValueError("Current test data/manifest differs from E0. Do not change the agreed test split.")
    if len(test) != 798 or len({r["sample_id"] for r in test}) != 798:
        raise ValueError("Invalid frozen test manifest")
    images, masks = paired_index(project / "dataset/TrainDataset")
    if len(images) != 1450:
        raise ValueError(f"Expected 1450 training pairs, found {len(images)}")
    test_keys = {(r["height"], r["width"], r["decoded_rgb_sha256"]) for r in test}
    training, overlap, seen = [], [], {}
    for stem, image in images.items():
        rgb, gt = read_rgb(image), read_mask(masks[stem])
        if rgb.shape[:2] != gt.shape:
            raise ValueError("Training image/mask dimensions differ: " + stem)
        digest = hashlib.sha256(rgb.tobytes()).hexdigest()
        key = (rgb.shape[0], rgb.shape[1], digest)
        if key in test_keys:
            overlap.append(stem)
        seen[key] = seen.get(key, 0) + 1
        training.append({"sample_id": "TrainDataset/" + stem, "stem": stem,
                         "image": image.relative_to(project).as_posix(), "mask": masks[stem].relative_to(project).as_posix(),
                         "height": rgb.shape[0], "width": rgb.shape[1],
                         "gt_foreground_pixels": int(gt.sum()), "decoded_rgb_sha256": digest,
                         "image_sha256": file_hash(image), "mask_sha256": file_hash(masks[stem])})
    audit = {"passed": not overlap, "training_count": len(training), "test_count": len(test),
             "training_unique_rgb_images": len(seen), "training_duplicate_groups": sum(n > 1 for n in seen.values()),
             "train_test_exact_rgb_overlap_count": len(overlap), "overlap_training_stems": overlap,
             "test_matches_e0": True, "e0_run": str(e0_directory),
             "limitations": "Exact RGB equality only; no patient/sequence-level independence verification. Existing duplicates retained."}
    return training, test, test_audit, audit


def audit_into(project, e0_value, output):
    e0_directory, reference = resolve_e0(project, e0_value)
    training, test, test_audit, audit = prepare_data(project, e0_directory, reference)
    write_json(output / "train_manifest.json", training)
    write_json(output / "test_manifest.json", test)
    write_json(output / "test_audit.json", test_audit)
    write_json(output / "data_audit.json", audit)
    if not audit["passed"]:
        raise ValueError("Exact training/test RGB overlap found. Read data_audit.json; split was not silently changed.")
    print(f"[audit] training=1450 test=798; exact train/test overlap=0; E0={e0_directory.name}", flush=True)
    return training, test, e0_directory


def normalize_rgb(image, size):
    arr = np.asarray(image.resize((size, size), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    return torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1)))


class TrainingDataset(Dataset):
    def __init__(self, project, records, size, seed, epoch=0):
        self.project, self.records, self.size, self.seed, self.epoch = project, records, size, seed, epoch

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        with Image.open(self.project / record["image"]) as im:
            rgb = im.convert("RGB")
        mask = Image.fromarray(read_mask(self.project / record["mask"]).astype(np.uint8) * 255)
        # Per-sample/epoch RNG independent of worker scheduling and resume.
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, index]))
        for operation in (Image.Transpose.FLIP_LEFT_RIGHT, Image.Transpose.FLIP_TOP_BOTTOM):
            if rng.random() < 0.5:
                rgb, mask = rgb.transpose(operation), mask.transpose(operation)
        x = normalize_rgb(rgb, self.size)
        y = np.asarray(mask.resize((self.size, self.size), Image.Resampling.NEAREST), dtype=np.uint8) > 127
        return x, torch.from_numpy(y.astype(np.float32)[None].copy())


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    return value


def atomic_save(value, path):
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def load_checkpoint(path):
    # Package-generated tensors/primitives only; no custom pickle classes.
    return torch.load(path, map_location="cpu", weights_only=True)


def optimizer_to(optimizer, device):
    for state in optimizer.state.values():
        for key, value in state.items():
            # AdamW non-capturable scalar step stays CPU; moment tensors follow parameter.
            if isinstance(value, torch.Tensor) and key != "step":
                state[key] = value.to(device)


def training_config(args):
    return {"model": "standard_padded_unet", "base_channels": args.base_channels,
            "input_size": args.input_size, "epochs": 1 if args.smoke else args.epochs,
            "physical_batch_size": args.batch_size, "accumulation_steps": args.accumulation,
            "effective_batch_size": args.batch_size * args.accumulation,
            "optimizer": "AdamW", "lr": args.lr, "weight_decay": args.weight_decay,
            "betas": [0.9, 0.999], "loss": "BCEWithLogits + per-image soft Dice (1+1, smooth=1)",
            "seed": args.seed, "augmentation": "paired horizontal/vertical flips, each p=0.5",
            "scheduler": None, "pretrained": False, "batch_normalization": False,
            "checkpoint_selection": "last configured epoch; never select using test scores", "smoke": args.smoke}


def run_training(args):
    project = resolve_project(args.project)
    device = choose_device(args.device)
    config = training_config(args)
    if args.resume:
        if args.output:
            raise ValueError("--resume reuses its run directory; do not also pass --output")
        output = Path(args.resume).expanduser().resolve()
        info = json.loads((output / "run_info.json").read_text())
        checkpoint = load_checkpoint(output / "checkpoints/last.pth")
        if checkpoint["config"] != config:
            raise ValueError("Resume requires identical training configuration, including physical batch and accumulation.")
        if checkpoint["source_hashes"] != code_hashes():
            raise ValueError("Code changed since the saved epoch; resume blocked to prevent mixed runs.")
        if checkpoint["environment"]["device"] != str(device) or checkpoint["environment"]["versions"] != environment(device)["versions"]:
            raise ValueError("Resume requires the same device and torch/numpy/Pillow versions.")
        training, test, _, audit = prepare_data(project, Path(info["e0_run"]), json.loads((output / "test_manifest.json").read_text()))
        if not audit["passed"] or signature(training) != checkpoint["training_signature"] or signature(test) != checkpoint["test_signature"]:
            raise ValueError("Training or test data changed since checkpoint")
        history = checkpoint["history"]
        start_epoch = checkpoint["epoch"] + 1
        info.setdefault("resume_events", []).append({"at": utc(), "from_completed_epoch": checkpoint["epoch"]})
    else:
        tag = "e1_smoke_" if args.smoke else "e1_"
        output = Path(args.output).expanduser().resolve() if args.output else project / "runs" / (tag + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
        output.mkdir(parents=True, exist_ok=False)
        info = {"status": "AUDITING", "started_at": utc(), "project": str(project), "config": config,
                "environment": environment(device), "source_hashes": code_hashes(), "completed_epochs": 0}
        write_json(output / "run_info.json", info)
        try:
            training, test, e0 = audit_into(project, args.e0_run, output)
        except BaseException as error:
            info.update(status="FAILED", error=f"{type(error).__name__}: {error}")
            write_json(output / "run_info.json", info)
            raise
        info["e0_run"] = str(e0)
        history, checkpoint, start_epoch = [], None, 1
    formal = (not config["smoke"] and config["epochs"] == 100 and config["input_size"] == 352 and
              config["base_channels"] == 64 and config["effective_batch_size"] == 8 and
              config["lr"] == 1e-4 and config["weight_decay"] == 1e-4 and config["seed"] == 42)
    info["run_kind"] = "SMOKE" if args.smoke else "FULL_E1" if formal else "CUSTOM_E1"
    frozen_code_hashes = info["source_hashes"]
    selected = training[:8] if args.smoke else training
    info.update(status="TRAINING", training_signature=signature(training), test_signature=signature(test),
                training_samples_per_epoch=len(selected), test_processed_count=0,
                reproducibility="fixed seed, epoch order and paired flips; bitwise equality across backends not guaranteed")
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    if device.type == "mps":
        torch.mps.manual_seed(args.seed)
    model = UNet(args.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if checkpoint:
        model.load_state_dict(checkpoint["model_state"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        optimizer_to(optimizer, device)
    del checkpoint
    (output / "checkpoints").mkdir(exist_ok=True)
    info["parameter_count"] = sum(p.numel() for p in model.parameters())
    write_json(output / "run_info.json", info)
    write_csv(output / "train_history.csv", history)
    print(f"Run: {output}\nDevice: {device}; parameters={info['parameter_count']:,}; physical batch={args.batch_size}; effective batch={config['effective_batch_size']}", flush=True)
    try:
        for epoch in range(start_epoch, config["epochs"] + 1):
            if code_hashes() != frozen_code_hashes:
                raise ValueError("Source files changed during training; stop to avoid recording mixed code.")
            model.train()
            dataset = TrainingDataset(project, selected, args.input_size, args.seed, epoch)
            generator = torch.Generator().manual_seed(args.seed + epoch)
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, generator=generator,
                                num_workers=args.workers, pin_memory=False, drop_last=False)
            optimizer.zero_grad(set_to_none=True)
            totals = torch.zeros(3, device=device)
            n_seen, updates = 0, 0
            sync(device)
            start = time.perf_counter()
            for batch_index, (x, y) in enumerate(loader):
                group_start = (batch_index // args.accumulation) * args.accumulation
                # Weight by the actual group sample count, including final partial batch/group.
                group_count = min(args.batch_size * args.accumulation, len(dataset) - group_start * args.batch_size)
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss, bce, dice_loss = segmentation_loss(logits, y)
                if not torch.isfinite(loss).item():
                    raise RuntimeError(f"Nonfinite loss at epoch {epoch}, batch {batch_index + 1}")
                (loss * (len(x) / group_count)).backward()
                totals += torch.stack((loss.detach(), bce.detach(), dice_loss.detach())) * len(x)
                n_seen += len(x)
                end_group = (batch_index + 1) % args.accumulation == 0 or batch_index + 1 == len(loader)
                if end_group:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    updates += 1
                if (batch_index + 1) % args.log_every == 0 or batch_index + 1 == len(loader):
                    print(f"[epoch {epoch}/{config['epochs']}] batch {batch_index + 1}/{len(loader)} loss={loss.item():.5f}", flush=True)
            sync(device)
            elapsed = time.perf_counter() - start
            values = (totals / n_seen).cpu().tolist()
            if n_seen != len(selected):
                raise RuntimeError("Not all training samples were consumed")
            row = {"epoch": epoch, "train_loss": values[0], "bce_loss": values[1], "soft_dice_loss": values[2],
                   "learning_rate": optimizer.param_groups[0]["lr"], "samples": n_seen,
                   "optimizer_updates": updates, "epoch_seconds": elapsed}
            history.append(row)
            state = {"format_version": 1, "epoch": epoch, "model_state": cpu_tree(model.state_dict()),
                     "optimizer_state": cpu_tree(optimizer.state_dict()), "config": config,
                     "training_signature": signature(training), "test_signature": signature(test),
                     "history": history, "source_hashes": frozen_code_hashes, "environment": environment(device)}
            atomic_save(state, output / "checkpoints/last.pth")
            del state
            write_csv(output / "train_history.csv", history)
            info.update(completed_epochs=epoch, last_checkpoint="checkpoints/last.pth", updated_at=utc())
            write_json(output / "run_info.json", info)
            eta = elapsed * (config["epochs"] - epoch) / 3600
            print(f"Epoch {epoch}: mean_loss={values[0]:.6f}, {elapsed:.1f}s (excluding checkpoint write); rough remaining training={eta:.2f}h", flush=True)
            # Release the last autograd graph/inputs before checkpoint evaluation.
            del x, y, logits, loss, bce, dice_loss, totals
        final = {"format_version": 1, "epoch": config["epochs"], "model_state": cpu_tree(model.state_dict()),
                 "config": config, "training_signature": signature(training), "test_signature": signature(test),
                 "source_hashes": frozen_code_hashes, "selection": config["checkpoint_selection"]}
        atomic_save(final, output / "checkpoints/unet_final.pth")
        info.update(status="TRAINED", training_finished_at=utc(), final_checkpoint="checkpoints/unet_final.pth",
                    final_checkpoint_sha256=file_hash(output / "checkpoints/unet_final.pth"))
        write_json(output / "run_info.json", info)
        # Do not retain training model/AdamW states while loading evaluation model.
        del final, model, optimizer
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()
        if not args.train_only:
            evaluate_run(project, output, device, args.warmup)
        else:
            print(f"Training complete. Evaluate using: python e1_windows/run_e1.py evaluate --run-dir '{output}' --device cpu")
    except BaseException as error:
        info = json.loads((output / "run_info.json").read_text())
        info.update(status="INTERRUPTED" if isinstance(error, KeyboardInterrupt) else "FAILED", error=f"{type(error).__name__}: {error}", updated_at=utc())
        write_json(output / "run_info.json", info)
        print("Run incomplete. Resume from the last completed epoch with identical configuration.\nFor MPS out-of-memory, start a new run with --batch-size 1 --accumulation 8. Do not disable the MPS memory limit.", file=sys.stderr)
        raise
    return output


def evaluate_run(project, run_dir, device, warmup, e0_override=None):
    info = json.loads((run_dir / "run_info.json").read_text())
    checkpoint_path = run_dir / "checkpoints/unet_final.pth"
    if file_hash(checkpoint_path) != info["final_checkpoint_sha256"]:
        raise ValueError("Final checkpoint hash changed")
    checkpoint = load_checkpoint(checkpoint_path)
    if checkpoint["source_hashes"] != code_hashes():
        raise ValueError("Evaluation code differs from the training snapshot")
    manifest = json.loads((run_dir / "test_manifest.json").read_text())
    if signature(manifest) != checkpoint["test_signature"]:
        raise ValueError("Test manifest changed")
    # Verify current images/masks against immutable files before evaluation.
    for record in manifest:
        for kind in ("image", "mask"):
            if file_hash(project / record[kind]) != record[kind + "_sha256"]:
                raise ValueError("Test content changed: " + record["sample_id"])
    # Resolve a local E0 reference after moving an E1 run across operating systems.
    # Never trust the source computer's absolute path on the destination computer.
    e0_directory, local_e0_manifest = resolve_e0(project, e0_override)
    if signature(local_e0_manifest) != checkpoint["test_signature"]:
        raise ValueError("Local E0 reference uses a different test manifest")
    config = checkpoint["config"]
    if checkpoint["epoch"] != config["epochs"]:
        raise ValueError("Evaluation requires the predeclared final epoch")
    selected = [next(r for r in manifest if r["dataset"] == d) for d in DATASETS] if config["smoke"] else manifest
    output = run_dir / ("evaluation_" + str(device) + "_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    output.mkdir(exist_ok=False)
    evaluation = {"status": "RUNNING", "run_kind": info["run_kind"], "started_at": utc(),
                  "checkpoint_sha256": info["final_checkpoint_sha256"], "epoch": checkpoint["epoch"],
                  "protocol": {**PROTOCOL, "input_size": config["input_size"]},
                  "environment": environment(device), "evaluation_project": str(project),
                  "e0_reference_run": str(e0_directory), "batch_size": 1, "warmup": warmup,
                  "processed_count": 0, "expected_processed_count": len(selected),
                  "test_signature": checkpoint["test_signature"],
                  "timing": "forward only, synchronized; excludes preprocessing/postprocessing and writes"}
    write_json(output / "run_info.json", evaluation)
    info.update(status="EVALUATING", current_evaluation=str(output.relative_to(run_dir)))
    write_json(run_dir / "run_info.json", info)
    rows = []
    try:
        model = UNet(config["base_channels"])
        model.load_state_dict(checkpoint["model_state"], strict=True)
        model.to(device).eval().requires_grad_(False)
        del checkpoint
        with torch.inference_mode():
            for _ in range(warmup):
                model(torch.zeros(1, 3, config["input_size"], config["input_size"], device=device))
            sync(device)
            loop_start = time.perf_counter()
            for index, record in enumerate(selected, 1):
                gt = read_mask(project / record["mask"])
                with Image.open(project / record["image"]) as im:
                    x = normalize_rgb(im.convert("RGB"), config["input_size"]).unsqueeze(0).to(device)
                sync(device)
                start = time.perf_counter()
                logits = model(x)
                sync(device)
                forward_ms = (time.perf_counter() - start) * 1000
                logits = F.interpolate(logits, size=gt.shape, mode="bilinear", align_corners=False)
                probability = torch.sigmoid(logits)[0, 0].cpu().numpy().astype(np.float32)
                if probability.shape != gt.shape or not np.isfinite(probability).all():
                    raise ValueError("Invalid output: " + record["sample_id"])
                dataset, stem = record["dataset"], record["source_stem"]
                pp = Path("probabilities") / dataset / (stem + ".npz")
                mp = Path("masks") / dataset / (stem + ".png")
                for relative in (pp, mp):
                    (output / relative.parent).mkdir(parents=True, exist_ok=True)
                np.savez_compressed(output / pp, probability=probability)
                Image.fromarray((probability >= 0.5).astype(np.uint8) * 255).save(output / mp)
                row = {"sample_id": record["sample_id"], "dataset": dataset, "merged_stem": record["merged_stem"],
                       "height": record["height"], "width": record["width"], **binary_metrics(probability >= 0.5, gt),
                       "mae": float(np.abs(probability - gt.astype(np.float32)).mean()), "forward_ms": forward_ms,
                       "probability_min": float(probability.min()), "probability_max": float(probability.max()),
                       "probability_path": pp.as_posix(), "mask_path": mp.as_posix()}
                rows.append(row)
                if index % 25 == 0 or index == len(selected):
                    evaluation["processed_count"] = index
                    write_json(output / "run_info.json", evaluation)
                    write_csv(output / "per_image.csv", rows)
                    print(f"[evaluate {index}/{len(selected)}] {record['sample_id']} Dice={row['dice']:.4f}", flush=True)
            evaluation["analysis_loop_seconds"] = time.perf_counter() - loop_start
        summary = group_metrics(rows, include_legacy=False)
        write_csv(output / "per_image.csv", rows)
        write_csv(output / "summary.csv", summary)
        write_json(output / "metrics.json", {"run_kind": info["run_kind"], "protocol": evaluation["protocol"], "processed_count": len(rows), "groups": summary})
        compare_e0(e0_directory, output, rows, summary)
        evaluation.update(status="COMPLETED", processed_count=len(rows), finished_at=utc(),
                          analysis_loop_timing="decode, preprocess, forward, postprocess, metrics and writes; excludes audit/load/warmup")
        write_json(output / "run_info.json", evaluation)
        info.update(status="COMPLETED", test_processed_count=len(rows), last_evaluation=str(output.relative_to(run_dir)), finished_at=utc())
        write_json(run_dir / "run_info.json", info)
        print("SMOKE ONLY" if config["smoke"] else "E1 evaluation complete", flush=True)
        for r in summary:
            print(f"{r['dataset']:20s} n={r['count']:3d} Dice={r['mean_dice']:.6f} IoU={r['mean_iou']:.6f}")
        print("Evaluation output:", output)
    except BaseException as error:
        evaluation.update(status="INTERRUPTED" if isinstance(error, KeyboardInterrupt) else "FAILED", processed_count=len(rows), error=f"{type(error).__name__}: {error}")
        write_json(output / "run_info.json", evaluation)
        write_csv(output / "per_image.csv", rows)
        info.update(status="EVALUATION_FAILED", error=evaluation["error"])
        write_json(run_dir / "run_info.json", info)
        raise


def compare_e0(e0_directory, output, rows, summary):
    import csv
    with (e0_directory / "per_image.csv").open(encoding="utf-8-sig") as f:
        e0 = {r["sample_id"]: r for r in csv.DictReader(f)}
    if len(e0) != 798:
        raise ValueError("E0 per-image results incomplete")
    deltas = []
    for r in rows:
        base = e0[r["sample_id"]]
        if (int(base["height"]), int(base["width"])) != (r["height"], r["width"]):
            raise ValueError("E0/E1 dimensions differ")
        deltas.append({"sample_id": r["sample_id"], "dataset": r["dataset"], "e0_dice": float(base["dice"]),
                       "e1_dice": r["dice"], "delta_dice": r["dice"] - float(base["dice"]),
                       "e0_iou": float(base["iou"]), "e1_iou": r["iou"], "delta_iou": r["iou"] - float(base["iou"])})
    write_csv(output / "comparison_per_image.csv", deltas)
    comparison = []
    for r in summary:
        selected = deltas if r["dataset"] == "ALL" else [d for d in deltas if d["dataset"] == r["dataset"]]
        comparison.append({"dataset": r["dataset"], "count": len(selected),
                           "e0_dice": float(np.mean([d["e0_dice"] for d in selected])), "e1_dice": r["mean_dice"],
                           "delta_dice": float(np.mean([d["delta_dice"] for d in selected])),
                           "e0_iou": float(np.mean([d["e0_iou"] for d in selected])), "e1_iou": r["mean_iou"],
                           "delta_iou": float(np.mean([d["delta_iou"] for d in selected]))})
    write_csv(output / "comparison_e0_e1.csv", comparison)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    for command in ("audit", "train", "evaluate"):
        sub = subs.add_parser(command)
        sub.add_argument("--project")
        sub.add_argument("--device", choices=("auto", "mps", "cpu", "cuda"), default="auto")
        sub.add_argument("--threads", type=int, default=4)
        sub.add_argument("--e0-run", help="Local E0 directory; default newest completed FULL_E0 in project/runs")
        if command in ("audit", "train"):
            sub.add_argument("--output", help="New output directory; must not exist")
        if command == "train":
            sub.add_argument("--resume", help="Existing E1 run directory; resume checkpoints/last.pth")
            sub.add_argument("--epochs", type=int, default=100)
            sub.add_argument("--batch-size", type=int, default=2)
            sub.add_argument("--accumulation", type=int, default=4)
            sub.add_argument("--lr", type=float, default=1e-4)
            sub.add_argument("--weight-decay", type=float, default=1e-4)
            sub.add_argument("--seed", type=int, default=42)
            sub.add_argument("--workers", type=int, default=0, help="Mac-safe default 0")
            sub.add_argument("--input-size", type=int, default=352)
            sub.add_argument("--base-channels", type=int, default=64)
            sub.add_argument("--log-every", type=int, default=20)
            sub.add_argument("--smoke", action="store_true", help="8 train images, 1 epoch, 5 test images; NOT formal E1")
            sub.add_argument("--train-only", action="store_true", help="Evaluate separately, e.g. CPU for E0 timing comparability")
        if command in ("train", "evaluate"):
            sub.add_argument("--warmup", type=int, default=3)
        if command == "evaluate":
            sub.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.command == "train":
        if min(args.epochs, args.batch_size, args.accumulation, args.input_size, args.base_channels, args.log_every) < 1 or args.workers < 0 or args.seed < 0:
            parser.error("Invalid positive count/seed/workers")
        if args.input_size % 16 or args.lr <= 0 or args.weight_decay < 0:
            parser.error("Input size must be divisible by 16; lr positive and weight decay nonnegative")
    if hasattr(args, "warmup") and args.warmup < 0:
        parser.error("--warmup cannot be negative")
    return args


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    if args.command == "train":
        run_training(args)
    elif args.command == "evaluate":
        run_dir = Path(args.run_dir).expanduser().resolve()
        project = Path(args.project).expanduser().resolve() if args.project else Path(json.loads((run_dir / "run_info.json").read_text())["project"])
        evaluate_run(project, run_dir, choose_device(args.device), args.warmup, args.e0_run)
    else:
        project = resolve_project(args.project)
        output = Path(args.output).expanduser().resolve() if args.output else project / "runs" / ("e1_audit_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
        output.mkdir(parents=True, exist_ok=False)
        audit_into(project, args.e0_run, output)
        print("Audit passed:", output)


if __name__ == "__main__":
    main()
