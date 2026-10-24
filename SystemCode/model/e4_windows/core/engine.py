"""CUDA training engine, epoch checkpoints, validation and portable evaluation."""
import csv
import gc
import importlib.metadata
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data import TrainingDataset, normalize_rgb, signature
from models import make_model, segmentation_loss
from e0_protocol import binary_metrics, group_metrics, file_hash, read_mask, write_csv, write_json

PACKAGE = Path(__file__).resolve().parent
CODE_FILES = ("run_e2.py", "engine.py", "data.py", "models.py", "original_unet.py", "e0_protocol.py", "check_device.py")
PROTOCOL = {
    "id": "endolens-e2-probability-v1", "input": "RGB, Pillow bilinear resize, /255, ImageNet mean/std",
    "output": "sigmoid(bilinear_resize(logits, original H,W, align_corners=False))",
    "threshold": .5, "per_image_minmax": False,
    "aggregation": "arithmetic mean of per-image binary Dice/IoU; ALL weighted by entry count",
    "empty_masks": "both empty =>1, exactly one empty =>0", "training_loss": "FP32 BCEWithLogits + per-image soft Dice, smooth=1",
}


def utc():
    return datetime.now(timezone.utc).isoformat()


def code_hashes():
    return {n: file_hash(PACKAGE / n) for n in CODE_FILES}


def say(output, message):
    print(message, flush=True)
    with (Path(output) / "console.log").open("a", encoding="utf-8") as f:
        f.write(message + "\n")


def environment(device):
    result = {"python": sys.version, "platform": platform.platform(), "device": str(device),
              "python_executable": sys.executable, "torch": str(torch.__version__), "cuda_build": torch.version.cuda,
              "versions": {k: importlib.metadata.version(k) for k in ("torch", "torchvision", "numpy", "Pillow")},
              "cpu_threads": torch.get_num_threads()}
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        result.update(gpu=props.name, gpu_total_bytes=props.total_memory,
                      gpu_capability=list(torch.cuda.get_device_capability(device)),
                      cuda_arch_list=torch.cuda.get_arch_list(), cudnn=torch.backends.cudnn.version())
        try:
            result["nvidia_smi"] = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], capture_output=True, text=True, timeout=5).stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            result["nvidia_smi"] = "unavailable"
    return result


def require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("E2 TRAINING REQUIRES CUDA. No CPU fallback. Activate the CUDA environment and run check_device.py.")
    device = torch.device("cuda:0")
    # Kernel execution rather than availability alone checks architecture support.
    x = torch.randn(32, 32, device=device)
    (x @ x).sum().item()
    torch.cuda.synchronize(device)
    return device


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def cpu_tree(v):
    if isinstance(v, torch.Tensor):
        return v.detach().cpu()
    if isinstance(v, dict):
        return {k: cpu_tree(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return type(v)(cpu_tree(x) for x in v)
    return v


def atomic_save(value, path):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(value, tmp)
    os.replace(tmp, path)


def load_checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def set_seed(seed, device):
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def lr_at_epoch(config, epoch):
    warmup, horizon = config["warmup_epochs"], config["schedule_horizon"]
    if epoch <= warmup and warmup:
        factor = epoch / warmup
    else:
        progress = (epoch - warmup) / max(1, horizon - warmup)
        factor = .1 + .9 * .5 * (1 + math.cos(math.pi * min(1, progress)))
    return config["lr"] * factor


def evaluate_records(model, project, records, device, size, output=None, warmup=0):
    """FP32 forward and original-size metrics. No validation augmentation."""
    rows = []
    model.eval()
    with torch.inference_mode():
        for _ in range(warmup):
            model(torch.zeros(1, 3, size, size, device=device))
        sync(device)
        loop_start = time.perf_counter()
        for index, record in enumerate(records, 1):
            gt = read_mask(Path(project) / record["mask"])
            with Image.open(Path(project) / record["image"]) as im:
                x = normalize_rgb(im.convert("RGB"), size).unsqueeze(0).to(device)
            sync(device)
            start = time.perf_counter()
            logits = model(x)
            sync(device)
            ms = (time.perf_counter() - start) * 1000
            probability = torch.sigmoid(F.interpolate(logits.float(), size=gt.shape, mode="bilinear", align_corners=False))[0, 0].cpu().numpy().astype(np.float32)
            if not np.isfinite(probability).all() or probability.shape != gt.shape:
                raise ValueError("Invalid probability: " + record["sample_id"])
            row = {"sample_id": record["sample_id"], "dataset": record.get("dataset", record.get("source_hint", "validation")),
                   "height": gt.shape[0], "width": gt.shape[1], **binary_metrics(probability >= .5, gt),
                   "mae": float(np.abs(probability - gt.astype(np.float32)).mean()), "forward_ms": ms,
                   "gt_area_fraction": float(gt.mean())}
            if output is not None:
                dataset = record["dataset"]
                stem = record["source_stem"]
                pp, mp = Path("probabilities") / dataset / (stem + ".npz"), Path("masks") / dataset / (stem + ".png")
                for path in (pp, mp):
                    (Path(output) / path.parent).mkdir(parents=True, exist_ok=True)
                np.savez_compressed(Path(output) / pp, probability=probability)
                Image.fromarray((probability >= .5).astype(np.uint8) * 255).save(Path(output) / mp)
                row.update(probability_path=pp.as_posix(), mask_path=mp.as_posix())
            rows.append(row)
            if output is not None and (index % 25 == 0 or index == len(records)):
                write_csv(Path(output) / "per_image.csv", rows)
                print(f"[evaluation {index}/{len(records)}] {record['sample_id']} Dice={row['dice']:.4f}", flush=True)
    summary = {"count": len(rows), "mean_dice": float(np.mean([r["dice"] for r in rows])),
               "mean_iou": float(np.mean([r["iou"] for r in rows])), "mean_mae": float(np.mean([r["mae"] for r in rows])),
               "analysis_loop_seconds": time.perf_counter() - loop_start,
               "empty_prediction_count": sum(r["tp"] + r["fp"] == 0 for r in rows),
               "zero_dice_count": sum(r["dice"] == 0 for r in rows)}
    return summary, rows


def execute_training(project, output, config, bundle_info, records, device, workers=2, resume=False, _test_only=False):
    """Only unit tests may inject CPU. CLI training always requires CUDA."""
    project, output = Path(project), Path(output)
    if device.type != "cuda" and not _test_only:
        raise RuntimeError("Training on CPU/MPS is blocked. No automatic fallback.")
    if workers < 0:
        raise ValueError("workers must be nonnegative")
    hashes = code_hashes()
    if resume:
        info = json.loads((output / "run_info.json").read_text(encoding="utf-8"))
        if info["status"] == "TRAINED":
            raise ValueError("This run is already complete")
        checkpoint = load_checkpoint(output / "checkpoints/last.pth")
        if checkpoint["config"] != config or checkpoint["source_hashes"] != hashes:
            raise ValueError("Resume config/code mismatch. Use the unchanged code snapshot.")
        env = environment(device)
        if checkpoint["environment"]["versions"] != env["versions"] or checkpoint["environment"]["device"] != str(device) or checkpoint["environment"].get("gpu") != env.get("gpu"):
            raise ValueError("Resume requires matching library versions and training device")
        if checkpoint["bundle_signature"] != bundle_info["bundle_signature"]:
            raise ValueError("Resume data bundle changed")
        # Repair sidecar records from the atomic last checkpoint if interrupted
        # between checkpoint commit and JSON/CSV writes.
        history = checkpoint["history"]
        best_epoch, best_score = checkpoint["best_epoch"], checkpoint["best_score"]
        if config["phase"] == "development" and not config["smoke"]:
            best_file = output / "checkpoints/best.pth"
            expected_best = checkpoint.get("best_checkpoint_sha256")
            if not best_file.is_file() or file_hash(best_file) != expected_best:
                # A Ctrl+C between best.pth and last.pth commits may leave a
                # newer best file alongside the previous completed epoch.
                backup = output / "checkpoints/best_backup.pth"
                if not backup.is_file() or file_hash(backup) != expected_best:
                    raise ValueError("Best checkpoint missing/inconsistent. Copy the complete run, including backups.")
                temporary = best_file.with_name("best.pth.restore.tmp")
                shutil.copyfile(backup, temporary)
                os.replace(temporary, best_file)
                csv_backup = output / "validation_best_backup_per_image.csv"
                if csv_backup.is_file():
                    shutil.copyfile(csv_backup, output / "validation_best_per_image.csv")
        start_epoch = checkpoint["epoch"] + 1
        info.setdefault("resume_events", []).append({"at": utc(), "from_epoch": checkpoint["epoch"]})
    else:
        output.mkdir(parents=True, exist_ok=False)
        (output / "checkpoints").mkdir()
        (output / "code_snapshot").mkdir()
        for name in CODE_FILES:
            (output / "code_snapshot" / name).write_bytes((PACKAGE / name).read_bytes())
        write_json(output / "bundle_snapshot.json", bundle_info)
        for key, rows in records.items():
            write_json(output / (key + "_manifest.json"), rows)
        info = {"status": "INITIALIZING", "started_at": utc(), "config": config,
                "run_kind": "GPU_SMOKE" if config["smoke"] else "E2_DEVELOPMENT" if config["phase"] == "development" else "E2_FINAL",
                "environment": environment(device), "source_hashes": hashes,
                "bundle_signature": bundle_info["bundle_signature"], "completed_epochs": 0,
                "reproducibility": "epoch-specific order/augmentation and RNG restored on resume; exact cross-device equality not guaranteed"}
        write_json(output / "run_info.json", info)
        try:
            info["pip_freeze"] = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, timeout=15).stdout
        except (OSError, subprocess.TimeoutExpired):
            info["pip_freeze"] = "unavailable"
        history, checkpoint, start_epoch, best_epoch, best_score = [], None, 1, 0, -1.0
    training = records["dev_train"] if config["phase"] == "development" else records["full_train"]
    validation = records["validation"] if config["phase"] == "development" else []
    if config["smoke"]:
        training, validation = training[:8], validation[:5]
    set_seed(config["seed"], device)
    # Never load pretraining weights again on resume or evaluation.
    model = make_model(config, project, initialization=not resume).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    use_amp = config["amp"] and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    if checkpoint:
        model.load_state_dict(checkpoint["model_state"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor) and key != "step":
                    state[key] = value.to(device)
        scaler.load_state_dict(checkpoint["scaler_state"])
        torch.set_rng_state(checkpoint["cpu_rng_state"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_states"])
    del checkpoint
    info.update(best_epoch=best_epoch, best_validation_dice=best_score if validation else None,
                best_checkpoint_sha256=file_hash(output / "checkpoints/best.pth") if resume and validation else None)
    info.update(status="TRAINING", parameter_count=sum(p.numel() for p in model.parameters()),
                model_device=str(next(model.parameters()).device), amp_enabled=use_amp,
                training_samples_per_epoch=len(training), validation_samples=len(validation))
    if config["experiment"] == "pretrained" and not resume:
        info["initial_encoder_sha256"] = model.initial_weight_sha256
    write_json(output / "run_info.json", info)
    write_csv(output / "train_history.csv", history)
    say(output, f"Run: {output}\nTRAIN DEVICE={device}; GPU={info['environment'].get('gpu','unit-test CPU')}; parameters={info['parameter_count']:,}; AMP={use_amp}\nphysical batch={config['batch_size']}; accumulation={config['accumulation']}; effective batch={config['batch_size'] * config['accumulation']}; phase={config['phase']}")
    try:
        for epoch in range(start_epoch, config["epochs"] + 1):
            if code_hashes() != hashes:
                raise ValueError("Training source changed mid-run")
            model.train()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            lr = lr_at_epoch(config, epoch)
            for group in optimizer.param_groups:
                group["lr"] = lr
            dataset = TrainingDataset(project, training, config["input_size"], config["seed"], epoch, config["augmentation"])
            loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True,
                                generator=torch.Generator().manual_seed(config["seed"] + epoch),
                                num_workers=workers, pin_memory=device.type == "cuda", drop_last=False)
            optimizer.zero_grad(set_to_none=True)
            total, seen, updates, skipped_updates = torch.zeros(3, device=device), 0, 0, 0
            sync(device)
            started = time.perf_counter()
            first_step = None
            for i, (x, y) in enumerate(loader):
                group_start = (i // config["accumulation"]) * config["accumulation"]
                group_samples = min(config["batch_size"] * config["accumulation"], len(dataset) - group_start * config["batch_size"])
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    logits = model(x)
                loss, bce, dice = segmentation_loss(logits, y)
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError(f"Nonfinite loss: epoch={epoch}, batch={i+1}")
                if i == 0:
                    if not (x.device == y.device == logits.device == next(model.parameters()).device == device):
                        raise RuntimeError("MODEL/BATCH/OUTPUT device mismatch")
                    first_step = {"epoch": epoch, "model": str(next(model.parameters()).device), "images": str(x.device),
                                  "masks": str(y.device), "logits": str(logits.device), "forward_dtype": str(logits.dtype),
                                  "loss_dtype": str(loss.dtype)}
                scaler.scale(loss * (len(x) / group_samples)).backward()
                total += torch.stack((loss.detach(), bce.detach(), dice.detach())) * len(x)
                seen += len(x)
                if (i + 1) % config["accumulation"] == 0 or i + 1 == len(loader):
                    scale_before = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    skipped = use_amp and scaler.get_scale() < scale_before
                    skipped_updates += int(skipped)
                    updates += int(not skipped)
                    optimizer.zero_grad(set_to_none=True)
                if (i + 1) % 40 == 0 or i + 1 == len(loader):
                    say(output, f"[epoch {epoch}/{config['epochs']}] batch {i+1}/{len(loader)} loss={loss.item():.5f}")
            sync(device)
            train_seconds = time.perf_counter() - started
            if seen != len(training):
                raise RuntimeError("Training sample coverage mismatch")
            values = (total / seen).cpu().tolist()
            val_started = time.perf_counter()
            val, val_rows = evaluate_records(model, project, validation, device, config["input_size"]) if validation else (None, [])
            val_seconds = time.perf_counter() - val_started if validation else 0.0
            row = {"epoch": epoch, "train_loss": values[0], "bce_loss": values[1], "soft_dice_loss": values[2],
                   "lr": lr, "train_samples": seen, "optimizer_updates": updates, "amp_skipped_updates": skipped_updates,
                   "train_seconds": train_seconds, "validation_seconds": val_seconds,
                   "validation_dice": val["mean_dice"] if val else None,
                   "validation_iou": val["mean_iou"] if val else None,
                   "peak_cuda_allocated_mb": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else 0}
            history.append(row)
            if val:
                write_csv(output / "validation_last_per_image.csv", val_rows)
            weights = {"format_version": 1, "config": config, "epoch": epoch, "model_state": cpu_tree(model.state_dict()),
                       "source_hashes": hashes, "bundle_signature": bundle_info["bundle_signature"],
                       "selection": "maximum validation Dice, earlier epoch breaks ties" if validation else "predeclared final epoch"}
            if val and val["mean_dice"] > best_score:
                best_file = output / "checkpoints/best.pth"
                if best_file.is_file():
                    backup_tmp = output / "checkpoints/best_backup.pth.tmp"
                    shutil.copyfile(best_file, backup_tmp)
                    os.replace(backup_tmp, output / "checkpoints/best_backup.pth")
                    if (output / "validation_best_per_image.csv").is_file():
                        shutil.copyfile(output / "validation_best_per_image.csv", output / "validation_best_backup_per_image.csv")
                best_epoch, best_score = epoch, val["mean_dice"]
                atomic_save(weights, best_file)
                write_csv(output / "validation_best_per_image.csv", val_rows)
            if epoch == config["epochs"]:
                atomic_save(weights, output / "checkpoints/final.pth")
            state = {**weights, "optimizer_state": cpu_tree(optimizer.state_dict()), "scaler_state": scaler.state_dict(),
                     "history": history, "best_epoch": best_epoch, "best_score": best_score,
                     "best_checkpoint_sha256": file_hash(output / "checkpoints/best.pth") if validation else None,
                     "environment": environment(device), "cpu_rng_state": torch.get_rng_state(),
                     "cuda_rng_states": torch.cuda.get_rng_state_all() if device.type == "cuda" else []}
            atomic_save(state, output / "checkpoints/last.pth")
            del state, weights, total, logits, loss, bce, dice, x, y
            write_csv(output / "train_history.csv", history)
            info.update(completed_epochs=epoch, best_epoch=best_epoch, best_validation_dice=best_score if validation else None,
                        best_checkpoint_sha256=file_hash(output / "checkpoints/best.pth") if validation else None,
                        last_device_proof=first_step, updated_at=utc())
            write_json(output / "run_info.json", info)
            eta = (train_seconds + val_seconds) * (config["epochs"] - epoch) / 3600
            val_text = f"val Dice={val['mean_dice']:.4f}, best={best_score:.4f}@{best_epoch}" if val else "no validation in full-data final training"
            say(output, f"Epoch {epoch}: loss={values[0]:.5f}, {val_text}; train={train_seconds:.1f}s val={val_seconds:.1f}s; rough remaining={eta:.2f}h; peak CUDA={row['peak_cuda_allocated_mb']:.0f}MB")
            if _test_only and config.get("test_interrupt_epoch") == epoch:
                raise KeyboardInterrupt("unit-test interruption after checkpoint commit")
        info.update(status="TRAINED", completed_epochs=config["epochs"], finished_at=utc(),
                    final_checkpoint_sha256=file_hash(output / "checkpoints/final.pth"),
                    elapsed_wall_seconds=(datetime.now(timezone.utc) - datetime.fromisoformat(info["started_at"])).total_seconds())
        write_json(output / "run_info.json", info)
        say(output, "TRAINING COMPLETE. Development runs have not evaluated any test images.")
    except BaseException as error:
        info.update(status="INTERRUPTED" if isinstance(error, KeyboardInterrupt) else "FAILED",
                    error=f"{type(error).__name__}: {error}", updated_at=utc())
        write_json(output / "run_info.json", info)
        say(output, "Run incomplete. Resume the unchanged run from its last completed epoch. CUDA OOM never triggers CPU fallback.")
        raise
    finally:
        del model, optimizer
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return output


def compare_reference(rows, reference_file, label, output):
    with Path(reference_file).open(encoding="utf-8-sig", newline="") as f:
        base_rows = list(csv.DictReader(f))
    base = {r["sample_id"]: r for r in base_rows}
    if len(base_rows) != 798 or len(base) != 798 or set(base) != {r["sample_id"] for r in rows}:
        raise ValueError("Reference sample membership mismatch")
    deltas = []
    for r in rows:
        b = base[r["sample_id"]]
        if (int(b["height"]), int(b["width"])) != (r["height"], r["width"]):
            raise ValueError("Reference dimensions mismatch")
        deltas.append({"sample_id": r["sample_id"], "dataset": r["dataset"], "reference_dice": float(b["dice"]),
                       "e2_dice": r["dice"], "delta_dice": r["dice"] - float(b["dice"]),
                       "reference_iou": float(b["iou"]), "e2_iou": r["iou"], "delta_iou": r["iou"] - float(b["iou"])})
    write_csv(Path(output) / f"comparison_{label}_per_image.csv", deltas)
    grouped = []
    for group in group_metrics(rows, include_legacy=False):
        selected = deltas if group["dataset"] == "ALL" else [r for r in deltas if r["dataset"] == group["dataset"]]
        grouped.append({"dataset": group["dataset"], "count": len(selected),
                        "reference_dice": float(np.mean([r["reference_dice"] for r in selected])), "e2_dice": group["mean_dice"],
                        "delta_dice": float(np.mean([r["delta_dice"] for r in selected])),
                        "reference_iou": float(np.mean([r["reference_iou"] for r in selected])), "e2_iou": group["mean_iou"],
                        "delta_iou": float(np.mean([r["delta_iou"] for r in selected]))})
    write_csv(Path(output) / f"comparison_{label}.csv", grouped)
