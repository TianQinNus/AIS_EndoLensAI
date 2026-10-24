#!/usr/bin/env python3
"""E0: fixed Polyp-PVT inference on the existing merged test set.

No training, no test-time tuning, no modification of the original repository.
Python 3.10; tested dependency versions are in requirements-e0-mac.txt.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time
from collections import Counter
from unittest.mock import patch

import numpy as np
from PIL import Image

DATASETS = ("Kvasir", "CVC-ClinicDB", "CVC-ColonDB", "ETIS-LaribPolypDB", "CVC-300")
EXPECTED = dict(zip(DATASETS, (100, 62, 380, 196, 60)))
EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
PROTOCOL = {
    "id": "endolens-e0-probability-v1",
    "input": "RGB; Pillow bilinear resize 352x352; float32 /255; ImageNet normalization",
    "output": "sigmoid(bilinear_resize(P1_logits + P2_logits, original H,W, align_corners=False))",
    "probability_storage": "compressed NPZ, key probability, float32, original HxW, no per-image minmax",
    "binary_prediction": "probability >= 0.5; saved uint8 mask uses 0/255",
    "ground_truth": "8-bit grayscale: {0,1} encoded masks use >0; otherwise >127.5",
    "metrics": "per-image binary Dice and IoU, arithmetic mean within each group",
    "empty_masks": "GT and prediction both empty => Dice=IoU=1; exactly one empty => 0",
    "overall": "image-weighted mean across all samples, NOT equal-weight mean of dataset means",
    "legacy": "per-image minmax eps=1e-8, round to uint8, /255, threshold 0.5; both empty => 0 (old script convention)",
}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def file_hash(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def index_images(folder):
    if not folder.is_dir():
        raise FileNotFoundError(f"Missing directory: {folder}")
    result = {}
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix.lower() in EXTENSIONS:
            if p.stem in result:
                raise ValueError(f"Duplicate stem {p.stem}: {folder}")
            result[p.stem] = p
    if not result:
        raise ValueError(f"No images: {folder}")
    return result


def read_rgb(path):
    with Image.open(path) as im:
        return np.array(im.convert("RGB"), dtype=np.uint8)


def read_mask(path):
    with Image.open(path) as im:
        if im.mode not in ("1", "L", "P", "RGB", "RGBA"):
            raise ValueError(f"Unsupported mask mode {im.mode}: {path}")
        a = np.array(im.convert("L"), dtype=np.uint8)
    return a > (0 if int(a.max()) <= 1 else 127.5)


def paired_index(folder):
    images = index_images(folder / "images")
    masks = index_images(folder / "masks")
    if images.keys() != masks.keys():
        raise ValueError(f"Unpaired images/masks in {folder}: {sorted(images.keys() ^ masks.keys())[:10]}")
    return images, masks


def audit_dataset(project, output):
    """Require one-to-one membership AND decoded image/binary-mask equality."""
    root = project / "dataset" / "TestDataset"
    merged_images, merged_masks = paired_index(root / "test")
    manifest, seen, errors = [], set(), []
    for dataset in DATASETS:
        images, masks = paired_index(root / dataset)
        if len(images) != EXPECTED[dataset]:
            errors.append(f"{dataset}: expected {EXPECTED[dataset]}, found {len(images)}")
        for stem, original in images.items():
            merged_stem = stem + dataset  # Naming convention of the supplied merged folder.
            if merged_stem in seen:
                errors.append(f"Duplicate mapping: {merged_stem}")
                continue
            seen.add(merged_stem)
            if merged_stem not in merged_images:
                errors.append(f"Missing merged sample: {merged_stem}")
                continue
            image, mask = merged_images[merged_stem], merged_masks[merged_stem]
            a, b = read_rgb(image), read_rgb(original)
            gt, source_gt = read_mask(mask), read_mask(masks[stem])
            if a.shape != b.shape or not np.array_equal(a, b):
                errors.append(f"Decoded RGB differs: {merged_stem}")
            if gt.shape != source_gt.shape or not np.array_equal(gt, source_gt):
                errors.append(f"Binary mask differs: {merged_stem}")
            if gt.shape != a.shape[:2]:
                errors.append(f"Image-mask dimensions differ: {merged_stem}")
            manifest.append({
                "sample_id": dataset + "/" + stem,
                "dataset": dataset, "source_stem": stem, "merged_stem": merged_stem,
                "image": str(image.relative_to(project)), "mask": str(mask.relative_to(project)),
                "source_image": str(original.relative_to(project)),
                "source_mask": str(masks[stem].relative_to(project)),
                "height": a.shape[0], "width": a.shape[1],
                "gt_foreground_pixels": int(gt.sum()),
                "image_sha256": file_hash(image), "mask_sha256": file_hash(mask),
                "decoded_rgb_sha256": hashlib.sha256(a.tobytes()).hexdigest(),
            })
        print(f"[audit] {dataset}: {len(images)} pairs", flush=True)
    extra = set(merged_images) - seen
    if extra:
        errors.append(f"Unmapped merged samples: {sorted(extra)[:10]}")
    if len(merged_images) != sum(EXPECTED.values()):
        errors.append(f"Expected 798 merged samples, found {len(merged_images)}")
    duplicate_counts = Counter((r["height"], r["width"], r["decoded_rgb_sha256"]) for r in manifest)
    report = {"passed": not errors, "merged_count": len(merged_images),
              "mapped_count": len(manifest), "counts": dict(Counter(r["dataset"] for r in manifest)),
              "duplicate_decoded_image_groups": sum(v > 1 for v in duplicate_counts.values()),
              "checks": "paired stems, expected counts, full membership, decoded RGB equality, binary mask equality, dimensions",
              "errors": errors}
    write_json(output / "audit.json", report)
    write_json(output / "manifest.json", manifest)
    if errors:
        raise ValueError("Dataset audit failed. Read audit.json. " + errors[0])
    return manifest


def binary_metrics(pred, gt, empty_score=1.0):
    if pred.shape != gt.shape:
        raise ValueError("Prediction and GT shape mismatch")
    pred, gt = pred.astype(bool), gt.astype(bool)
    tp = int(np.count_nonzero(pred & gt))
    fp = int(np.count_nonzero(pred & ~gt))
    fn = int(np.count_nonzero(~pred & gt))
    total = 2 * tp + fp + fn
    union = tp + fp + fn
    return {"dice": 2 * tp / total if total else empty_score,
            "iou": tp / union if union else empty_score,
            "tp": tp, "fp": fp, "fn": fn}


def legacy_uint8(probability):
    p = (probability - probability.min()) / (probability.max() - probability.min() + 1e-8)
    return np.clip(np.rint(p * 255), 0, 255).astype(np.uint8)


def group_metrics(rows, include_legacy):
    out = []
    for name in ("ALL",) + DATASETS:
        selected = rows if name == "ALL" else [r for r in rows if r["dataset"] == name]
        if not selected:
            continue
        item = {"dataset": name, "count": len(selected)}
        columns = ["dice", "iou", "mae", "forward_ms"]
        if include_legacy:
            columns += ["legacy_dice", "legacy_iou"]
        for col in columns:
            item["mean_" + col] = float(np.mean([r[col] for r in selected]))
        item["median_forward_ms"] = float(np.median([r["forward_ms"] for r in selected]))
        out.append(item)
    return out


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def resolve_project(value):
    if value:
        project = Path(value).expanduser().resolve()
    else:
        candidates = [Path.cwd(), Path(__file__).resolve().parent, Path(__file__).resolve().parent.parent]
        project = next((p for p in candidates if (p / "lib" / "pvt.py").is_file()), None)
        if project is None:
            raise ValueError("Run inside Polyp-PVT or supply --project '/absolute/path/Polyp-PVT'")
    if not (project / "lib" / "pvt.py").is_file():
        raise FileNotFoundError(f"Not a Polyp-PVT repository: {project}")
    return project


def choose_device(torch, requested):
    if requested == "auto":
        # CPU is the Mac reference. MPS is deliberately opt-in for this legacy model.
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. On Mac use --device cpu.")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable in this environment. Use --device cpu.")
    return torch.device(requested)


def load_model(torch, project, checkpoint, device):
    """Original constructor loads a relative backbone path. Keep changes scoped."""
    sys.path.insert(0, str(project))
    from lib.pvt import PolypPVT
    real_load = torch.load
    def cpu_state_load(*a, **kw):
        kw["map_location"] = "cpu"
        kw["weights_only"] = True
        return real_load(*a, **kw)
    cwd = Path.cwd()
    try:
        os.chdir(project)
        with patch.object(torch, "load", side_effect=cpu_state_load):
            model = PolypPVT()
    finally:
        os.chdir(cwd)
    state = real_load(str(checkpoint), map_location="cpu", weights_only=True)
    if "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    model.requires_grad_(False)
    return model.to(device).eval()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--project", help="Polyp-PVT repository root; default: discover from cwd/script location")
    p.add_argument("--output", help="New output directory. Default: PROJECT/runs/e0_TIMESTAMP")
    p.add_argument("--device", choices=("cpu", "auto", "mps", "cuda"), default="cpu")
    p.add_argument("--threads", type=int, default=4, help="Torch CPU threads; default 4")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--audit-only", action="store_true", help="Validate all 798 pairs without loading torch/model")
    p.add_argument("--smoke", action="store_true", help="Infer one image per dataset, clearly marked as SMOKE")
    p.add_argument("--skip-legacy", action="store_true", help="Skip auxiliary old-style minmax PNG/metrics")
    a = p.parse_args()
    if a.threads < 1 or a.warmup < 0:
        p.error("--threads must be positive and --warmup nonnegative")
    return a


def run(args, project, output, info):
    manifest = audit_dataset(project, output)
    info["manifest_sha256"] = file_hash(output / "manifest.json")
    if args.audit_only:
        info.update(status="AUDIT_PASSED", audited_count=len(manifest))
        write_json(output / "run_info.json", info)
        print(f"All 798 pairs verified. Audit: {output}")
        return
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    device = choose_device(torch, args.device)
    checkpoint = project / "model_pth" / "PolypPVT.pth"
    backbone = project / "pretrained_pth" / "pvt_v2_b2.pth"
    for path in (checkpoint, backbone):
        if not path.is_file():
            raise FileNotFoundError(f"Required weight file missing: {path}")
    selected = [next(r for r in manifest if r["dataset"] == d) for d in DATASETS] if args.smoke else manifest
    info.update(device=str(device), torch_cpu_threads=torch.get_num_threads(), batch_size=1,
                input_size=352, precision="float32", threshold=0.5,
                checkpoint=str(checkpoint), checkpoint_sha256=file_hash(checkpoint),
                backbone_sha256=file_hash(backbone),
                source_hashes={s: file_hash(project / s) for s in ("lib/pvt.py", "lib/pvtv2.py")},
                versions={n: importlib.metadata.version(n) for n in ("torch", "torchvision", "timm", "numpy", "Pillow")},
                expected_processed_count=len(selected), include_legacy=not args.skip_legacy,
                mps_fallback=os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "unset"),
                timing="forward only, batch=1, synchronization before/after; excludes preprocess, resize/sigmoid, metrics and files")
    write_json(output / "run_info.json", info)
    print(f"Loading fixed checkpoint on {device} ...", flush=True)
    model = load_model(torch, project, checkpoint, device)
    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elif device.type == "mps":
            torch.mps.synchronize()
    with torch.inference_mode():
        for _ in range(args.warmup):
            model(torch.zeros(1, 3, 352, 352, device=device))
        synchronize()
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    rows = []
    loop_start = time.perf_counter()
    with (output / "per_image.csv").open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = None
        with torch.inference_mode():
            for index, record in enumerate(selected, 1):
                image = project / record["image"]
                gt = read_mask(project / record["mask"])
                with Image.open(image) as im:
                    rgb = im.convert("RGB")
                    h, w = rgb.height, rgb.width
                    arr = np.array(rgb.resize((352, 352), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
                arr = (arr - mean) / std
                tensor = torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1))).unsqueeze(0).to(device)
                synchronize()
                start = time.perf_counter()
                p1, p2 = model(tensor)
                synchronize()
                elapsed_ms = (time.perf_counter() - start) * 1000
                logits = F.interpolate(p1 + p2, size=(h, w), mode="bilinear", align_corners=False)
                prob = torch.sigmoid(logits).squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)
                if prob.shape != gt.shape or not np.isfinite(prob).all() or prob.min() < 0 or prob.max() > 1:
                    raise ValueError(f"Invalid probability output: {record['sample_id']}")
                metrics = binary_metrics(prob >= 0.5, gt)
                sample_path = Path(record["dataset"]) / record["source_stem"]
                prob_path = Path("probabilities") / sample_path.with_suffix(".npz")
                mask_path = Path("masks") / sample_path.with_suffix(".png")
                # Source stems in these datasets have no extension-like suffix.
                for folder in (prob_path.parent, mask_path.parent):
                    (output / folder).mkdir(parents=True, exist_ok=True)
                np.savez_compressed(output / prob_path, probability=prob)
                Image.fromarray((prob >= 0.5).astype(np.uint8) * 255).save(output / mask_path)
                row = {"sample_id": record["sample_id"], "dataset": record["dataset"],
                       "merged_stem": record["merged_stem"], "height": h, "width": w,
                       **metrics, "mae": float(np.abs(prob - gt.astype(np.float32)).mean()),
                       "forward_ms": elapsed_ms, "probability_min": float(prob.min()),
                       "probability_max": float(prob.max()), "probability_path": prob_path.as_posix(),
                       "mask_path": mask_path.as_posix()}
                if not args.skip_legacy:
                    legacy = legacy_uint8(prob)
                    lm = binary_metrics(legacy.astype(np.float64) / 255.0 >= 0.5, gt, empty_score=0.0)
                    legacy_path = Path("legacy_minmax") / sample_path.with_suffix(".png")
                    (output / legacy_path.parent).mkdir(parents=True, exist_ok=True)
                    Image.fromarray(legacy).save(output / legacy_path)
                    row.update(legacy_dice=lm["dice"], legacy_iou=lm["iou"], legacy_path=legacy_path.as_posix())
                if writer is None:
                    writer = csv.DictWriter(csv_file, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                csv_file.flush()
                rows.append(row)
                info["processed_count"] = len(rows)
                if index % 25 == 0 or index == len(selected):
                    write_json(output / "run_info.json", info)
                print(f"[{index:3d}/{len(selected)}] {record['sample_id']} Dice={row['dice']:.4f} IoU={row['iou']:.4f} forward={elapsed_ms:.1f}ms", flush=True)
    info["analysis_loop_seconds"] = time.perf_counter() - loop_start
    info["analysis_loop_timing"] = "all selected images: decode, preprocess, transfer, inference, metrics and result writing; excludes audit, model loading and warmup"
    summary = group_metrics(rows, not args.skip_legacy)
    write_csv(output / "summary.csv", summary)
    write_json(output / "metrics.json", {"run_kind": info["run_kind"], "protocol": PROTOCOL,
               "processed_count": len(rows), "test_manifest_count": len(manifest), "groups": summary})
    info.update(status="COMPLETED", finished_at=datetime.now(timezone.utc).isoformat())
    write_json(output / "run_info.json", info)
    print("\n" + ("SMOKE TEST ONLY; not the full E0 result." if args.smoke else "E0 complete."))
    for r in summary:
        print(f"{r['dataset']:20s} n={r['count']:3d} Dice={r['mean_dice']:.6f} IoU={r['mean_iou']:.6f}")
    print(f"\nResults: {output}")


def main():
    args = parse_args()
    project = resolve_project(args.project)
    name = "e0_" + ("audit_" if args.audit_only else "smoke_" if args.smoke else "") + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = Path(args.output).expanduser().resolve() if args.output else project / "runs" / name
    output.mkdir(parents=True, exist_ok=False)
    info = {"status": "RUNNING", "run_kind": "AUDIT" if args.audit_only else "SMOKE" if args.smoke else "FULL_E0",
            "started_at": datetime.now(timezone.utc).isoformat(), "project": str(project),
            "python": sys.version, "platform": platform.platform(), "machine": platform.machine(),
            "logical_cpu_count": os.cpu_count(), "arguments": vars(args), "protocol": PROTOCOL,
            "script_sha256": file_hash(Path(__file__).resolve()), "processed_count": 0}
    write_json(output / "run_info.json", info)
    try:
        run(args, project, output, info)
    except BaseException as e:
        info.update(status="INTERRUPTED" if isinstance(e, KeyboardInterrupt) else "FAILED", error=f"{type(e).__name__}: {e}")
        write_json(output / "run_info.json", info)
        print(f"\nRun did not complete. Read {output / 'run_info.json'}. Partial CSV is not a full result.", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
