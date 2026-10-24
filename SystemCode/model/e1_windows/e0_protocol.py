"""Dataset audit and evaluation helpers inherited from the validated E0 protocol.
The E1 package is self-contained; no import from the original model is required.
"""
import csv
import hashlib
import json
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image
DATASETS = ("Kvasir", "CVC-ClinicDB", "CVC-ColonDB", "ETIS-LaribPolypDB", "CVC-300")
EXPECTED = dict(zip(DATASETS, (100, 62, 380, 196, 60)))
EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
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
                "image": image.relative_to(project).as_posix(), "mask": mask.relative_to(project).as_posix(),
                "source_image": original.relative_to(project).as_posix(),
                "source_mask": masks[stem].relative_to(project).as_posix(),
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


