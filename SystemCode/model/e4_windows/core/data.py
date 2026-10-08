"""Immutable E2 data bundle, source hints, optional patient/video grouping."""
import csv
import hashlib
import json
import math
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
from PIL import Image, ImageEnhance
import torch
from torch.utils.data import Dataset
from e0_protocol import audit_dataset, paired_index, read_rgb, read_mask, file_hash, write_json

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def local_path(project, value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (Path(project) / path).resolve()


def normalize_rgb(image, size):
    a = np.asarray(image.resize((size, size), Image.Resampling.BILINEAR), dtype=np.float32) / 255
    a = (a - MEAN) / STD
    return torch.from_numpy(np.ascontiguousarray(a.transpose(2, 0, 1)))


def area_bin(record):
    fraction = record["gt_foreground_pixels"] / (record["height"] * record["width"])
    return "small_lt1pct" if fraction < .01 else "medium_1to10pct" if fraction < .1 else "large_ge10pct"


def split_training(records, seed=42, fraction=.2, metadata=None):
    """Group exact duplicates and optional supplied patient/video identities.

    Filename families are documented hints, not recovered patient identities.
    Patient grouping may yield an approximate rather than exact 80:20 split.
    """
    records = sorted(records, key=lambda r: r["sample_id"])
    if not 0 < fraction < 1:
        raise ValueError("Validation fraction must be between zero and one")
    metadata = metadata or {}
    if metadata and set(metadata) != {r["sample_id"] for r in records}:
        raise ValueError("Metadata must contain exactly one entry for every training sample_id")
    parent = list(range(len(records)))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(a, b):
        parent[root(b)] = root(a)
    seen_rgb, seen_group = {}, {}
    for i, r in enumerate(records):
        key = (r["height"], r["width"], r["decoded_rgb_sha256"])
        if key in seen_rgb:
            union(i, seen_rgb[key])
        seen_rgb[key] = i
        if metadata:
            m = metadata[r["sample_id"]]
            if not m.get("source") or not m.get("group_id"):
                raise ValueError("Each metadata row needs source and group_id")
            key = (m["source"], m["group_id"])
            if key in seen_group:
                union(i, seen_group[key])
            seen_group[key] = i
    groups = defaultdict(list)
    for i, r in enumerate(records):
        r = dict(r)
        r["source_hint"] = metadata[r["sample_id"]]["source"] if metadata else (
            "numeric_filename_family" if r["stem"].isdigit() else "opaque_filename_family")
        r["area_bin"] = area_bin(r)
        r["group_id"] = min(records[j]["sample_id"] for j in range(len(records)) if root(j) == root(i))
        groups[root(i)].append(r)
    bins = defaultdict(list)
    for group in groups.values():
        key = Counter((r["source_hint"], r["area_bin"]) for r in group).most_common(1)[0][0]
        bins[key].append(group)
    # Largest-remainder allocation across group strata. For this local dataset
    # (unique images and no metadata), exact 1160/290 with 1450 records.
    target_groups = round(len(groups) * fraction)
    allocations = {k: math.floor(len(v) * fraction) for k, v in bins.items()}
    remainder = target_groups - sum(allocations.values())
    order = sorted(bins, key=lambda k: (-(len(bins[k]) * fraction - allocations[k]), k))
    for k in order[:remainder]:
        allocations[k] += 1
    train, validation = [], []
    for k in sorted(bins):
        values = sorted(bins[k], key=lambda g: signature([seed, g[0]["group_id"]]))
        n = allocations[k]
        for i, group in enumerate(values):
            (validation if i < n else train).extend(group)
    if not train or not validation:
        raise ValueError("Grouping produced an empty training or validation partition")
    train.sort(key=lambda r: r["sample_id"])
    validation.sort(key=lambda r: r["sample_id"])
    return train, validation


def prepare_bundle(project, output, e0_run, seed=42, metadata_path=None):
    project, output = Path(project), Path(output)
    e0_run = local_path(project, e0_run)
    e0_info = json.loads((e0_run / "run_info.json").read_text(encoding="utf-8"))
    if not (e0_info.get("status") == "COMPLETED" and e0_info.get("run_kind") == "FULL_E0" and e0_info.get("processed_count") == 798):
        raise ValueError("Supply a completed FULL_E0 with 798 entries")
    if file_hash(e0_run / "manifest.json") != e0_info.get("manifest_sha256"):
        raise ValueError("E0 manifest checksum changed")
    if e0_info.get("protocol", {}).get("id") != "endolens-e0-probability-v1":
        raise ValueError("Unsupported E0 protocol")
    output.mkdir(parents=True, exist_ok=False)
    test = audit_dataset(project, output)
    reference = json.loads((e0_run / "manifest.json").read_text(encoding="utf-8"))
    if signature(test) != signature(reference):
        raise ValueError("Current test content differs from the frozen E0 manifest")
    images, masks = paired_index(project / "dataset/TrainDataset")
    if len(images) != 1450:
        raise ValueError(f"Expected 1450 training pairs, found {len(images)}")
    test_keys = {(r["height"], r["width"], r["decoded_rgb_sha256"]) for r in test}
    training = []
    for stem, image in images.items():
        rgb, gt = read_rgb(image), read_mask(masks[stem])
        if rgb.shape[:2] != gt.shape:
            raise ValueError("Training dimensions differ: " + stem)
        digest = hashlib.sha256(rgb.tobytes()).hexdigest()
        if (rgb.shape[0], rgb.shape[1], digest) in test_keys:
            raise ValueError("Exact train/test overlap: " + stem)
        training.append({"sample_id": "TrainDataset/" + stem, "stem": stem,
                         "image": image.relative_to(project).as_posix(), "mask": masks[stem].relative_to(project).as_posix(),
                         "height": rgb.shape[0], "width": rgb.shape[1], "gt_foreground_pixels": int(gt.sum()),
                         "decoded_rgb_sha256": digest, "image_sha256": file_hash(image), "mask_sha256": file_hash(masks[stem])})
    metadata = None
    if metadata_path:
        with local_path(project, metadata_path).open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        metadata = {r["sample_id"]: r for r in rows}
        if len(metadata) != len(rows):
            raise ValueError("Duplicate metadata sample_id")
        write_json(output / "source_metadata.json", rows)
    train, val = split_training(training, seed, metadata=metadata)
    full = sorted(train + val, key=lambda r: r["sample_id"])
    descriptor = {"format_version": 1, "split_seed": seed, "requested_validation_fraction": .2,
                  "files": {}, "counts": {"full_train": len(full), "dev_train": len(train), "validation": len(val), "test": len(test)},
                  "e0_test_signature": signature(test), "grouped_by_supplied_identity": bool(metadata),
                  "strata": {name: dict(Counter(r["source_hint"] + "/" + r["area_bin"] for r in rows))
                             for name, rows in (("dev_train", train), ("validation", val))},
                  "limitations": ["No patient/sequence independence claim without supplied reliable identities.",
                                  "Filename families are source hints only. Near-duplicate grouping not verified.",
                                  "Original test entries include duplicate content and have already been inspected during E1.",
                                  "Internal validation may have been used to train the existing PVT reference. Do not tune fusion on it."]}
    for name, rows in (("full_train", full), ("dev_train", train), ("validation", val), ("test", test)):
        write_json(output / (name + ".json"), rows)
        descriptor["files"][name] = {"filename": name + ".json", "sha256": file_hash(output / (name + ".json"))}
    descriptor["bundle_signature"] = signature(descriptor)
    write_json(output / "bundle.json", descriptor)
    print(f"[E2 data] full={len(full)} development={len(train)} validation={len(val)} test={len(test)}")
    return descriptor


def load_bundle(path, project=None, verify_content=True):
    path = Path(path)
    info = json.loads((path / "bundle.json").read_text(encoding="utf-8"))
    recorded = info.pop("bundle_signature")
    if signature(info) != recorded:
        raise ValueError("Bundle descriptor changed")
    info["bundle_signature"] = recorded
    result = {}
    for name, item in info["files"].items():
        p = path / item["filename"]
        if file_hash(p) != item["sha256"]:
            raise ValueError("Frozen data manifest changed: " + name)
        result[name] = json.loads(p.read_text(encoding="utf-8"))
    if len(result["full_train"]) != 1450 or len(result["test"]) != 798:
        raise ValueError("Invalid original outer split counts")
    train_ids = {r["sample_id"] for r in result["dev_train"]}
    val_ids = {r["sample_id"] for r in result["validation"]}
    if train_ids & val_ids or train_ids | val_ids != {r["sample_id"] for r in result["full_train"]}:
        raise ValueError("Invalid internal partition membership")
    if {r["group_id"] for r in result["dev_train"]} & {r["group_id"] for r in result["validation"]}:
        raise ValueError("Group overlap in internal partition")
    if verify_content:
        if project is None:
            raise ValueError("Project required to verify data contents")
        for r in result["full_train"] + result["test"]:
            for kind in ("image", "mask"):
                if file_hash(Path(project) / r[kind]) != r[kind + "_sha256"]:
                    raise ValueError("Dataset file changed: " + r["sample_id"])
    return info, result


class TrainingDataset(Dataset):
    def __init__(self, project, records, size, seed, epoch, augmentation="flips"):
        self.project, self.records, self.size = Path(project), records, size
        self.seed, self.epoch, self.augmentation = seed, epoch, augmentation

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        with Image.open(self.project / r["image"]) as im:
            rgb = im.convert("RGB")
        mask = Image.fromarray(read_mask(self.project / r["mask"]).astype(np.uint8) * 255)
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, i]))
        for operation in (Image.Transpose.FLIP_LEFT_RIGHT, Image.Transpose.FLIP_TOP_BOTTOM):
            if rng.random() < .5:
                rgb, mask = rgb.transpose(operation), mask.transpose(operation)
        if self.augmentation == "strong":
            angle = float(rng.uniform(-15, 15))
            rgb = rgb.rotate(angle, resample=Image.Resampling.BILINEAR)
            mask = mask.rotate(angle, resample=Image.Resampling.NEAREST)
            rgb = ImageEnhance.Brightness(rgb).enhance(float(rng.uniform(.85, 1.15)))
            rgb = ImageEnhance.Contrast(rgb).enhance(float(rng.uniform(.85, 1.15)))
        x = normalize_rgb(rgb, self.size)
        y = np.asarray(mask.resize((self.size, self.size), Image.Resampling.NEAREST), dtype=np.uint8) > 127
        return x, torch.from_numpy(y.astype(np.float32)[None].copy())
