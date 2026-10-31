"""Build mixed-label splits before inference; preserve E8 positive membership."""
import csv
import hashlib
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from PIL import Image
from common import (ROOT, DATA, POLYPGEN, STARTER, sha, read_json, write_json,
                    write_csv, relative, safe_path, signature, load_data, utc)

PAPER = 'https://doi.org/10.1038/s41597-023-01981-y'
FIGURE = 'https://media.springernature.com/full/springer-static/image/art%3A10.1038%2Fs41597-023-01981-y/MediaObjects/41597_2023_1981_Fig7_HTML.png'
# Figure 7 was visually inspected. In particular seq7_neg is red (C1).
# Figure 2's aggregate C1/C4 counts differ by 207 (=seq7_neg). Both are TRAIN
# centers; role membership is unaffected. Do not silently reassign by counts.
NEGATIVE_CENTERS = {i: ('C1' if 4 <= i <= 7 else 'C4') for i in range(1, 11)}
NEGATIVE_CENTERS.update({i: 'C5' for i in range(11, 14)})
NEGATIVE_CENTERS.update({i: 'C6' for i in range(14, 24)})


def mask_array(path):
    with Image.open(path) as im:
        return np.asarray(im.convert('L'), np.float32) > 127.5


def bbox_for_single(directory, center, mask, stem):
    path = directory / f'bbox_C{center}' / (mask.stem + '.txt')
    return path if path.is_file() else directory / f'bbox_C{center}' / (stem + '.txt')


def verified_empty_single(image, mask, bbox):
    return image.is_file() and mask.is_file() and bbox.is_file() and not bbox.read_text().strip() and not mask_array(mask).any()


def fingerprint(path):
    with Image.open(path) as im:
        rgb = im.convert('RGB')
        a = np.asarray(rgb, np.uint8)
        gray = np.asarray(rgb.convert('L').resize((9, 8), Image.Resampling.BILINEAR))
        dh = np.packbits((gray[:, 1:] > gray[:, :-1]).reshape(-1)).tobytes().hex()
        thumb = np.asarray(rgb.resize((16, 16), Image.Resampling.BILINEAR), np.uint8)
        return {'decoded_rgb_sha256': hashlib.sha256(a.tobytes()).hexdigest(),
                'dhash64': dh, 'height': a.shape[0], 'width': a.shape[1]}, thumb


class NearIndex:
    def __init__(self):
        self.rows, self.thumbs, self.buckets, self.exact = [], [], defaultdict(set), defaultdict(list)

    def add(self, row, thumb):
        i = len(self.rows)
        self.rows.append(row)
        self.thumbs.append(thumb)
        value = int(row['dhash64'], 16)
        for j in range(4):
            self.buckets[(j, (value >> (16*j)) & 65535)].add(i)
        self.exact[row['decoded_rgb_sha256']].append(i)

    def matches(self, row, thumb):
        value = int(row['dhash64'], 16)
        candidates = set(self.exact[row['decoded_rgb_sha256']])
        for j in range(4):
            candidates.update(self.buckets.get((j, (value >> (16*j)) & 65535), ()))
        result = []
        for i in sorted(candidates):
            other = self.rows[i]
            if other['decoded_rgb_sha256'] == row['decoded_rgb_sha256']:
                result.append((other, 'exact_decoded', 0.0))
                continue
            if (value ^ int(other['dhash64'], 16)).bit_count() > 2:
                continue
            if abs(row['width']/row['height'] - other['width']/other['height']) > .01:
                continue
            mae = float(np.abs(thumb.astype(np.float32) - self.thumbs[i]).mean()/255)
            if mae <= .01:
                result.append((other, 'near_image_heuristic', mae))
        return result


def verify_teammate_membership(positive):
    mapping = {'train': 'dev_train', 'validation': 'validation', 'test': 'test'}
    for role, name in mapping.items():
        a = {r['sample_id']: (r['image_sha256'], r['mask_sha256']) for r in positive[role]}
        b = {r['sample_id']: (r['image_sha256'], r['mask_sha256'])
             for r in read_json(STARTER / 'splits' / (name + '.json'))}
        if a != b:
            raise ValueError('R34/R50 positive split membership differs: ' + role)


def prepare():
    if (DATA / 'bundle.json').exists():
        info, rows = load_data(verify_bytes=True)
        print('EXISTING_DATA_VERIFIED', info['counts'], flush=True)
        return
    DATA.mkdir(parents=True, exist_ok=True)
    old = ROOT / 'runs/e8_data_seed42'
    sources = {str(p.relative_to(ROOT)).replace('\\', '/'): sha(p)
               for p in old.glob('*.json')}
    positive = {
        'train': read_json(old / 'original_train.json') + read_json(old / 'new_train.json'),
        'validation': read_json(old / 'validation.json'),
        'test': read_json(old / 'holdout.json'),
    }
    verify_teammate_membership(positive)
    for role, rows in positive.items():
        for r in rows:
            r.update(is_positive=True, label_kind='positive_mask', role=role,
                     center=r['dataset'].replace('PolypGen-', '') if r['dataset'].startswith('PolypGen-') else 'original')
    protected = {}
    for rows in positive.values():
        for r in rows:
            protected.setdefault(r['image'], r)
    for path in [ROOT/'runs/e0_20261004_221345_872444/manifest.json',
                 ROOT/'runs/e6_development_seed42/development_manifest.json']:
        sources[relative(path)] = sha(path)
        for r in read_json(path):
            protected.setdefault(r['image'], r)
    index = NearIndex()

    def protected_record(row):
        for key in ['image', 'mask']:
            if sha(safe_path(row[key])) != row[key + '_sha256']:
                raise ValueError('Protected data changed: ' + row['sample_id'])
        fp, thumb = fingerprint(safe_path(row['image']))
        if row.get('decoded_rgb_sha256') and fp['decoded_rgb_sha256'] != row['decoded_rgb_sha256']:
            raise ValueError('Protected decoded image mismatch')
        return {**row, **fp}, thumb

    with ThreadPoolExecutor(max_workers=4) as pool:
        for i, (r, thumb) in enumerate(pool.map(protected_record, protected.values()), 1):
            index.add(r, thumb)
            if i % 250 == 0 or i == len(protected):
                print(f'[protected fingerprint] {i}/{len(protected)}', flush=True)
    candidates, quarantine = [], []
    # Only explicit, present, all-zero masks AND empty author bbox files qualify.
    raw_empty = Counter()
    for center in range(1, 7):
        directory = POLYPGEN / f'data_C{center}'
        for mask in sorted((directory/f'masks_C{center}').glob('*.jpg')):
            gt = mask_array(mask)
            if gt.any():
                continue
            raw_empty['C'+str(center)] += 1
            stem = mask.stem.removesuffix('_mask')
            image = directory / f'images_C{center}' / (stem + '.jpg')
            bbox = bbox_for_single(directory, center, mask, stem)
            if not verified_empty_single(image, mask, bbox):
                quarantine.append({'sample_id': f'negative-single-C{center}/{stem}',
                                   'reason': 'empty_mask_without_matching_empty_author_bbox'})
                continue
            role = 'train' if center <= 4 else ('validation' if center == 5 else 'test')
            candidates.append({'sample_id': f'negative-single-C{center}/{stem}',
                'dataset': f'PolypGen-negative-single-C{center}', 'center': f'C{center}',
                'group_id': f'negative-single-C{center}', 'role': role, 'is_positive': False,
                'label_kind': 'verified_empty_mask_and_bbox', 'image': relative(image),
                'mask': relative(mask), 'bbox': relative(bbox), 'bbox_sha256': sha(bbox),
                'gt_foreground_pixels': 0})
    sequence_raw = {}
    for seq in range(1, 24):
        center = NEGATIVE_CENTERS[seq]
        group = f'seq{seq}_neg'
        directory = POLYPGEN / 'sequenceData/negativeOnly' / group
        images = sorted(p for p in directory.iterdir() if p.suffix.lower() in ('.jpg', '.png', '.jpeg')
                        and not p.name.startswith('._'))
        sequence_raw[group] = {'center_figure7': center, 'frames': len(images)}
        if not images:
            raise ValueError('Missing explicit negative sequence: ' + group)
        role = 'train' if seq <= 10 else ('validation' if seq <= 13 else 'test')
        for image in images:
            candidates.append({'sample_id': group + '/' + image.stem,
                'dataset': f'PolypGen-negative-sequence-{center}', 'center': center,
                'group_id': group, 'role': role, 'is_positive': False,
                'label_kind': 'author_negative_only_sequence', 'image': relative(image),
                'mask': relative(DATA / 'zero_masks' / group / (image.stem + '.png')),
                'gt_foreground_pixels': 0})
    if sum(r['frames'] for r in sequence_raw.values()) != 4275 or sum(raw_empty.values()) != 125:
        raise ValueError('Unexpected raw negative inventory; do not silently change protocol')
    print('[negative inventory] 4275 authored negative sequence frames; 125 empty single-frame masks', flush=True)

    def inspect_negative(row):
        fp, thumb = fingerprint(safe_path(row['image']))
        if row['label_kind'] == 'author_negative_only_sequence':
            path = safe_path(row['mask'])
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                Image.new('L', (fp['width'], fp['height']), 0).save(path)
        gt = mask_array(safe_path(row['mask']))
        if gt.shape != (fp['height'], fp['width']) or gt.any():
            raise ValueError('Negative label/dimension mismatch: ' + row['sample_id'])
        return {**row, **fp, 'image_sha256': sha(safe_path(row['image'])),
                'mask_sha256': sha(safe_path(row['mask']))}, thumb

    fingerprints = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i, pair in enumerate(pool.map(inspect_negative, candidates), 1):
            fingerprints.append(pair)
            if i % 250 == 0 or i == len(candidates):
                print(f'[negative label + fingerprint] {i}/{len(candidates)}', flush=True)
    kept = {k: [] for k in positive}
    negative_index = NearIndex()
    # Test/validation priority is fixed BEFORE any model inference.
    priority = {'test': 0, 'validation': 1, 'train': 2}
    for i, (row, thumb) in enumerate(sorted(fingerprints, key=lambda p: (priority[p[0]['role']], p[0]['sample_id'])), 1):
        positive_matches = index.matches(row, thumb)
        if positive_matches:
            quarantine.append({'sample_id': row['sample_id'], 'role': row['role'],
                'reason': 'negative_conflicts_with_protected_positive_image',
                'matches': [[r['sample_id'], kind, mae] for r, kind, mae in positive_matches]})
            continue
        matches = negative_index.matches(row, thumb)
        bad = [(r, kind, mae) for r, kind, mae in matches
               if r['role'] != row['role'] or kind == 'exact_decoded']
        if bad:
            quarantine.append({'sample_id': row['sample_id'], 'role': row['role'],
                'reason': 'negative_cross_role_near_or_repeated_exact_image',
                'matches': [[r['sample_id'], kind, mae] for r, kind, mae in bad]})
            continue
        kept[row['role']].append(row)
        negative_index.add(row, thumb)
        if i % 500 == 0:
            print(f'[negative overlap audit] {i}/{len(fingerprints)}', flush=True)
    manifests, counts = {}, {}
    for role in ['train', 'validation', 'test']:
        rows = positive[role] + sorted(kept[role], key=lambda r: r['sample_id'])
        path = DATA / (role + '.json')
        write_json(path, rows)
        manifests[role] = sha(path)
        counts[role] = {'positive': len(positive[role]), 'negative': len(kept[role]),
                        'negative_sequence': sum(r['label_kind'] == 'author_negative_only_sequence' for r in kept[role]),
                        'negative_single': sum(r['label_kind'] == 'verified_empty_mask_and_bbox' for r in kept[role]),
                        'all': len(rows)}
        if not counts[role]['negative_sequence'] or not counts[role]['negative_single']:
            raise ValueError('A required negative evaluation stratum is empty')
    roles = ['train', 'validation', 'test']
    for i, left in enumerate(roles):
        for right in roles[i+1:]:
            a = {r['decoded_rgb_sha256'] for r in positive[left]+kept[left]}
            b = {r['decoded_rgb_sha256'] for r in positive[right]+kept[right]}
            if a & b:
                raise ValueError('Cross-role exact decoded overlap')
    write_json(DATA/'quarantine.json', quarantine)
    write_json(DATA/'negative_sequence_inventory.json', sequence_raw)
    write_csv(DATA/'counts.csv', [{'split': role, **value} for role, value in counts.items()])
    info = {'stage': 'E12_POSITIVE_NEGATIVE_COMPARISON', 'created_utc': utc(),
        'manifests': manifests, 'counts': counts, 'input_sources': sources,
        'raw_negative_sequence_frames': 4275, 'raw_empty_single_masks': dict(raw_empty),
        'quarantined_negative_candidates': len(quarantine),
        'negative_label_evidence': 'Present empty mask plus empty author bbox, or explicit authored negativeOnly folder; never missing-mask inference.',
        'negative_center_mapping': {'source': PAPER, 'figure7': FIGURE,
            'visually_verified_local_figure': 'e12_negative_windows/resources/polypgen_fig7.png',
            'seq7_caveat': 'Figure7 labels seq7_neg C1; Figure2 totals imply C4. Both train; no validation/test reassignment. Negative image sampling is uniform, not center-balanced.'},
        'roles': {'training': 'positive E8 train plus negatives C1-C4/seq1-10',
                  'validation': 'positive E8 validation plus negatives C5/seq11-13',
                  'historical_test': 'fixed83 positive C6 plus negatives C6/seq14-23'},
        'audit': 'Exact decoded RGB and dHash<=2/RGB16 MAE<=.01/aspect difference<=.01; fixed test>validation>train priority; no metrics used.',
        'limitations': ['No complete patient mapping for original images; sequence IDs are grouping proxies, not verified patient IDs.',
                       'C6 positive results were viewed before this experiment; all final comparison is exploratory.',
                       'The R50 initialization lineage includes nine now-quarantined images.',
                       'Image heuristics cannot establish complete absence of related frames.']}
    info['signature'] = signature(info)
    write_json(DATA/'bundle.json', info)
    print('DATA_PREPARE_PASS', counts, 'quarantined=', len(quarantine), flush=True)


if __name__ == '__main__':
    prepare()
