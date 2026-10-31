"""Separate positive segmentation and negative false-foreground outcomes."""
from collections import defaultdict
import numpy as np


def per_image(probability, gt, record, threshold=.5):
    if probability.shape != gt.shape or not np.isfinite(probability).all() or probability.min()<0 or probability.max()>1:
        raise ValueError('Invalid native probability output')
    pred = probability >= threshold
    tp = int((pred & gt).sum())
    fp = int((pred & ~gt).sum())
    fn = int((~pred & gt).sum())
    denominator = 2*tp+fp+fn
    dice = 2*tp/denominator if denominator else 1.
    union = tp+fp+fn
    area = float(pred.mean())
    return {'sample_id': record['sample_id'], 'dataset': record['dataset'],
        'group_id': record['group_id'], 'is_positive': bool(record['is_positive']),
        'label_kind': record['label_kind'], 'center': record['center'],
        'height': gt.shape[0], 'width': gt.shape[1], 'dice': dice,
        'iou': tp/union if union else 1., 'mae': float(np.abs(probability-gt.astype(np.float32)).mean()),
        'tp': tp, 'fp': fp, 'fn': fn, 'gt_pixels': int(gt.sum()),
        'pred_pixels': tp+fp, 'pred_area_fraction': area,
        'any_foreground': int(tp+fp>0), 'area_ge_0_001': int(area>=.001),
        'area_ge_0_01': int(area>=.01),
        'probability_min': float(probability.min()), 'probability_max': float(probability.max()),
        'probability_mean': float(probability.mean())}


def mean(rows, key):
    return float(np.mean([r[key] for r in rows])) if rows else None


def segmentation(rows):
    return {'count': len(rows), 'dice': mean(rows, 'dice'), 'iou': mean(rows, 'iou'),
            'mae': mean(rows, 'mae'), 'low_dice': sum(r['dice']<.5 for r in rows),
            'near_zero_dice': sum(r['dice']<.1 for r in rows),
            'zero_dice': sum(r['dice']==0 for r in rows)}


def negative(rows):
    return {'count': len(rows), 'fpr_any': mean(rows, 'any_foreground'),
            'fpr_area_ge_0_001': mean(rows, 'area_ge_0_001'),
            'fpr_area_ge_0_01': mean(rows, 'area_ge_0_01'),
            'mean_fp_area_fraction': mean(rows, 'pred_area_fraction'),
            'mean_foreground_score': mean(rows, 'probability_mean')}


def summarize(rows):
    pos = [r for r in rows if r['is_positive']]
    neg = [r for r in rows if not r['is_positive']]
    sources = sorted({r['dataset'] for r in pos})
    source_metrics = {s: segmentation([r for r in pos if r['dataset']==s]) for s in sources}
    groups = defaultdict(list)
    for r in neg:
        groups[r['group_id']].append(r)
    group_metrics = {k: negative(v) for k, v in sorted(groups.items())}
    seq = [r for r in neg if r['label_kind']=='author_negative_only_sequence']
    seq_groups = [v for k, v in group_metrics.items() if k.startswith('seq')]
    return {'positive': segmentation(pos), 'positive_sources': source_metrics,
        'positive_source_macro_dice': float(np.mean([v['dice'] for v in source_metrics.values()])) if pos else None,
        'negative_all': negative(neg), 'negative_sequence': negative(seq),
        'negative_single': negative([r for r in neg if r['label_kind']=='verified_empty_mask_and_bbox']),
        'negative_groups': group_metrics,
        'negative_sequence_macro_fpr_any': float(np.mean([v['fpr_any'] for v in seq_groups])) if seq_groups else None,
        'negative_sequence_macro_fp_area': float(np.mean([v['mean_fp_area_fraction'] for v in seq_groups])) if seq_groups else None,
        'negative_sequence_macro_fpr_area_ge_0_001': float(np.mean([v['fpr_area_ge_0_001'] for v in seq_groups])) if seq_groups else None,
        'mixed_dice_supplement_only': mean(rows, 'dice'),
        'mixed_dice_warning': 'Correct negative empty masks contribute Dice=1; never use mixed Dice for checkpoint/model selection.'}


def positive_protection(candidate, reference):
    reasons = []
    if candidate['positive_source_macro_dice'] < reference['positive_source_macro_dice']-.01:
        reasons.append('positive_macro_dice_drop_gt_0.01')
    for source, r in reference['positive_sources'].items():
        if candidate['positive_sources'][source]['dice'] < r['dice']-.02:
            reasons.append('positive_source_drop_gt_0.02:' + source)
    if candidate['positive']['low_dice'] > reference['positive']['low_dice']+2:
        reasons.append('positive_low_dice_increase_gt_2')
    if candidate['positive']['zero_dice'] > reference['positive']['zero_dice']+1:
        reasons.append('positive_zero_dice_increase_gt_1')
    return reasons


def checkpoint_rank(summary):
    return (-summary['negative_sequence_macro_fpr_any'],
            -summary['negative_sequence_macro_fp_area'], summary['positive_source_macro_dice'])
