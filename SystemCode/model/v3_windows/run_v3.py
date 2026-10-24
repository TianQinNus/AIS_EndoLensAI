"""V3: predeclared equal probability fusion of frozen E0 and E2 outputs.
No training, CUDA model inference, threshold search, or weight search occurs.
"""
import argparse
import csv
import hashlib
import html
import json
import platform
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

DATASETS = ('Kvasir', 'CVC-ClinicDB', 'CVC-ColonDB', 'ETIS-LaribPolypDB', 'CVC-300')
COUNTS = dict(zip(DATASETS, (100, 62, 380, 196, 60)))
WEIGHT_E0 = WEIGHT_E2 = 0.5
THRESHOLD = 0.5

def now():
    return datetime.now(timezone.utc).isoformat()

def digest(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def read_json(p):
    return json.loads(p.read_text(encoding='utf-8'))

def write_json(p, value):
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

def rows(p):
    with p.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def write_csv(p, data):
    if not data:
        return
    with p.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)

def under(root, relative):
    # Stored relative POSIX paths must not escape their containing directory.
    if not isinstance(relative, str) or '\\' in relative or ':' in relative:
        raise ValueError('Expected relative POSIX path: ' + str(relative))
    p = (root / relative).resolve()
    if p == root or root not in p.parents:
        raise ValueError('Unsafe relative path: ' + relative)
    return p

def by_id(data):
    result = {}
    for r in data:
        k = r['sample_id']
        if k in result:
            raise ValueError('Duplicate sample_id: ' + k)
        result[k] = r
    return result

def load_gt(p):
    with Image.open(p) as im:
        a = np.array(im.convert('L'), dtype=np.uint8)
    return a > (0 if a.max() <= 1 else 127.5)

def load_prob(p, shape):
    with np.load(p, allow_pickle=False) as z:
        a = z['probability']
    if a.dtype != np.float32 or a.shape != shape or not np.isfinite(a).all():
        raise ValueError('Invalid dtype/shape/nonfinite probability: ' + str(p))
    if a.min() < 0 or a.max() > 1:
        raise ValueError('Probability outside [0,1]: ' + str(p))
    return a

def metrics(prob, gt):
    pred = prob >= THRESHOLD
    tp = int(np.count_nonzero(pred & gt)); fp = int(np.count_nonzero(pred & ~gt))
    fn = int(np.count_nonzero(~pred & gt))
    total = 2 * tp + fp + fn; union = tp + fp + fn
    return dict(dice=2 * tp / total if total else 1., iou=tp / union if union else 1.,
                mae=float(np.abs(prob - gt.astype(np.float32)).mean()), tp=tp, fp=fp, fn=fn)

def fuse(a, b):
    if a.shape != b.shape:
        raise ValueError('Fusion shape mismatch')
    return np.float32(.5) * a + np.float32(.5) * b

def close(actual, reference, name, tolerance=1e-12):
    if abs(float(actual) - float(reference)) > tolerance:
        raise ValueError('Reference metric mismatch: ' + name)

def validate_sources(project, e0, e2run, ev):
    i0 = read_json(e0 / 'run_info.json'); i2 = read_json(e2run / 'run_info.json')
    ie = read_json(ev / 'run_info.json')
    if i0.get('status') != 'COMPLETED' or i0.get('run_kind') != 'FULL_E0' or i0.get('processed_count') != 798:
        raise ValueError('E0 must be completed FULL_E0 with 798 entries')
    if i0.get('protocol', {}).get('id') != 'endolens-e0-probability-v1' or i0.get('threshold') != .5:
        raise ValueError('Wrong E0 probability protocol; do not use legacy_minmax outputs')
    if i2.get('status') != 'TRAINED' or i2.get('run_kind') != 'E2_FINAL':
        raise ValueError('E2 must be trained E2_FINAL, not a development/smoke model')
    if ie.get('status') != 'COMPLETED' or ie.get('processed_count') != 798:
        raise ValueError('E2 evaluation must be completed with 798 entries')
    pr = ie.get('protocol', {})
    if pr.get('id') != 'endolens-e2-probability-v1' or pr.get('threshold') != .5 or pr.get('per_image_minmax') is not False:
        raise ValueError('Wrong E2 probability protocol')
    cp_hash = digest(e2run / 'checkpoints/final.pth')
    if cp_hash != i2['final_checkpoint_sha256'] or cp_hash != ie['checkpoint_sha256']:
        raise ValueError('E2 final checkpoint/evaluation hash mismatch')
    if digest(e0 / 'manifest.json') != i0['manifest_sha256']:
        raise ValueError('E0 manifest hash mismatch')
    m0 = by_id(read_json(e0 / 'manifest.json')); m2 = by_id(read_json(e2run / 'test_manifest.json'))
    if m0 != m2 or dict(Counter(r['dataset'] for r in m0.values())) != COUNTS:
        raise ValueError('Frozen E0/E2 test manifests differ or subset counts are wrong')
    r0 = by_id(rows(e0 / 'per_image.csv')); r2 = by_id(rows(ev / 'per_image.csv'))
    if set(r0) != set(m0) or set(r2) != set(m0):
        raise ValueError('Missing/extra samples in source per_image.csv')
    return m0, r0, r2, cp_hash

def audit_sample(project, e0, ev, record, r0, r2):
    image = under(project, record['image']); mask = under(project, record['mask'])
    if digest(image) != record['image_sha256'] or digest(mask) != record['mask_sha256']:
        raise ValueError('Test image/mask changed: ' + record['sample_id'])
    shape = (int(record['height']), int(record['width']))
    gt = load_gt(mask)
    with Image.open(image) as im:
        rgb = np.array(im.convert('RGB'), dtype=np.uint8)
    if gt.shape != shape or rgb.shape[:2] != shape or int(gt.sum()) != record['gt_foreground_pixels']:
        raise ValueError('Test shape/GT count mismatch: ' + record['sample_id'])
    if hashlib.sha256(rgb.tobytes()).hexdigest() != record['decoded_rgb_sha256']:
        raise ValueError('Decoded image mismatch: ' + record['sample_id'])
    arrays = []; calculated = []; hashes = []
    for directory, row in [(e0, r0), (ev, r2)]:
        if row['dataset'] != record['dataset'] or (int(row['height']), int(row['width'])) != shape:
            raise ValueError('Source CSV dataset/shape mismatch')
        probfile = under(directory, row['probability_path']); prob = load_prob(probfile, shape)
        bm = metrics(prob, gt)
        for key, val in bm.items():
            # float32 MAE reductions can differ slightly across NumPy versions.
            close(val, row[key], record['sample_id'] + '/' + key, 2e-6 if key == 'mae' else 1e-12)
        with Image.open(under(directory, row['mask_path'])) as im:
            saved = np.array(im.convert('L'))
        if not np.array_equal(saved, (prob >= .5).astype(np.uint8) * 255):
            raise ValueError('Source saved mask differs from raw probability threshold')
        arrays.append(prob); calculated.append(bm); hashes.append(digest(probfile))
    return rgb, gt, arrays, calculated, hashes

def summarize(data):
    result = []
    for name in ('ALL',) + DATASETS:
        group = data if name == 'ALL' else [r for r in data if r['dataset'] == name]
        s = dict(dataset=name, count=len(group))
        for model in ('e0', 'e2', 'fusion'):
            for metric in ('dice', 'iou', 'mae'):
                s[model + '_' + metric] = float(np.mean([r[model + '_' + metric] for r in group]))
            d = np.array([r[model + '_dice'] for r in group])
            s[model + '_median_dice'] = float(np.median(d))
            s[model + '_dice_lt05'] = int((d < .5).sum())
            s[model + '_zero_dice'] = int((d == 0).sum())
        for model in ('e0', 'e2'):
            for metric in ('dice', 'iou', 'mae'):
                s['delta_' + metric + '_vs_' + model] = s['fusion_' + metric] - s[model + '_' + metric]
            delta = np.array([r['fusion_dice'] - r[model + '_dice'] for r in group])
            s['wins_vs_' + model] = int((delta > 1e-12).sum())
            s['ties_vs_' + model] = int((np.abs(delta) <= 1e-12).sum())
            s['losses_vs_' + model] = int((delta < -1e-12).sum())
        result.append(s)
    return result

def preview(output, project, e0, ev, manifest, ref0, ref2, data):
    # Deterministic descriptive selection, not a selection of models or parameters.
    selected = {}
    for ds in DATASETS:
        group = [r for r in data if r['dataset'] == ds]
        picks = [('middle fusion Dice', sorted(group, key=lambda r:(r['fusion_dice'],r['sample_id']))[(len(group)-1)//2]),
                 ('largest gain vs E0', min(group, key=lambda r:(-r['delta_dice_vs_e0'],r['sample_id']))),
                 ('largest drop vs E0', min(group, key=lambda r:(r['delta_dice_vs_e0'],r['sample_id'])))]
        for rule, row in picks:
            selected.setdefault(row['sample_id'], {'row':row, 'rules':[]})['rules'].append(rule)
    dest = output / 'visualizations'; dest.mkdir()
    blocks = []; selections = []
    for idx, (sid, item) in enumerate(selected.items(), 1):
        row = item['row']; rgb, gt, ps, _, _ = audit_sample(project, e0, ev, manifest[sid], ref0[sid], ref2[sid])
        pf = fuse(*ps); labels = ['Image', 'Ground truth', 'E0 Polyp-PVT', 'E2 pretrained U-Net', 'V3 equal fusion']
        masks = [None, gt, ps[0] >= .5, ps[1] >= .5, pf >= .5]
        canvas = Image.new('RGB', (5*300, 260), 'white'); draw = ImageDraw.Draw(canvas)
        for j, (label, mask) in enumerate(zip(labels,masks)):
            im = Image.fromarray(rgb if mask is None else np.where(mask[...,None], np.array([0,220,120],dtype=np.float32)*.45+rgb*.55, rgb).astype(np.uint8))
            im.thumbnail((292,210)); canvas.paste(im,(j*300+(300-im.width)//2,30+(210-im.height)//2))
            draw.text((j*300+8,8),label,fill='black')
        filename = f'{idx:02d}.png'; canvas.save(dest/filename)
        info = dict(sample_id=sid, selection_rules=item['rules'], e0_dice=row['e0_dice'], e2_dice=row['e2_dice'], fusion_dice=row['fusion_dice'])
        selections.append(info)
        label = html.escape(sid + ' | ' + ', '.join(item['rules']))
        blocks.append(f'<section><h2>{label}</h2><p>E0 Dice {row["e0_dice"]:.4f} / E2 {row["e2_dice"]:.4f} / Fusion {row["fusion_dice"]:.4f} / delta vs E0 {row["delta_dice_vs_e0"]:+.4f}</p><img src="visualizations/{filename}" alt="Segmentation comparison"></section>')
    write_json(output/'visualization_selection.json', selections)
    (output/'comparison.html').write_text('<!doctype html><meta charset="utf-8"><title>EndoLens AI V3</title><style>body{font:16px Arial;margin:32px;max-width:1500px;background:#f4f7fa;color:#17324b}section{background:white;padding:20px;margin:20px 0}img{width:100%;height:auto}h2{font-size:18px}</style><h1>V3 fixed equal probability fusion</h1><p>Descriptive examples include gains and drops. No weights or thresholds were selected using these results. Green overlays indicate foreground; frame counts are not lesion counts.</p>' + ''.join(blocks), encoding='utf-8')

def run(args):
    project = Path(args.project).expanduser().resolve()
    resolve = lambda x: Path(x).resolve() if Path(x).is_absolute() else (project/x).resolve()
    e0 = resolve(args.e0_run); e2run = resolve(args.e2_run); ev = resolve(args.e2_eval)
    output = resolve(args.output)
    if output.exists():
        raise FileExistsError('Output already exists; choose a new --output. No overwrite: ' + str(output))
    output.mkdir(parents=True)
    start = time.perf_counter()
    mode = 'AUDIT_ONLY' if args.audit_only else ('SMOKE_ONLY' if args.smoke else 'FULL_V3')
    info = dict(status='RUNNING', run_kind=mode, started_at=now(), project=str(project),
                sources=dict(e0_run=str(e0),e2_run=str(e2run),e2_eval=str(ev)),
                config=dict(weight_e0=.5,weight_e2=.5,threshold=.5,per_image_minmax=False,probability_precision='float32',parameter_search=False),
                environment=dict(python=sys.version,numpy=np.__version__,pillow=Image.__version__,platform=platform.platform(),device='CPU (NumPy cached probability fusion; no training/model inference)'),
                source_hashes={p.name:digest(p) for p in Path(__file__).parent.glob('*.py')})
    write_json(output/'run_info.json', info)
    try:
        manifest, ref0, ref2, checkpoint = validate_sources(project,e0,e2run,ev)
        records = sorted(manifest.values(),key=lambda r:(DATASETS.index(r['dataset']),r['sample_id']))
        if args.smoke and not args.audit_only:
            records = [next(r for r in records if r['dataset']==ds) for ds in DATASETS]
        write_json(output/'manifest.json', records)
        info['e2_checkpoint_sha256']=checkpoint
        info['e0_checkpoint_sha256']=read_json(e0/'run_info.json')['checkpoint_sha256']
        info['source_metadata_sha256']={str(p):digest(p) for p in [e0/'run_info.json',e0/'manifest.json',e0/'per_image.csv',e2run/'run_info.json',e2run/'test_manifest.json',ev/'run_info.json',ev/'per_image.csv']}
        data = []; inputs = []
        for i, record in enumerate(records,1):
            sid=record['sample_id']; rgb,gt,ps,bms,hashes=audit_sample(project,e0,ev,record,ref0[sid],ref2[sid])
            inputs.append(dict(sample_id=sid,e0_probability_sha256=hashes[0],e2_probability_sha256=hashes[1],image_sha256=record['image_sha256'],mask_sha256=record['mask_sha256']))
            if not args.audit_only:
                pf=fuse(*ps); fm=metrics(pf,gt)
                probpath=Path('probabilities')/(sid+'.npz'); maskpath=Path('masks')/(sid+'.png')
                for path in [output/probpath,output/maskpath]:path.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(output/probpath, probability=pf)
                Image.fromarray((pf>=.5).astype(np.uint8)*255).save(output/maskpath)
                row=dict(sample_id=sid,dataset=record['dataset'],height=gt.shape[0],width=gt.shape[1],gt_area_fraction=float(gt.mean()))
                for name,bm in zip(['e0','e2','fusion'],bms+[fm]):row.update({name+'_'+k:v for k,v in bm.items()})
                row.update(delta_dice_vs_e0=fm['dice']-bms[0]['dice'],delta_dice_vs_e2=fm['dice']-bms[1]['dice'],probability_path=probpath.as_posix(),mask_path=maskpath.as_posix())
                data.append(row)
            if i%50==0 or i==len(records):print(f'[{mode} {i}/{len(records)}] {sid}',flush=True)
        write_csv(output/'input_fingerprints.csv',inputs)
        if args.audit_only:
            write_json(output/'audit.json',dict(passed=True,processed_count=len(records),checks='source status/checkpoint/test manifest, image/GT bytes, probability dtype/shape/range, masks, recomputed E0/E2 metrics'))
            print('AUDIT PASSED: all 798 source entries verified. No fusion metrics produced.',flush=True)
        else:
            write_csv(output/'per_image.csv',data); summary=summarize(data); write_csv(output/'summary.csv',summary)
            write_json(output/'metrics.json',dict(run_kind=mode,summary=summary,selection='fixed weights and threshold before observing V3',timing='No model forward or video FPS measured'))
            preview(output,project,e0,ev,manifest,ref0,ref2,data)
            print(mode,flush=True)
            for s in summary:
                print(f"{s['dataset']:20s} n={s['count']:3d} E0={s['e0_dice']:.4f} E2={s['e2_dice']:.4f} Fusion={s['fusion_dice']:.4f} delta_E0={s['delta_dice_vs_e0']:+.4f}",flush=True)
        info.update(status='COMPLETED',processed_count=len(records),finished_at=now(),elapsed_seconds=time.perf_counter()-start,timing='Offline audit/fusion/file output wall time; not model inference or video FPS')
        write_json(output/'run_info.json',info);print('Output:',output,flush=True)
    except Exception as e:
        info.update(status='FAILED',finished_at=now(),error=repr(e));write_json(output/'run_info.json',info);raise

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project',default=str(Path(__file__).resolve().parent.parent))
    ap.add_argument('--e0-run',default='runs/e0_20261004_221345_872444')
    ap.add_argument('--e2-run',default='runs/e2_final_seed42')
    ap.add_argument('--e2-eval',default='runs/e2_final_seed42/evaluation_cuda_20261005_231641_102143')
    ap.add_argument('--output',required=True,help='New output folder relative to --project or absolute')
    flags=ap.add_mutually_exclusive_group();flags.add_argument('--smoke',action='store_true');flags.add_argument('--audit-only',action='store_true')
    run(ap.parse_args())
if __name__=='__main__':main()
