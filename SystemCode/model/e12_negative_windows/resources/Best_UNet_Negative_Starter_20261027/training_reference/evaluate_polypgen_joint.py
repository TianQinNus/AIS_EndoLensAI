"""Validation-only TTA selection followed by a locked known-set evaluation."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from resnet50_model import UNet

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT.parents[1]/'outputs/polypgen_joint_20261026/splits'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def save(p, value):
    tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(p)


def score(p, y):
    p = p >= .5
    a, b = np.count_nonzero(p & y), np.count_nonzero(p) + np.count_nonzero(y)
    union = np.count_nonzero(p | y)
    return 2*a/b if b else 1., a/union if union else 1.


def predict(model, x, mode):
    # Average aligned logits; interpolate before sigmoid, matching the baseline.
    if mode == 'identity':
        return model(x)
    if mode == 'flip4':
        dims = [(), (-1,), (-2,), (-2, -1)]
        return torch.stack([model(x.flip(d)).flip(d) if d else model(x)
                            for d in dims]).mean(0)
    return torch.stack([torch.rot90(model(torch.rot90(x, k, (-2, -1))),
                                    -k, (-2, -1)) for k in range(4)]).mean(0)


def evaluate(model, records, size, mode, device, validation=False):
    rows = []
    with torch.inference_mode():
        for i, r in enumerate(records):
            with Image.open(ROOT / r['image']) as im:
                im = im.convert('RGB').resize((size, size), Image.Resampling.BILINEAR)
                x = torch.from_numpy(np.array(im, dtype=np.float32).transpose(2, 0, 1).copy()/255)[None].to(device)
            with Image.open(ROOT / r['mask']) as im:
                raw = np.array(im.convert('L'))
            y = raw > (0 if raw.max() <= 1 else 127)
            if validation:
                y = np.array(Image.fromarray(y).resize((256, 256), Image.Resampling.NEAREST))
            p = F.interpolate(predict(model, x, mode), size=y.shape,
                              mode='bilinear', align_corners=False).sigmoid()[0, 0].cpu().numpy()
            if not np.isfinite(p).all():
                raise ValueError('Non-finite prediction')
            dice, iou = score(p, y)
            rows.append(dict(sample_id=r['sample_id'], dataset=r.get('dataset', r.get('source', 'validation')), dice=dice, iou=iou))
            if (i+1) % 50 == 0:
                print(f'{mode}: {i+1}/{len(records)}', flush=True)
    return rows


def summarize(rows):
    output = {}
    for group in ['ALL'] + sorted({r['dataset'] for r in rows}):
        sub = [r for r in rows if group == 'ALL' or r['dataset'] == group]
        output[group] = dict(count=len(sub), dice=float(np.mean([r['dice'] for r in sub])),
                             iou=float(np.mean([r['iou'] for r in sub])),
                             dice_below_05=sum(r['dice'] < .5 for r in sub),
                             zero_dice=sum(r['dice'] == 0 for r in sub))
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='mps', choices=['mps', 'cuda', 'cpu'])
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    cp = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    config = cp['config']
    model = UNet(config['base']).to(args.device)
    model.load_state_dict(cp['model'], strict=True)
    model.eval()
    records = {n: json.loads((SPLITS / f'{n}.json').read_text()) for n in ['dev_train', 'validation', 'test']}
    for name, rs in records.items():
        for r in rs:
            for key in ['image', 'mask']:
                if sha(ROOT / r[key]) != r[key+'_sha256']:
                    raise ValueError(f'Data hash mismatch: {r[key]}')
    for name in ['dev_train', 'validation']:
        if sha(SPLITS / f'{name}.json') != config[name+'_sha256']:
            raise ValueError('Checkpoint split mismatch')
    ids = {n: {r['decoded_rgb_sha256'] for r in rs} for n, rs in records.items()}
    if ids['dev_train'] & (ids['validation'] | ids['test']) or ids['validation'] & ids['test']:
        raise ValueError('Decoded RGB overlap across splits')
    meta = dict(status='RUNNING', checkpoint=str(args.checkpoint), checkpoint_sha256=sha(args.checkpoint),
                checkpoint_epoch=cp.get('epoch'), checkpoint_config=config,
                source_sha256=sha(Path(__file__)), architecture_sha256=sha(Path(__file__).with_name('resnet50_model.py')),
                manifests={n: sha(SPLITS / f'{n}.json') for n in records},
                candidates=['identity', 'flip4', 'rotate4'], threshold=.5,
                selection='Highest validation per-image binary mean Dice at 256; ties prefer identity',
                evaluation='PolypGen C6 83-image set; initialized from previously trained model; native-mask resolution; image-weighted mean Dice; no test selection',
                torch_version=str(torch.__version__), device=args.device)
    save(args.output/'protocol.json', meta)
    started = time.time()
    try:
        validation = {}
        for mode in meta['candidates']:
            rows = evaluate(model, records['validation'], config['size'], mode, args.device, True)
            validation[mode] = summarize(rows)
            with (args.output/f'validation_{mode}.csv').open('w') as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
            save(args.output/'validation.json', validation)
            print(json.dumps({mode: validation[mode]}), flush=True)
        selected = max(meta['candidates'], key=lambda m: validation[m]['ALL']['dice'])
        save(args.output/'locked_selection.json', dict(mode=selected, validation=validation,
                                                       checkpoint_sha256=meta['checkpoint_sha256']))
        rows = evaluate(model, records['test'], config['size'], selected, args.device)
        with (args.output/'c6_evaluation.csv').open('w') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        summary = summarize(rows)
        save(args.output/'summary.json', summary)
        meta.update(status='COMPLETED', selected=selected, target_reached=summary['ALL']['dice'] >= .85)
        print(json.dumps(summary), flush=True)
    except BaseException as e:
        meta.update(status='FAILED', error=repr(e))
        raise
    finally:
        meta['elapsed_seconds'] = time.time()-started
        save(args.output/'protocol.json', meta)


if __name__ == '__main__':
    main()
