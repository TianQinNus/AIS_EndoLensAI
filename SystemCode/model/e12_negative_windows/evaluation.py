"""Shared full-resolution evaluation without reading any training-time test data."""
import time
import numpy as np
from PIL import Image
import torch
from common import safe_path, write_csv, write_json
from modeling import input_tensor, probabilities
from metrics import per_image, summarize


def read_mask(record):
    with Image.open(safe_path(record['mask'])) as im:
        gt = np.asarray(im.convert('L'), np.float32)>127.5
    if bool(gt.any()) != bool(record['is_positive']):
        raise ValueError('Label polarity changed: ' + record['sample_id'])
    return gt


def evaluate(model, key, records, device, folder=None, prefix='validation', tta='identity', size=352, progress=True):
    model.eval()
    result = []
    started = time.perf_counter()
    with torch.inference_mode():
        for i, record in enumerate(records, 1):
            gt = read_mask(record)
            with Image.open(safe_path(record['image'])) as im:
                x = input_tensor(im, size).unsqueeze(0).to(device)
            p = probabilities(model, key, x, gt.shape, tta)
            result.append(per_image(p, gt, record))
            if progress and (i%100==0 or i==len(records)):
                print(f'[{prefix} {key}] {i}/{len(records)} elapsed={time.perf_counter()-started:.1f}s', flush=True)
    summary = summarize(result)
    summary['protocol'] = {'input': size, 'tta': tta, 'threshold': .5,
        'native_logit_resize_before_sigmoid': True, 'image_minmax': False,
        'connected_component_filter': False, 'evaluation_seconds': time.perf_counter()-started}
    if folder is not None:
        write_csv(folder/'per_image.csv', result)
        write_json(folder/'per_image.json', result)
        write_json(folder/'summary.json', summary)
    return summary, result
