"""Matched low-LR continuation / photometric adaptation of experiment 14.

No test images are loaded. Run both arms with the same initialization and budget.
"""
import argparse
import importlib.util
import json
import math
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance
import torch
from torch.utils.data import DataLoader

from resnet50_model import UNet
from evaluate_tta import ROOT, SPLITS, sha, save


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--photometric', action='store_true')
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default='mps')
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    if min(args.epochs, args.lr, args.batch_size) <= 0:
        p.error('Positive training settings required')
    source = ROOT/'unetpp_resolution/train.py'
    spec = importlib.util.spec_from_file_location('historical_data', source)
    data = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(data)
    if args.photometric:
        original = data.augment_pair

        def augment(image, mask):
            image, mask = original(image, mask)
            if random.random() < .5:
                image = ImageEnhance.Color(image).enhance(random.uniform(.5, 1.5))
            if random.random() < .5:
                gamma = random.uniform(.7, 1.5)
                image = Image.fromarray(np.round((np.array(image)/255.)**gamma*255).astype(np.uint8))
            return image, mask

        data.augment_pair = augment
    torch.set_num_threads(4)
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    if args.device == 'mps':
        torch.mps.manual_seed(42)
    initial = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    old = initial['config']
    records = {}
    for name in ['dev_train', 'validation']:
        manifest = SPLITS/f'{name}.json'
        if sha(manifest) != old[name+'_sha256']:
            raise ValueError('Split mismatch')
        records[name] = json.loads(manifest.read_text())
        for r in records[name]:
            for key in ['image', 'mask']:
                if sha(ROOT/r[key]) != r[key+'_sha256']:
                    raise ValueError('Data mismatch: '+r[key])
    config = dict(old, epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
                  device=args.device, photometric=args.photometric, training_sizes=[320], loss_policy="original BCE+Dice+.2boundary", microbatch_size=2,
                  initialization_sha256=sha(args.checkpoint), initialization_epoch=initial['epoch'],
                  source_sha256=sha(Path(__file__)), data_source_sha256=sha(source),
                  architecture_sha256=sha(Path(__file__).with_name('resnet50_model.py')),
                  schedule='cosine from initial LR to 10%, new AdamW optimizer',
                  selection='maximum validation Dice, including unchanged initial checkpoint',
                  added_augmentation='p=.5 saturation U(.5,1.5); p=.5 gamma U(.7,1.5)' if args.photometric else 'none',
                  torch_version=str(torch.__version__))
    args.output.mkdir(parents=True, exist_ok=args.resume)
    model = UNet(old['base']).to(args.device)
    model.load_state_dict(initial['model'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    train = DataLoader(data.Pairs(records['dev_train'], old['size'], True), batch_size=args.batch_size, shuffle=True)
    valid = DataLoader(data.Pairs(records['validation'], old['size'], False, 256), batch_size=args.batch_size)
    history, start = [], 1

    def validation():
        model.eval(); total = 0.
        with torch.inference_mode():
            for x, y in valid:
                x, y = x.to(args.device), y.to(args.device).bool()
                pred = torch.nn.functional.interpolate(model(x), y.shape[-2:], mode='bilinear', align_corners=False).sigmoid() >= .5
                axes = (1, 2, 3)
                denominator = pred.sum(axes)+y.sum(axes)
                total += torch.where(denominator > 0, 2*(pred & y).sum(axes)/denominator.clamp_min(1), 1.).sum().item()
        return total/len(valid.dataset)

    if args.resume:
        cp = torch.load(args.output/'last.pt', map_location='cpu', weights_only=False)
        if cp['config'] != config:
            raise ValueError('Resume source/config mismatch')
        model.load_state_dict(cp['model']); optimizer.load_state_dict(cp['optimizer'])
        random.setstate(cp['python_rng']); np.random.set_state(cp['numpy_rng']); torch.set_rng_state(cp['torch_rng'])
        if args.device == 'mps':
            torch.mps.set_rng_state(cp['mps_rng'])
        if args.device == 'cuda':
            torch.cuda.set_rng_state_all(cp['cuda_rng'])
        start, history, best = cp['epoch']+1, cp['history'], cp['best_dice']
    else:
        best = validation()
        data.save_checkpoint(args.output/'best.pt', dict(model=initial['model'], config=config, epoch=0, val_dice=best))
    save(args.output/'config.json', config)
    status = dict(status='RUNNING', initial_val_dice=initial['val_dice'], best_val_dice=best)
    try:
        for epoch in range(start, args.epochs+1):
            begin = time.time()
            lr = args.lr*(.1+.9*(1+math.cos(math.pi*(epoch-1)/max(args.epochs-1, 1)))/2)
            for group in optimizer.param_groups:
                group['lr'] = lr
            model.train(); total = 0.
            for step, (x, y) in enumerate(train, 1):
                optimizer.zero_grad(set_to_none=True)
                training_size = 320
                x = torch.nn.functional.interpolate(x, (training_size, training_size), mode="bilinear", align_corners=False)
                y = torch.nn.functional.interpolate(y, (training_size, training_size), mode="nearest")
                for offset in range(0, len(x), 2):
                    mx, my = x[offset:offset+2].to(args.device), y[offset:offset+2].to(args.device)
                    loss = data.loss_fn(model(mx), my)
                    if not torch.isfinite(loss):
                        raise ValueError('Non-finite loss')
                    (loss * (len(mx)/len(x))).backward()
                    total += loss.item()*len(mx)
                    del loss, mx, my
                optimizer.step()
                if step == 1 or step % 10 == 0:
                    status.update(epoch=epoch, step=step)
                    save(args.output/'status.json', status)
                    print(f'epoch {epoch}/{args.epochs} step {step}/{len(train)}', flush=True)
            dice = validation()
            weights = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            if dice > best:
                best = dice
                data.save_checkpoint(args.output/'best.pt', dict(model=weights, config=config, epoch=epoch, val_dice=best))
            history.append(dict(epoch=epoch, train_loss=total/len(train.dataset), val_dice=dice, lr=lr, seconds=time.time()-begin))
            data.save_checkpoint(args.output/'last.pt', dict(model=weights, config=config, epoch=epoch,
                optimizer=optimizer.state_dict(), best_dice=best, history=history,
                python_rng=random.getstate(), numpy_rng=np.random.get_state(), torch_rng=torch.get_rng_state(),
                mps_rng=torch.mps.get_rng_state() if args.device == 'mps' else None,
                cuda_rng=torch.cuda.get_rng_state_all() if args.device == 'cuda' else None))
            save(args.output/'history.json', history)
            status.update(completed_epochs=epoch, best_val_dice=best)
            save(args.output/'status.json', status)
            print(json.dumps(history[-1]), flush=True)
        status['status'] = 'COMPLETED'
    except BaseException as e:
        status.update(status='FAILED', error=repr(e)); raise
    finally:
        save(args.output/'status.json', status)


if __name__ == '__main__':
    main()
