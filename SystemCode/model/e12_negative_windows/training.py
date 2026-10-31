"""Matched positive-control and negative-mixture continuations of frozen U-Nets."""
import gc
import math
import os
import time
from pathlib import Path
import numpy as np
from PIL import Image, ImageEnhance
import torch
from torch.utils.data import Dataset, DataLoader
from common import (ROOT, OUTPUT, WEIGHTS, sha, safe_path, read_json, write_json,
                    write_csv, signature, load_data, say, exclusive_lock, utc)
from modeling import setup, build, forward, train_mode, input_tensor, mixed_loss
from evaluation import evaluate
from metrics import positive_protection, checkpoint_rank


def sampling_probabilities(records, arm):
    positive = [i for i, r in enumerate(records) if r['is_positive']]
    negative = [i for i, r in enumerate(records) if not r['is_positive']]
    p = np.zeros(len(records), np.float64)
    positive_mass = 1. if arm == 'positive' else .75
    if arm not in ['positive', 'mixed'] or not positive or not negative:
        raise ValueError('Invalid training arm or missing label stratum')
    strata = {}
    for i in positive:
        strata.setdefault(records[i]['dataset'], []).append(i)
    if set(strata) != {'original_training', 'PolypGen-C1', 'PolypGen-C2', 'PolypGen-C3', 'PolypGen-C4'}:
        raise ValueError('Positive training membership changed')
    for name, indices in strata.items():
        p[indices] = positive_mass*(.5 if name=='original_training' else .125)/len(indices)
    if arm == 'mixed':
        p[negative] = .25/len(negative)
    if not np.isclose(p.sum(), 1.):
        raise ValueError('Sampling probabilities do not sum to 1')
    return p


def draw_indices(records, arm, seed, epoch, count=1160):
    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch, 1212]))
    return rng.choice(len(records), count, replace=True, p=sampling_probabilities(records, arm))


class DrawDataset(Dataset):
    def __init__(self, records, indices, seed, epoch, size=352):
        self.records, self.indices = records, indices
        self.seed, self.epoch, self.size = seed, epoch, size

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        row = self.records[int(self.indices[index])]
        with Image.open(safe_path(row['image'])) as im:
            image = im.convert('RGB')
        with Image.open(safe_path(row['mask'])) as im:
            mask = Image.fromarray((np.asarray(im.convert('L'), np.float32)>127.5).astype(np.uint8)*255)
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, index, 1213]))
        for operation in [Image.Transpose.FLIP_LEFT_RIGHT, Image.Transpose.FLIP_TOP_BOTTOM]:
            if rng.random()<.5:
                image, mask = image.transpose(operation), mask.transpose(operation)
        angle = float(rng.uniform(-15, 15))
        image = image.rotate(angle, resample=Image.Resampling.BILINEAR)
        mask = mask.rotate(angle, resample=Image.Resampling.NEAREST)
        image = ImageEnhance.Brightness(image).enhance(float(rng.uniform(.85, 1.15)))
        image = ImageEnhance.Contrast(image).enhance(float(rng.uniform(.85, 1.15)))
        y = np.asarray(mask.resize((self.size, self.size), Image.Resampling.NEAREST), np.uint8)>127
        return input_tensor(image, self.size), torch.from_numpy(y.astype(np.float32)[None].copy())


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    return value


def atomic_save(value, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+'.tmp')
    torch.save(value, temp)
    os.replace(temp, path)


def run_directory(key, arm, seed):
    return OUTPUT/'training'/f'{key}_{arm}_seed{seed}'


def write_best(folder, state):
    atomic_save({'model': state['best_model'], 'model_key': state['config']['model_key'],
        'epoch': state['best_epoch'], 'config': state['config'],
        'plan_signature': state['plan_signature'], 'validation': state['best_summary']}, folder/'best.pth')
    write_csv(folder/'best_validation.csv', state['best_rows'])
    write_json(folder/'best_validation_summary.json', state['best_summary'])


def train(plan, key, arm, seed=42, workers=2):
    if key not in ['r34', 'r50']:
        raise ValueError('PVT is a frozen reference; no PVT training command')
    device = setup()
    info, records = load_data(verify_bytes=True)
    folder = run_directory(key, arm, seed)
    config = {**plan['training'], 'model_key': key, 'arm': arm, 'seed': seed,
              'data_signature': info['signature'], 'initial_sha256': sha(WEIGHTS[key])}
    base_folder = OUTPUT/'baseline_validation'/key
    base_meta = read_json(base_folder/'run_info.json')
    if base_meta.get('status')!='COMPLETED' or base_meta.get('plan_signature')!=plan['signature']:
        raise ValueError('Run baseline-validation first under the locked plan')
    reference = read_json(base_folder/'summary.json')
    with exclusive_lock(OUTPUT/'gpu.lock'), exclusive_lock(folder/'run.lock'):
        if (folder/'run_info.json').exists():
            completed = read_json(folder/'run_info.json')
            if completed.get('status') == 'COMPLETED':
                if (completed.get('config') != config or completed.get('plan_signature') != plan['signature']
                        or completed.get('completed_rounds') != config['rounds']
                        or sha(folder/'best.pth') != completed['best_sha256']):
                    raise ValueError('Completed run integrity mismatch')
                say(folder, 'ALREADY_COMPLETED; checkpoints unchanged; no new training')
                return
        model = build(key, device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
        scaler = torch.amp.GradScaler('cuda')
        last = folder/'last.pth'
        if last.exists():
            state = torch.load(last, map_location='cpu', weights_only=True)
            if state['config']!=config or state['plan_signature']!=plan['signature']:
                raise ValueError('Resume configuration/plan mismatch; preserve original run')
            model.load_state_dict(state['model'], strict=True)
            optimizer.load_state_dict(state['optimizer'])
            scaler.load_state_dict(state['scaler'])
            torch.set_rng_state(state['torch_rng'])
            torch.cuda.set_rng_state_all(state['cuda_rng'])
            start = state['epoch']+1
            write_best(folder, state)  # restore best consistently from committed last
            say(folder, f'RESUME {key}/{arm} seed={seed} next_round={start}')
        else:
            if (folder/'run_info.json').exists():
                raise ValueError('Run metadata exists without last checkpoint; inspect rather than restart')
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            state = {'config': config, 'plan_signature': plan['signature'], 'epoch': 0,
                'best_model': cpu_tree(model.state_dict()), 'best_epoch': 0,
                'best_summary': reference, 'best_rows': [], 'history': []}
            # Exact same frozen weights and locked evaluation protocol; reuse the
            # already verified baseline rather than repeating full inference.
            initial_rows = read_json(base_folder/'per_image.json')
            if sha(base_folder/'per_image.json') != base_meta['outputs']['per_image.json']:
                raise ValueError('Initial validation rows changed')
            state.update(best_summary=reference, best_rows=initial_rows)
            write_best(folder, state)
            state.update(model=cpu_tree(model.state_dict()), optimizer=cpu_tree(optimizer.state_dict()),
                         scaler=scaler.state_dict(), torch_rng=torch.get_rng_state(),
                         cuda_rng=torch.cuda.get_rng_state_all())
            atomic_save(state, last)  # epoch-zero recovery also covers first-round interruption
            start = 1
        epochs = config['rounds']
        if start>epochs:
            write_json(folder/'run_info.json', {'status':'COMPLETED','config':config,
                'plan_signature':plan['signature'], 'completed_rounds':state['epoch'],
                'best_epoch':state['best_epoch'], 'best_sha256':sha(folder/'best.pth'), 'updated_utc':utc()})
            say(folder, 'ALREADY_COMPLETED; no new training')
            return
        committed_epoch = state['epoch']
        try:
            for epoch in range(start, epochs+1):
                write_json(folder/'run_info.json', {'status':'TRAINING','config':config,
                    'plan_signature':plan['signature'], 'completed_rounds':epoch-1, 'pid':os.getpid(), 'updated_utc':utc()})
                started = time.perf_counter()
                train_mode(model)
                lr = config['min_lr']+(config['lr']-config['min_lr'])*.5*(1+math.cos(math.pi*(epoch-1)/max(1,epochs-1)))
                for group in optimizer.param_groups:
                    group['lr'] = lr
                indices = draw_indices(records['train'], arm, seed, epoch, config['draws_per_round'])
                data = DrawDataset(records['train'], indices, seed, epoch, config['input_size'])
                loader = DataLoader(data, batch_size=config['microbatch'], shuffle=False, num_workers=workers,
                    pin_memory=True, generator=torch.Generator().manual_seed(seed+epoch))
                optimizer.zero_grad(set_to_none=True)
                total_loss = total_bce = total_dice = 0.
                seen = updates = skipped = 0
                torch.cuda.reset_peak_memory_stats()
                say(folder, f'ROUND {epoch}/{epochs} {key}/{arm} lr={lr:.3g} draws={len(indices)}')
                for step, (x, y) in enumerate(loader, 1):
                    x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                    group_start = ((step-1)//config['accumulation'])*config['microbatch']*config['accumulation']
                    group_images = min(config['microbatch']*config['accumulation'], len(indices)-group_start)
                    with torch.autocast('cuda', dtype=torch.float16):
                        logits = forward(model, key, x)
                    loss, bce, dice = mixed_loss(logits, y)
                    if not torch.isfinite(loss):
                        raise ValueError('Non-finite loss; preserve checkpoint and investigate')
                    scaler.scale(loss*len(x)/group_images).backward()
                    total_loss += float(loss.detach())*len(x)
                    total_bce += float(bce.detach())*len(x)
                    total_dice += float(dice.detach())*len(x)
                    seen += len(x)
                    if step%config['accumulation']==0 or step==len(loader):
                        previous = scaler.get_scale()
                        scaler.step(optimizer)
                        scaler.update()
                        if scaler.get_scale()<previous:
                            skipped += 1
                        else:
                            updates += 1
                        optimizer.zero_grad(set_to_none=True)
                    if step%40==0 or step==len(loader):
                        say(folder, f'  train {seen}/{len(indices)} loss={total_loss/seen:.5f} updates={updates} skipped={skipped}')
                summary, rows = evaluate(model, key, records['validation'], device, prefix=f'round{epoch}_validation')
                reasons = positive_protection(summary, reference)
                improved = not reasons and checkpoint_rank(summary)>checkpoint_rank(state['best_summary'])
                if improved:
                    state.update(best_model=cpu_tree(model.state_dict()), best_epoch=epoch,
                                 best_summary=summary, best_rows=rows)
                sampled_negative = sum(not records['train'][int(i)]['is_positive'] for i in indices)
                history_row = {'round':epoch, 'train_loss':total_loss/seen, 'train_bce':total_bce/seen,
                    'train_positive_dice_term':total_dice/seen, 'lr':lr, 'negative_draws':sampled_negative,
                    'positive_draws':seen-sampled_negative, 'actual_updates':updates, 'amp_skips':skipped,
                    'positive_macro_dice':summary['positive_source_macro_dice'],
                    'positive_image_dice':summary['positive']['dice'], 'positive_low_dice':summary['positive']['low_dice'],
                    'positive_zero_dice':summary['positive']['zero_dice'],
                    'negative_seq_macro_fpr_any':summary['negative_sequence_macro_fpr_any'],
                    'negative_seq_macro_fp_area':summary['negative_sequence_macro_fp_area'],
                    'eligible_positive_protection':not reasons, 'rejection_reasons':';'.join(reasons),
                    'selected_this_round':improved, 'best_round':state['best_epoch'],
                    'draw_indices_sha256':__import__('hashlib').sha256(indices.astype('<i8').tobytes()).hexdigest(),
                    'peak_gpu_gib':torch.cuda.max_memory_allocated()/2**30,
                    'round_seconds':time.perf_counter()-started}
                state['history'].append(history_row)
                state.update(epoch=epoch, model=cpu_tree(model.state_dict()), optimizer=cpu_tree(optimizer.state_dict()),
                    scaler=scaler.state_dict(), torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all())
                atomic_save(state, last)  # canonical recovery point includes the matching best model
                committed_epoch = epoch
                if improved:
                    write_best(folder, state)
                write_csv(folder/'history.csv', state['history'])
                write_json(folder/'last_validation_summary.json', summary)
                say(folder, f'ROUND_DONE {epoch} posDice={summary["positive_source_macro_dice"]:.6f} '
                    f'negativeFPR={summary["negative_sequence_macro_fpr_any"]:.4f} '
                    f'negativeArea={summary["negative_sequence_macro_fp_area"]:.6f} '
                    f'best={state["best_epoch"]} seconds={history_row["round_seconds"]:.1f}')
            write_best(folder, state)
            write_json(folder/'run_info.json', {'status':'COMPLETED','config':config,
                'plan_signature':plan['signature'], 'completed_rounds':epochs,
                'best_epoch':state['best_epoch'], 'best_sha256':sha(folder/'best.pth'), 'updated_utc':utc()})
            say(folder, 'TRAINING_COMPLETED; no C6 inference performed')
        except BaseException as error:
            write_json(folder/'interruption.json', {'type':type(error).__name__, 'message':str(error),
                'last_committed_round':committed_epoch, 'utc':utc(),
                'recovery':'Rerun the identical command; a live process lock will block duplicate work.'})
            raise
        finally:
            del model, optimizer
            gc.collect()
            torch.cuda.empty_cache()
