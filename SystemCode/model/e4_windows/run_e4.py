"""E4 controlled strong-augmentation experiment using the unchanged E2 engine."""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

CORE = Path(__file__).resolve().parent / 'core'
sys.path.insert(0, str(CORE))
import torch
import numpy as np
from run_e2 import make_config, select_recipe
from engine import code_hashes, require_cuda, load_checkpoint
from data import load_bundle, local_path
from models import read_pretrained
from e0_protocol import file_hash, write_json, write_csv

BASELINE='runs/e2_dev_C_resnet_pretrained_seed42'
DEV='runs/e4_dev_strong_seed42'
BUNDLE='runs/e2_data_seed42'
WEIGHTS='e2_weights/resnet34-b627a593.pth'
RECIPE='runs/e4_selection_seed42.json'

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def read_csv(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def config(smoke=False):
    return make_config(argparse.Namespace(experiment='pretrained',smoke=smoke,epochs=100,batch_size=2,accumulation=4,seed=42,no_amp=False,augmentation='strong',encoder_weights=WEIGHTS))

def assert_control(base, candidate):
    expected=dict(config(),augmentation='flips')
    if base != expected or candidate != config():
        raise ValueError('Controlled configuration differs. Preserve the matched E2-C recipe; only augmentation may change.')
    differences=[k for k in base if base[k]!=candidate[k]]
    if differences != ['augmentation']:raise ValueError('Not an augmentation-only comparison: '+str(differences))

def completed_dev(project, path, expected_config):
    info=read(path/'run_info.json')
    if info['status']!='TRAINED' or info['run_kind']!='E2_DEVELOPMENT' or info['completed_epochs']!=100:
        raise ValueError('Expected completed 100-epoch formal development run: '+str(path))
    if info['config']!=expected_config or info['source_hashes']!=code_hashes():
        raise ValueError('Development configuration/source mismatch: '+str(path))
    if file_hash(path/'checkpoints/best.pth')!=info['best_checkpoint_sha256']:
        raise ValueError('Best checkpoint hash mismatch')
    history=read_csv(path/'train_history.csv')
    if len(history)!=100 or [int(r['epoch']) for r in history]!=list(range(1,101)):
        raise ValueError('Incomplete development history')
    best=max(history,key=lambda r:float(r['validation_dice']))
    if int(best['epoch'])!=info['best_epoch'] or abs(float(best['validation_dice'])-info['best_validation_dice'])>1e-12:
        raise ValueError('Validation selection inconsistent')
    cp=load_checkpoint(path/'checkpoints/best.pth')
    if cp['config']!=info['config'] or cp['epoch']!=info['best_epoch'] or cp['bundle_signature']!=info['bundle_signature']:
        raise ValueError('Best checkpoint provenance mismatch')
    del cp
    vr=read_csv(path/'validation_best_per_image.csv')
    if len(vr)!=290 or len({r['sample_id'] for r in vr})!=290:
        raise ValueError('Expected 290 unique validation entries')
    if abs(float(np.mean([float(r['dice']) for r in vr]))-info['best_validation_dice'])>1e-12:
        raise ValueError('Best validation metrics inconsistent')
    return info,history,vr

def preflight(project, skip_gpu=False):
    if not skip_gpu:require_cuda()
    else:print('READ-ONLY MAC AUDIT: CUDA not tested. Not permission to train on CPU.',flush=True)
    baseline=local_path(project,BASELINE)
    base,_,vr=completed_dev(project,baseline,dict(config(),augmentation='flips'))
    bundle,records=load_bundle(local_path(project,BUNDLE),project)
    if bundle['bundle_signature']!=base['bundle_signature']:raise ValueError('Internal split changed')
    if {r['sample_id'] for r in vr}!={r['sample_id'] for r in records['validation']}:
        raise ValueError('Baseline validation IDs differ from frozen bundle')
    _,sha=read_pretrained(local_path(project,WEIGHTS))
    if sha!=base['initial_encoder_sha256']:raise ValueError('Pretrained encoder weights differ from E2-C')
    assert_control(base['config'],config())
    print('PASS: unchanged E2 core, matched C baseline, frozen split, encoder initialization, augmentation-only configuration.',flush=True)
    print('Training/validation:',len(records['dev_train']),len(records['validation']),flush=True)

def core_call(project, command, extras):
    cmd=[sys.executable,str(CORE/'run_e2.py'),command,'--project',str(project),*extras]
    print('Running:',subprocess.list2cmdline(cmd),flush=True)
    subprocess.run(cmd,check=True)

def compare(project,output,dev_value=DEV):
    base,bh,bv=completed_dev(project,local_path(project,BASELINE),dict(config(),augmentation='flips'))
    strong,sh,sv=completed_dev(project,local_path(project,dev_value),config())
    assert_control(base['config'],strong['config'])
    if base['bundle_signature']!=strong['bundle_signature'] or base['initial_encoder_sha256']!=strong['initial_encoder_sha256']:
        raise ValueError('Initialization/data split differs')
    if {r['sample_id'] for r in bv}!={r['sample_id'] for r in sv}:raise ValueError('Validation membership differs')
    if output.exists():raise FileExistsError('Comparison folder exists; select a new --output')
    recipe=local_path(project,RECIPE)
    if recipe.exists():raise FileExistsError('Selection recipe exists; preserve it. No overwrite.')
    output.mkdir(parents=True)
    table=[]
    for label,info,history,vr in [('E2_C_flips',base,bh,bv),('E4_strong',strong,sh,sv)]:
        dice=np.array([float(r['dice']) for r in vr])
        table.append(dict(experiment=label,best_epoch=info['best_epoch'],best_validation_dice=info['best_validation_dice'],best_validation_iou=float(np.mean([float(r['iou']) for r in vr])),median_dice=float(np.median(dice)),dice_lt05=int((dice<.5).sum()),zero_dice=int((dice==0).sum()),epoch100_validation_dice=float(history[-1]['validation_dice']),last20_mean_validation_dice=float(np.mean([float(r['validation_dice']) for r in history[-20:]])),amp_skipped_updates=sum(int(r['amp_skipped_updates']) for r in history)))
    write_csv(output/'comparison.csv',table)
    bm={r['sample_id']:r for r in bv}
    paired=[dict(sample_id=r['sample_id'],baseline_dice=float(bm[r['sample_id']]['dice']),strong_dice=float(r['dice']),delta_dice=float(r['dice'])-float(bm[r['sample_id']]['dice'])) for r in sv]
    write_csv(output/'validation_paired.csv',paired)
    delta=strong['best_validation_dice']-base['best_validation_dice']
    write_json(output/'comparison.json',dict(stage='E4_DEVELOPMENT_COMPARISON',selection_basis='internal validation only',delta_best_validation_dice=delta,baseline=table[0],strong=table[1],can_consider_final=delta>0,interpretation='Positive delta permits considering final retraining, not proof of robust gain. Inspect curve/tail and repeat a seed if small or unstable. No test scores used.',wrapper_sha256=file_hash(Path(__file__)),core_hashes=code_hashes()))
    select_recipe(project,[BASELINE,dev_value],recipe)
    print('Delta strong vs flips:',f'{delta:+.6f}',flush=True)
    print('NEXT: review comparison/validation curves before final retraining.' if delta>0 else 'NO VALIDATION IMPROVEMENT: retain E2/E3; do not run E4 final.',flush=True)
    print('Comparison:',output,flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['preflight','check-gpu','smoke','train','resume','compare','train-final','evaluate'])
    p.add_argument('--project',default=str(Path(__file__).resolve().parent.parent))
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--output')
    p.add_argument('--run-dir')
    p.add_argument('--skip-gpu',action='store_true',help='Read-only preflight on Mac, never enables CPU training')
    a=p.parse_args();project=Path(a.project).expanduser().resolve()
    if a.workers<0:p.error('workers must be nonnegative')
    if a.skip_gpu and a.command!='preflight':p.error('--skip-gpu is only valid for read-only preflight')
    if a.command=='preflight':preflight(project,a.skip_gpu)
    elif a.command=='check-gpu':
        out=local_path(project,a.output or 'runs/e4_gpu_device_proof.json')
        if out.exists():raise FileExistsError('GPU proof exists; choose new --output')
        subprocess.run([sys.executable,str(CORE/'check_device.py'),'--device','cuda','--output',str(out)],check=True,cwd=project)
    elif a.command in ['train','smoke']:
        preflight(project)
        out=a.output or ('runs/e4_smoke_seed42' if a.command=='smoke' else DEV)
        cmd=['--experiment','pretrained','--augmentation','strong','--epochs','100','--batch-size','2','--accumulation','4','--seed','42','--device','cuda','--encoder-weights',WEIGHTS,'--bundle',BUNDLE,'--workers',str(a.workers),'--output',out]
        if a.command=='smoke':cmd+=['--smoke']
        core_call(project,'train',cmd)
        write_json(local_path(project,out)/'e4_stage.json',dict(stage='E4_SMOKE' if a.command=='smoke' else 'E4_DEVELOPMENT',wrapper_sha256=file_hash(Path(__file__)),core_unchanged=True,comparison='E2 C flips vs strong; all other settings fixed'))
    elif a.command=='resume':
        path=local_path(project,a.run_dir or DEV);saved=read(path/'run_info.json')['config']
        if saved not in [config(),dict(config(),phase='final',epochs=saved['epochs'])]:raise ValueError('Not an E4 strong development/final configuration')
        core_call(project,'resume',['--run-dir',str(path),'--bundle',BUNDLE,'--device','cuda','--workers',str(a.workers)])
    elif a.command=='compare':compare(project,local_path(project,a.output or 'runs/e4_comparison_seed42'),a.run_dir or DEV)
    elif a.command=='train-final':
        recipe=read(local_path(project,RECIPE))
        if recipe['config']['augmentation']!='strong':
            raise ValueError('E4 strong was not selected. Preserve E2; no E4 full retraining is needed.')
        out=a.output or 'runs/e4_final_seed42'
        core_call(project,'train-final',['--recipe',RECIPE,'--bundle',BUNDLE,'--device','cuda','--workers',str(a.workers),'--output',out])
        write_json(local_path(project,out)/'e4_stage.json',dict(stage='E4_FINAL',wrapper_sha256=file_hash(Path(__file__)),core_unchanged=True))
    else:
        core_call(project,'evaluate',['--run-dir',a.run_dir or 'runs/e4_final_seed42','--device','cuda','--e0-run','runs/e0_20261004_221345_872444'])
if __name__=='__main__':main()
