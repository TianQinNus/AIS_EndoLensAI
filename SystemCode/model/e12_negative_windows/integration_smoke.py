"""Exercise the actual train/save/interruption/resume path on a tiny smoke budget.

Only training/validation examples are used. Paths and a SMOKE_ONLY plan are
isolated from all formal outputs. No usable checkpoint can enter E12 selection.
"""
import uuid
from pathlib import Path
import torch
from common import OUTPUT, write_json, read_json, sha, signature
from modeling import build
from evaluation import evaluate


def run(rows, device):
    import training
    folder=OUTPUT/'smoke/integration'/('attempt_'+uuid.uuid4().hex[:10])
    selected=[]
    for source in ['original_training','PolypGen-C1','PolypGen-C2','PolypGen-C3','PolypGen-C4']:
        selected.append(next(r for r in rows['train'] if r['is_positive'] and r['dataset']==source))
    selected.extend([r for r in rows['train'] if not r['is_positive']][:2])
    validation=[next(r for r in rows['validation'] if r['is_positive'] and r['dataset']==s)
                for s in ['original_validation','PolypGen-C5']]
    validation.extend(next(r for r in rows['validation'] if r['label_kind']==kind)
                      for kind in ['author_negative_only_sequence','verified_empty_mask_and_bbox'])
    model=build('r34',device)
    base=folder/'baseline_validation/r34'
    summary,_=evaluate(model,'r34',validation,device,base,prefix='INTEGRATION_SMOKE_VALIDATION')
    del model
    torch.cuda.empty_cache()
    config={'rounds':2,'draws_per_round':8,'input_size':352,'microbatch':2,'accumulation':4,
            'lr':1e-5,'min_lr':1e-6,'weight_decay':1e-4,'run_kind':'SMOKE_ONLY'}
    plan={'training':config,'scope':'SMOKE_ONLY_NOT_A_FORMAL_EXPERIMENT'}
    plan['signature']=signature(plan)
    write_json(base/'run_info.json',{'status':'COMPLETED','plan_signature':plan['signature'],
        'outputs':{'per_image.json':sha(base/'per_image.json')}})
    old_output,old_load,old_draw=training.OUTPUT,training.load_data,training.draw_indices
    interrupted=False
    try:
        training.OUTPUT=folder
        training.load_data=lambda **kwargs:({'signature':'SMOKE_ONLY'}, {'train':selected,'validation':validation})
        def interrupt_second_epoch(records,arm,seed,epoch,count):
            if epoch==2:
                raise KeyboardInterrupt('Intentional smoke interruption after committed round 1')
            return old_draw(records,arm,seed,epoch,count)
        training.draw_indices=interrupt_second_epoch
        try:
            training.train(plan,'r34','mixed',42,workers=0)
        except KeyboardInterrupt:
            interrupted=True
        if not interrupted:
            raise ValueError('The smoke interruption path was not exercised')
        train_folder=training.run_directory('r34','mixed',42)
        state=torch.load(train_folder/'last.pth',map_location='cpu',weights_only=True)
        if state['epoch']!=1 or len(state['history'])!=1:
            raise ValueError('Smoke interruption did not preserve committed round 1')
        if (folder/'gpu.lock').exists() or (train_folder/'run.lock').exists():
            raise ValueError('Interrupted smoke left a live lock')
        del state
        training.draw_indices=old_draw
        training.train(plan,'r34','mixed',42,workers=0)
        final=torch.load(train_folder/'last.pth',map_location='cpu',weights_only=True)
        best=torch.load(train_folder/'best.pth',map_location='cpu',weights_only=True)
        info=read_json(train_folder/'run_info.json')
        if info['status']!='COMPLETED' or [r['round'] for r in final['history']]!=[1,2]:
            raise ValueError('Resume lost or duplicated a training round')
        if sum(r['actual_updates'] for r in final['history'])<1:
            raise ValueError('No real optimizer update in integration smoke')
        if not all(torch.equal(best['model'][k],v) for k,v in final['best_model'].items()):
            raise ValueError('Best checkpoint does not match committed recovery state')
        final_hash=sha(train_folder/'best.pth')
        training.train(plan,'r34','mixed',42,workers=0)
        if sha(train_folder/'best.pth') != final_hash:
            raise ValueError('Repeating a completed command changed the selected checkpoint')
        result={'status':'PASS','scope':'SMOKE_ONLY','rounds':2,'draws_per_round':8,
                'intentional_interruption_after_round':1,'resume_round':2,'history_rounds':[1,2],
                'gpu_optimizer_updates':sum(r['actual_updates'] for r in final['history']),
                'completed_rerun_keeps_checkpoint_unchanged':True,
                'c6_inference':False,'folder':str(folder)}
        write_json(folder/'integration_result.json',result)
        return result
    finally:
        training.OUTPUT,training.load_data,training.draw_indices=old_output,old_load,old_draw
        torch.cuda.empty_cache()
