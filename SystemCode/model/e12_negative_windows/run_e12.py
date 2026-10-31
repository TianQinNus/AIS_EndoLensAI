"""Windows CLI for E12 mixed-label model comparison. No silent test-set selection."""
import argparse
import gc
import os
import platform
import subprocess
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from common import (ROOT, PACKAGE, DATA, OUTPUT, WEIGHTS, EXPECTED_WEIGHTS, STARTER,
    read_json, write_json, write_csv, sha, signature, code_hashes, verify_weights,
    load_data, exclusive_lock, safe_path, utc)
from modeling import setup, build, forward, input_tensor, mixed_loss, train_mode
from evaluation import evaluate
from metrics import positive_protection, checkpoint_rank
from training import train, run_directory, atomic_save, cpu_tree

PLAN = OUTPUT/'plan.json'
ARMS = ('positive', 'mixed')
KEYS = ('pvt', 'r34', 'r50')


def environment():
    import torchvision
    from PIL import __version__ as pillow_version
    return {'python':sys.version, 'executable':sys.executable, 'platform':platform.platform(),
        'torch':str(torch.__version__), 'torchvision':str(torchvision.__version__),
        'numpy':np.__version__, 'pillow':pillow_version, 'cuda':torch.version.cuda,
        'cudnn':torch.backends.cudnn.version(),
        'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


def locked_plan():
    plan = read_json(PLAN)
    plain = {k:v for k,v in plan.items() if k!='signature'}
    if signature(plain)!=plan['signature']:
        raise ValueError('Locked plan was edited')
    if plan['code_hashes']!=code_hashes():
        raise ValueError('Code changed after locking; preserve existing experiment before designing a revision')
    if plan['weights']!=verify_weights():
        raise ValueError('Initial model identities changed')
    info, _ = load_data()
    if plan['data_signature']!=info['signature']:
        raise ValueError('Data signature changed')
    current = environment()
    for key in ['python','executable','torch','torchvision','numpy','pillow','cuda','cudnn','gpu']:
        if current[key]!=plan['environment'][key]:
            raise ValueError('Runtime changed after plan lock: '+key)
    return plan


def preflight():
    device = setup()
    verify_weights()
    info, _ = load_data(verify_bytes=True)
    if PLAN.exists():
        plan = locked_plan()
        print('EXISTING_PLAN_VERIFIED',plan['signature'],flush=True)
        return
    smoke_info = read_json(OUTPUT/'smoke/run_info.json')
    if smoke_info.get('status')!='PASS' or smoke_info.get('code_hashes')!=code_hashes():
        raise ValueError('Run smoke and unit tests with the final code before locking')
    if smoke_info.get('data_signature')!=info['signature']:
        raise ValueError('Smoke used a different data split')
    plan = {'stage':'E12_NEGATIVE_MIXTURE_COMPARISON','created_utc':utc(),
        'protocol_version':'E12-v1.1-before-formal-training',
        'pre_formal_revision':'The final positive-protection reference is the strongest frozen U-Net, not all frozen models. Original290 is not verified unseen by public PVT. Initial common validation outputs had already been viewed. Numeric protection gates and inference/training settings are unchanged; old plan and baseline metadata are preserved. No formal continuation or new C6 inference preceded this revision.',
        'data_signature':info['signature'], 'counts':info['counts'],
        'weights':verify_weights(), 'code_hashes':code_hashes(), 'environment':environment(),
        'primary_evaluation':{'input_size':352,'tta':'identity','threshold':.5,
            'mask_resolution':'native GT shape','logit_resize':'bilinear align_corners=False before sigmoid',
            'per_image_minmax':False,'connected_component_filter':False,
            'negative_primary':'sequence-equal any-foreground FPR; mean FP image-area fraction as tie-break',
            'positive_primary':'source-equal per-image Dice; original290 and C5_204 reported separately'},
        'training':{'rounds':30,'draws_per_round':1160,'input_size':352,
            'microbatch':2,'accumulation':4,'lr':1e-5,'min_lr':1e-6,'weight_decay':1e-4,
            'amp':True,'encoder_bn':'running stats frozen; affine trainable; both encoders/decoders train',
            'loss':'mean_images(BCEWithLogits + I(nonempty augmented GT)*SoftDice(smooth1)); no boundary term',
            'augmentation':'paired H/V flips p0.5, rotation U(-15,15), brightness/contrast U(.85,1.15); no shrink/pad/saturation/gamma',
            'positive_sampling':'conditional original=.5, C1-C4=.125 each, uniform within source',
            'negative_sampling':'uniform over audited train-negative images; not center equal',
            'mixed_mass':{'positive':.75,'negative':.25},
            'control_mass':{'positive':1.,'negative':0.},
            'draws':'with replacement; stochastic class/source counts, not fixed per-batch quotas',
            'seeds':[42,17,2024], 'screening_seed':42},
        'checkpoint_selection':{'reference':'own frozen model under common validation protocol',
            'max_positive_macro_dice_drop':.01,'max_per_source_dice_drop':.02,
            'max_low_count_increase':2,'max_zero_count_increase':1,
            'rank':'min negative-sequence macro any-FPR, then min negative macro area, then max positive macro Dice',
            'epoch_zero_eligible':True,'exact_tie':'earlier round'},
        'final_selection':{'positive_reference':'highest common frozen U-Net validation positive macro Dice; PVT remains a comparison entry, not the positive anchor',
            'protection':'same declared positive protection limits',
            'negative_training_gain':'mixed must reduce any-FPR at least 0.10 vs BOTH own frozen baseline and positive-only continuation; FP area must not increase vs either',
            'choice':'validation only; test table cannot switch the chosen model',
            'secondary_tta':'optional rotate4, same352 and same mode for all seven entries, no tuning'},
        'limitations':info['limitations']+['R50 and R34 histories differ; this compares delivered pipelines and matched continuation effects, not a clean architecture ablation.',
            'Original290 is an internal U-Net validation split; frozen public PVT or inherited lineages may have seen original-source images. A common evaluation list does not establish common historical exposure.',
            'A 30-round seed42 screen is not a three-seed replication or clinical deployment validation.',
            'No mixed positive/negative Dice is used to select a model.']}
    plan['signature']=signature(plan)
    write_json(PLAN,plan)
    print('PREFLIGHT_AND_PLAN_LOCK_PASS',plan['signature'],str(device),'NO_C6_MODEL_METRICS_READ',flush=True)


def smoke():
    """Small real train/validation-only GPU checks. Never saves a usable trained model."""
    device=setup()
    info, rows=load_data()
    folder=OUTPUT/'smoke'
    with exclusive_lock(OUTPUT/'gpu.lock'):
        records=[next(r for r in rows['train'] if r['is_positive']),
                 next(r for r in rows['train'] if not r['is_positive'])]
        validation=[next(r for r in rows['validation'] if r['is_positive']),
                    next(r for r in rows['validation'] if r['label_kind']=='author_negative_only_sequence'),
                    next(r for r in rows['validation'] if r['label_kind']=='verified_empty_mask_and_bbox')]
        results={}
        for key in KEYS:
            print('SMOKE',key,flush=True)
            model=build(key,device)
            summary,_=evaluate(model,key,validation,device,folder/key,prefix='SMOKE_VALIDATION_ONLY')
            item={'validation_images':len(validation),'positive_dice':summary['positive']['dice']}
            if key!='pvt':
                optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5,weight_decay=1e-4)
                scaler=torch.amp.GradScaler('cuda',init_scale=1024)
                train_mode(model)
                from training import DrawDataset
                data=DrawDataset(records,np.array([0,1]),42,1)
                x=torch.stack([data[i][0] for i in range(2)]).to(device)
                y=torch.stack([data[i][1] for i in range(2)]).to(device)
                before=model.head.weight.detach().clone()
                for label,target in [('mixed',y),('all_negative',torch.zeros_like(y))]:
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast('cuda',dtype=torch.float16):
                        logits=forward(model,key,x)
                    loss,_,_=mixed_loss(logits,target)
                    if not torch.isfinite(loss):
                        raise ValueError('Smoke loss not finite')
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                        raise ValueError('Smoke gradients not finite')
                    scaler.step(optimizer);scaler.update()
                    item[label+'_loss']=float(loss.detach())
                if torch.equal(before,model.head.weight):
                    raise ValueError('Smoke optimizer did not update model')
                checkpoint=folder/key/'smoke_roundtrip.pth'
                atomic_save({'model':cpu_tree(model.state_dict()),'optimizer':cpu_tree(optimizer.state_dict()),
                             'scaler':scaler.state_dict()},checkpoint)
                saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
                clone=build(key,device,checkpoint)
                for a,b in zip(model.state_dict().values(),clone.state_dict().values()):
                    if not torch.equal(a,b):
                        raise ValueError('State roundtrip mismatch')
                item['optimizer_updates']=2
                item['checkpoint_roundtrip']='PASS'
                del optimizer,clone,saved,x,y,logits,loss
            results[key]=item
            del model
            gc.collect();torch.cuda.empty_cache()
        # Run meaningful unit tests as part of the smoke command.
        subprocess.run([sys.executable,str(PACKAGE/'test_e12.py')],cwd=ROOT,check=True)
        from integration_smoke import run as integration_test
        results['actual_train_interrupt_resume']=integration_test(rows,device)
        write_json(folder/'run_info.json',{'status':'PASS','run_kind':'SMOKE_ONLY_NOT_FORMAL_TRAINING',
            'results':results,'data_signature':info['signature'],'code_hashes':code_hashes(),
            'environment':environment(),'c6_model_inference':False,'created_utc':utc()})
        print('SMOKE_PASS; no historical weights changed; no C6 inference',flush=True)


def baseline():
    plan=locked_plan()
    device=setup()
    _,rows=load_data()
    with exclusive_lock(OUTPUT/'gpu.lock'):
        for key in KEYS:
            folder=OUTPUT/'baseline_validation'/key
            meta=folder/'run_info.json'
            if meta.exists():
                old=read_json(meta)
                if old.get('status')=='COMPLETED' and old.get('plan_signature')==plan['signature']:
                    for name in ['summary.json','per_image.csv','per_image.json']:
                        if sha(folder/name)!=old['outputs'][name]:
                            raise ValueError('Baseline output changed')
                    print('BASELINE_ALREADY_COMPLETED',key,flush=True)
                    continue
                raise ValueError('Existing partial baseline; inspect instead of overwriting')
            model=build(key,device)
            evaluate(model,key,rows['validation'],device,folder,prefix='frozen_validation')
            write_json(meta,{'status':'COMPLETED','plan_signature':plan['signature'],
                'weight_sha256':sha(WEIGHTS[key]),'model_key':key,'created_utc':utc(),
                'outputs':{name:sha(folder/name) for name in ['summary.json','per_image.csv','per_image.json']}})
            del model
            gc.collect();torch.cuda.empty_cache()
    print('BASELINE_VALIDATION_COMPLETE; C6 not evaluated',flush=True)


def comparison_entries(plan, seed=42):
    result=[]
    for key in KEYS:
        folder=OUTPUT/'baseline_validation'/key
        meta=read_json(folder/'run_info.json')
        if meta.get('plan_signature')!=plan['signature'] or meta.get('status')!='COMPLETED':
            raise ValueError('Incomplete frozen validation')
        if sha(folder/'summary.json')!=meta['outputs']['summary.json']:
            raise ValueError('Frozen validation summary changed')
        result.append({'name':key+'_before','model_key':key,'arm':'frozen','checkpoint':str(WEIGHTS[key]),
                       'weight_sha256':sha(WEIGHTS[key]),'epoch':None,'summary':read_json(folder/'summary.json')})
    for key in ['r34','r50']:
        for arm in ARMS:
            folder=run_directory(key,arm,seed)
            meta=read_json(folder/'run_info.json')
            if meta.get('status')!='COMPLETED' or meta.get('plan_signature')!=plan['signature']:
                raise ValueError('All four controlled continuations must finish before selection')
            if sha(folder/'best.pth')!=meta['best_sha256']:
                raise ValueError('Training best checkpoint changed')
            cp=torch.load(folder/'best.pth',map_location='cpu',weights_only=True)
            if cp['validation']!=read_json(folder/'best_validation_summary.json'):
                raise ValueError('Best checkpoint/validation sidecar mismatch')
            result.append({'name':key+'_'+arm,'model_key':key,'arm':arm,'seed':seed,
                'checkpoint':str(folder/'best.pth'),'weight_sha256':meta['best_sha256'],
                'epoch':meta['best_epoch'],'summary':cp['validation']})
            del cp
    return result


def select():
    plan=locked_plan()
    path=OUTPUT/'selection.json'
    if path.exists():
        decision=read_json(path)
        if decision.get('plan_signature')!=plan['signature']:
            raise ValueError('Selection belongs to another plan')
        print('SELECTION_ALREADY_LOCKED',decision['selected_name'],flush=True)
        return
    entries=comparison_entries(plan)
    frozen=[e for e in entries if e['arm']=='frozen']
    reference=max([e for e in frozen if e['model_key'] in ['r34','r50']],
                  key=lambda e:e['summary']['positive_source_macro_dice'])
    for entry in entries:
        entry['positive_rejection_reasons']=positive_protection(entry['summary'],reference['summary'])
        entry['eligible']=not entry['positive_rejection_reasons']
    eligible=[e for e in entries if e['eligible']]
    chosen=max(eligible,key=lambda e:checkpoint_rank(e['summary']))
    effects={}
    for key in ['r34','r50']:
        before=next(e for e in entries if e['name']==key+'_before')['summary']
        control=next(e for e in entries if e['name']==key+'_positive')['summary']
        mixed=next(e for e in entries if e['name']==key+'_mixed')['summary']
        gain=(not positive_protection(mixed,before) and not positive_protection(mixed,control)
              and mixed['negative_sequence_macro_fpr_any']<=before['negative_sequence_macro_fpr_any']-.10
              and mixed['negative_sequence_macro_fpr_any']<=control['negative_sequence_macro_fpr_any']-.10
              and mixed['negative_sequence_macro_fp_area']<=before['negative_sequence_macro_fp_area']
              and mixed['negative_sequence_macro_fp_area']<=control['negative_sequence_macro_fp_area'])
        effects[key]={'declared_negative_training_gain_pass':gain,
            'fpr_any_delta_vs_frozen':mixed['negative_sequence_macro_fpr_any']-before['negative_sequence_macro_fpr_any'],
            'fpr_any_delta_vs_positive_control':mixed['negative_sequence_macro_fpr_any']-control['negative_sequence_macro_fpr_any'],
            'positive_macro_delta_vs_positive_control':mixed['positive_source_macro_dice']-control['positive_source_macro_dice']}
    decision={'status':'VALIDATION_SELECTION_LOCKED_EXPLORATORY','created_utc':utc(),
        'plan_signature':plan['signature'],'positive_reference':reference['name'],
        'selected_name':chosen['name'],'selected_checkpoint':chosen['checkpoint'],
        'selected_weight_sha256':chosen['weight_sha256'], 'negative_training_effects':effects,
        'entries':entries,'test_metrics_used':False,
        'warning':'Validation winner is a development candidate, not demonstrated negative-video safety; C6 cannot switch this selection.'}
    decision['signature']=signature(decision)
    write_json(path,decision)
    write_json(OUTPUT/'selected_model.json',{k:chosen[k] for k in ['name','model_key','arm','checkpoint','weight_sha256','epoch']})
    print('VALIDATION_SELECTION_LOCKED',chosen['name'],'negative_training_effects=',effects,flush=True)


def evaluate_test(tta='identity'):
    plan=locked_plan()
    decision=read_json(OUTPUT/'selection.json')
    if decision.get('plan_signature')!=plan['signature'] or signature({k:v for k,v in decision.items() if k!='signature'})!=decision['signature']:
        raise ValueError('Invalid locked selection; test is blocked')
    _,rows=load_data(verify_bytes=True)
    device=setup()
    with exclusive_lock(OUTPUT/'gpu.lock'):
        for entry in decision['entries']:
            folder=OUTPUT/('test_identity' if tta=='identity' else 'test_rotate4_secondary')/entry['name']
            meta_path=folder/'run_info.json'
            if meta_path.exists():
                old=read_json(meta_path)
                if old.get('status')=='COMPLETED' and old.get('selection_signature')==decision['signature']:
                    for name in ['summary.json','per_image.csv','per_image.json']:
                        if sha(folder/name)!=old['outputs'][name]:
                            raise ValueError('Completed test output changed')
                    print('TEST_ALREADY_COMPLETED',entry['name'],flush=True)
                    continue
                raise ValueError('Partial test exists; inspect before recovery')
            if sha(entry['checkpoint'])!=entry['weight_sha256']:
                raise ValueError('Locked evaluation checkpoint changed')
            model=build(entry['model_key'],device,Path(entry['checkpoint']) if entry['arm']!='frozen' else None)
            evaluate(model,entry['model_key'],rows['test'],device,folder,prefix='C6_HISTORICAL_'+entry['name'],tta=tta)
            write_json(meta_path,{'status':'COMPLETED','selection_signature':decision['signature'],
                'plan_signature':plan['signature'],'entry':{k:v for k,v in entry.items() if k!='summary'},
                'tta':tta,'input_size':352,'completed_utc':utc(),
                'outputs':{name:sha(folder/name) for name in ['summary.json','per_image.csv','per_image.json']}})
            del model
            gc.collect();torch.cuda.empty_cache()
    print('COMMON_TEST_COMPLETED; validation selection unchanged',flush=True)


def report():
    from report_results import report as create_report
    create_report(locked_plan())


def status():
    print('E12 experiment:',OUTPUT)
    print('Data ready:',(DATA/'bundle.json').exists(),'Plan locked:',PLAN.exists())
    for key in ['r34','r50']:
        for arm in ARMS:
            path=run_directory(key,arm,42)/'run_info.json'
            if not path.exists():
                print(key,arm,'NOT_STARTED')
            else:
                state=read_json(path)
                print(key,arm,state['status'],'committed_rounds=',state.get('completed_rounds'),
                      'best=',state.get('best_epoch'))
    selection=OUTPUT/'selection.json'
    if selection.exists():print('Validation-selected model:',read_json(selection)['selected_name'])


def infer(image_path, output_path):
    from modeling import probabilities
    plan=locked_plan()
    decision=read_json(OUTPUT/'selection.json')
    if decision['plan_signature']!=plan['signature']:
        raise ValueError('Selection/plan mismatch')
    entry=next(e for e in decision['entries'] if e['name']==decision['selected_name'])
    if sha(entry['checkpoint'])!=entry['weight_sha256']:
        raise ValueError('Selected weight changed')
    device=setup()
    with exclusive_lock(OUTPUT/'gpu.lock'):
        model=build(entry['model_key'],device,Path(entry['checkpoint']) if entry['arm']!='frozen' else None)
        with Image.open(image_path) as im:
            rgb=im.convert('RGB');shape=(rgb.height,rgb.width)
            x=input_tensor(rgb).unsqueeze(0).to(device)
        with torch.inference_mode():
            p=probabilities(model,entry['model_key'],x,shape)
        path=Path(output_path);path.parent.mkdir(parents=True,exist_ok=True)
        Image.fromarray((p>=.5).astype(np.uint8)*255).save(path)
        write_json(path.with_suffix('.json'),{'model':entry['name'],'checkpoint_sha256':entry['weight_sha256'],
            'input_size':352,'threshold':.5,'tta':'identity','any_foreground':bool((p>=.5).any()),
            'foreground_area_fraction':float((p>=.5).mean()),'foreground_score_is_clinical_probability':False})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    for name in ['prepare','preflight','smoke','baseline','train-all','select','report','status']:
        p=sub.add_parser(name)
        if name=='train-all':p.add_argument('--workers',type=int,default=2)
    p=sub.add_parser('train');p.add_argument('--model',choices=['r34','r50'],required=True)
    p.add_argument('--arm',choices=ARMS,required=True);p.add_argument('--seed',type=int,choices=[42,17,2024],default=42)
    p.add_argument('--workers',type=int,default=2)
    p=sub.add_parser('evaluate-test');p.add_argument('--tta',choices=['identity','rotate4'],default='identity')
    p=sub.add_parser('infer');p.add_argument('--image',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if getattr(args,'workers',0)<0:parser.error('workers must be nonnegative')
    if args.command=='prepare':
        import resources,prepare_data
        resources.prepare();prepare_data.prepare()
    elif args.command=='preflight':preflight()
    elif args.command=='smoke':smoke()
    elif args.command=='baseline':baseline()
    elif args.command=='train':train(locked_plan(),args.model,args.arm,args.seed,args.workers)
    elif args.command=='train-all':
        locked_plan()
        for key in ['r34','r50']:
            for arm in ARMS:
                subprocess.run([sys.executable,str(Path(__file__)),'train','--model',key,'--arm',arm,
                                '--workers',str(args.workers)],cwd=ROOT,check=True)
    elif args.command=='select':select()
    elif args.command=='evaluate-test':evaluate_test(args.tta)
    elif args.command=='report':report()
    elif args.command=='status':status()
    elif args.command=='infer':infer(args.image,args.output)


if __name__=='__main__':
    main()
