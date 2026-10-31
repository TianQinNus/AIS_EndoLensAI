"""Windows entry point. Long-running training is started by the user."""
import argparse, gc, json, os, sys, urllib.request
from pathlib import Path
import torch
from experiment import *

def download_encoder():
    ENCODER.parent.mkdir(parents=True,exist_ok=True)
    if ENCODER.exists():
        if sha(ENCODER)!=ENCODER_SHA:raise ValueError('Existing encoder has incorrect full SHA256; preserve and inspect it')
        print('ENCODER_ALREADY_VERIFIED',flush=True);return
    temp=ENCODER.with_suffix('.download')
    print('Downloading official ImageNet V2 ResNet50 encoder (about 98 MiB).',flush=True)
    request=urllib.request.Request(ENCODER_URL,headers={'User-Agent':'EndoLens-E13/1.0'})
    with urllib.request.urlopen(request,timeout=60) as source,temp.open('wb') as target:
        total=0
        while True:
            block=source.read(1024*1024)
            if not block:break
            target.write(block);total+=len(block)
            if total%(10*1024*1024)==0:print(f'[download] {total/2**20:.0f} MiB',flush=True)
    if sha(temp)!=ENCODER_SHA:raise ValueError('Official encoder full SHA256 mismatch; download not promoted')
    os.replace(temp,ENCODER);print('ENCODER_DOWNLOAD_VERIFIED '+ENCODER_SHA,flush=True)

def prepare():
    if (OUTPUT/'plan.json').exists():
        verify_plan();print('E13_ALREADY_PREPARED; immutable plan retained');return
    download_encoder();device=setup()
    with exclusive_lock(ROOT/'runs/e12_negative_comparison_seed42/gpu.lock'),exclusive_lock(OUTPUT/'process.lock'):
        bundle,records=load_data(True)
        for key,path in legacy.WEIGHTS.items():
            if sha(path)!=legacy.EXPECTED_WEIGHTS[key]:raise ValueError('Delivered baseline changed '+key)
        # Same seed initializes one shared random decoder for either optional arm.
        seed_all();model=ARCH.UNet(pretrained_path=ENCODER)
        if model.initial_weight_sha256!=ENCODER_SHA:raise ValueError('Encoder initialization not verified')
        initial=OUTPUT/'initial.pth'
        if initial.exists():
            existing=torch.load(initial,map_location='cpu',weights_only=True)
            if existing['encoder_sha256']!=ENCODER_SHA or any(not torch.equal(v,existing['model'][k]) for k,v in model.state_dict().items()):raise ValueError('Initial checkpoint differs; inspect preparation interruption')
        else:
            save_checkpoint(initial,{'model':cpu_tree(model.state_dict()),'epoch':0,'encoder_sha256':ENCODER_SHA,
                'seed':42,'initialization':RECIPE['initialization'],'parameter_count':sum(p.numel() for p in model.parameters())})
        del model;gc.collect()
        supplied=ARCH.UNet().to(device)
        supplied.load_state_dict(torch.load(STARTER/'weights/best.pt',map_location='cpu',weights_only=True)['model'],strict=True)
        summary,_=evaluate(supplied,'e13',records['validation'],device,mask_size=256,folder=OUTPUT/'reference_validation')
        del supplied;gc.collect();torch.cuda.empty_cache()
        plan={'stage':'E13_IMAGENET_JOINT_POSITIVE_NEGATIVE','created_utc':legacy.utc(),'recipe':RECIPE,
            'data_signature':bundle['signature'],'counts':bundle['counts'],
            'encoder_sha256':ENCODER_SHA,'initial_sha256':sha(initial),
            'reference_summary_sha256':sha(OUTPUT/'reference_validation/summary.json'),
            'reference_positive':summary['positive'],'reference_sources':summary['positive_sources'],
            'baseline_weights':{k:{'path':str(p.relative_to(ROOT)).replace('\\','/'),'sha256':sha(p)} for k,p in legacy.WEIGHTS.items()},
            'code_hashes':code_hashes(),'environment':{'python':sys.version,'torch':str(torch.__version__),
                'gpu':torch.cuda.get_device_name(0),'platform':sys.platform},
            'historical_caveat':'Supplied R50 has a multistage positive-only history and known C6 exposure. E13 versus supplied R50 is a pipeline comparison, not a matched causal ablation.',
            'optional_control':'positive arm uses the exact same initial.pth, recipe and budget; not run by default',
            'selection_caveat':'E13 negative-priority tolerances are new prespecified exploratory choices, not the unchanged E12 gates'}
        plan['signature']=signature(plan);write(OUTPUT/'plan.json',plan)
        print('E13_PREPARED; ImageNet encoder + random decoder; no segmentation training performed',flush=True)
        print(json.dumps({'reference_positive':summary['positive'],'mixed_sampling':'positive50/negative50','epochs':100},indent=2))

def select(arm):
    plan,_=verify_plan();folder=OUTPUT/'training'/arm
    with exclusive_lock(OUTPUT/'process.lock'):
        meta=read(folder/'run_info.json')
        if meta['status']!='COMPLETED' or meta['completed_epochs']!=100:raise ValueError('Complete all 100 epochs before final selection')
        path=folder/'selection.json'
        if path.exists():
            selected=read(path)
            if selected['plan_signature']!=plan['signature']:raise ValueError('Selection plan mismatch')
            for e in selected['entries']:
                if sha(ROOT/e['checkpoint'])!=e['sha256']:raise ValueError('Selected weight changed')
            print('SELECTION_ALREADY_LOCKED '+selected['selected_entry']);return
        entries=[]
        for label in ['positive','tradeoff']:
            digest=meta['best_'+label+'_sha256']
            if digest is None:continue
            cp=folder/('best_'+label+'.pth')
            if sha(cp)!=digest:raise ValueError('Best weight hash mismatch')
            state=torch.load(cp,map_location='cpu',weights_only=True)
            if state['plan_signature']!=plan['signature']:raise ValueError('Best plan mismatch')
            reasons=protection(state['validation'],read(OUTPUT/'reference_validation/summary.json'))
            if label=='tradeoff' and reasons:raise ValueError('Tradeoff weight failed its own protection')
            entries.append({'name':'e13_'+arm+'_best_'+label,'checkpoint':str(cp.relative_to(ROOT)).replace('\\','/'),
                'sha256':digest,'epoch':state['epoch'],'validation':state['validation'],'positive_rejection_reasons':reasons})
        chosen=next((e for e in entries if e['name'].endswith('tradeoff')),entries[0])
        result={'status':'LOCKED_ON_VALIDATION_EXPLORATORY' if not chosen['positive_rejection_reasons'] else 'LOCKED_BEST_POSITIVE_ONLY_NO_PROTECTED_TRADEOFF',
            'created_utc':legacy.utc(),'plan_signature':plan['signature'],'arm':arm,
            'selected_entry':chosen['name'],'entries':entries,'warning':'A best-positive fallback is not a successful negative-priority result.'}
        result['signature']=signature(result);write(path,result)
        print(json.dumps({'status':result['status'],'selected':chosen['name'],'epoch':chosen['epoch'],
            'positive':chosen['validation']['positive'],'negative_seq_fpr':chosen['validation']['negative_sequence_macro_fpr_any']},indent=2))

def evaluate_test(arm,tta):
    plan,records=verify_plan();selection=read(OUTPUT/'training'/arm/'selection.json')
    if signature({k:v for k,v in selection.items() if k!='signature'})!=selection['signature']:raise ValueError('Selection signature mismatch')
    if selection['plan_signature']!=plan['signature']:raise ValueError('Selection plan mismatch')
    device=setup()
    entries=[{'name':k+'_before','key':k,'checkpoint':v['path'],'sha256':v['sha256']} for k,v in plan['baseline_weights'].items()]
    entries += [dict(e,key='e13') for e in selection['entries']]
    from modeling import build as delivered_build
    with exclusive_lock(ROOT/'runs/e12_negative_comparison_seed42/gpu.lock'),exclusive_lock(OUTPUT/'process.lock'):
        for e in entries:
            folder=OUTPUT/'test'/arm/tta/e['name'];cp=ROOT/e['checkpoint']
            if sha(cp)!=e['sha256']:raise ValueError('Evaluation weight changed '+e['name'])
            if (folder/'run_info.json').exists():
                previous=read(folder/'run_info.json')
                if previous.get('status')=='COMPLETED':
                    if previous['selection_signature']!=selection['signature'] or previous['weight_sha256']!=e['sha256']:raise ValueError('Existing test context mismatch')
                    for name,h in previous['outputs'].items():
                        if sha(folder/name)!=h:raise ValueError('Existing test outputs changed')
                    print('TEST_ALREADY_COMPLETED '+e['name']);continue
            if e['key']=='e13':
                model=ARCH.UNet().to(device);model.load_state_dict(torch.load(cp,map_location='cpu',weights_only=True)['model'],strict=True)
            else:model=delivered_build(e['key'],device,cp)
            evaluate(model,e['key'],records['test'],device,tta=tta,folder=folder)
            write(folder/'run_info.json',{'status':'COMPLETED','weight_sha256':e['sha256'],
                'selection_signature':selection['signature'],'plan_signature':plan['signature'],'completed_utc':legacy.utc(),
                'outputs':{name:sha(folder/name) for name in ['summary.json','per_image.json','per_image.csv']}})
            del model;gc.collect();torch.cuda.empty_cache()

def report(arm):
    plan,_=verify_plan();folder=OUTPUT/'training'/arm
    selection=read(folder/'selection.json') if (folder/'selection.json').exists() else None
    rows=[]
    ref=read(OUTPUT/'reference_validation/summary.json')
    summaries=[('r50_supplied','validation320_mask256_identity',ref,None)]
    for key in ['positive','tradeoff']:
        p=folder/('best_'+key+'_validation.json')
        if p.exists():summaries.append(('e13_'+arm+'_best_'+key,'validation320_mask256_identity',read(p),None))
    for tta in ['rotate4','identity']:
        base=OUTPUT/'test'/arm/tta
        if base.exists():
            for f in sorted(base.iterdir()):
                if (f/'run_info.json').exists() and read(f/'run_info.json')['status']=='COMPLETED':summaries.append((f.name,'C6_native320_'+tta,read(f/'summary.json'),f))
    for name,split,s,_ in summaries:
        rows.append({'model':name,'split':split,'positive_count':s['positive']['count'],'positive_dice':s['positive']['dice'],
            'positive_iou':s['positive']['iou'],'low_dice':s['positive']['low_dice'],'near_zero_dice':s['positive']['near_zero_dice'],
            'zero_dice':s['positive']['zero_dice'],'negative_count':s['negative_all']['count'],
            'negative_frame_fpr':s['negative_all']['fpr_any'],'negative_seq_macro_fpr':s['negative_sequence_macro_fpr_any'],
            'negative_mean_fp_area':s['negative_all']['mean_fp_area_fraction'],'negative_seq_macro_fp_area':s['negative_sequence_macro_fp_area']})
    csv_write(OUTPUT/'report'/arm/'comparison.csv',rows)
    text=['# E13 joint-from-ImageNet experiment','',
        'Historical pipeline comparison; C6 has been viewed previously. Only an optional same-start positive arm provides a matched intervention control.',
        '','Primary test protocol: input320, rotate4, native-size masks, threshold0.5. Never compare directly to E12 input352 identity scores.',
        '','| Model | Split | Positive Dice | Positive IoU | Negative frame FPR | Negative seq macro FPR | Zero Dice |',
        '|---|---|---:|---:|---:|---:|---:|']
    text += [f"| {r['model']} | {r['split']} | {r['positive_dice']:.6f} | {r['positive_iou']:.6f} | {r['negative_frame_fpr']:.2%} | {r['negative_seq_macro_fpr']:.2%} | {r['zero_dice']} |" for r in rows]
    if selection:text+=['','Validation decision: '+selection['status']+'; '+selection['selected_entry']]
    target=OUTPUT/'report'/arm/'EXPERIMENT_SUMMARY.md';target.write_text('\n'.join(text)+'\n',encoding='utf-8')
    print(str(target));print(json.dumps(rows,indent=2))

def status():
    print('PLAN', 'LOCKED' if (OUTPUT/'plan.json').exists() else 'NOT_PREPARED')
    for arm in ['mixed50','positive']:
        folder=OUTPUT/'training'/arm;p=folder/'run_info.json'
        if p.exists():
            meta=read(p);print(arm,json.dumps({k:v for k,v in meta.items() if k!='config'}))
            if meta.get('status')=='TRAINING':print('PROCESS_ALIVE',legacy.pid_alive(meta['pid']),'Resume the identical command only after that process has exited.')
        else:print(arm,'NOT_STARTED')

def smoke():
    plan,records=verify_plan();device=setup();folder=OUTPUT/'smoke'
    with exclusive_lock(ROOT/'runs/e12_negative_comparison_seed42/gpu.lock'),exclusive_lock(OUTPUT/'process.lock'):
        seed_all();model=ARCH.UNet().to(device)
        initial=torch.load(OUTPUT/'initial.pth',map_location='cpu',weights_only=True)
        model.load_state_dict(initial['model'],strict=True);model.train()
        original_bn={k:v.clone() for k,v in model.state_dict().items() if 'running_mean' in k or 'running_var' in k}
        pos=next(i for i,r in enumerate(records['train']) if r['is_positive'])
        neg=next(i for i,r in enumerate(records['train']) if not r['is_positive'])
        ds=DrawDataset(records['train'],np.array([pos,neg]),0)
        samples=[ds[0],ds[1]];losses={}
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
        for label,ids in [('positive',[0,0]),('negative',[1,1]),('mixed',[0,1])]:
            optimizer.zero_grad(set_to_none=True)
            x=torch.stack([samples[i][0] for i in ids]).to(device);y=torch.stack([samples[i][1] for i in ids]).to(device)
            loss=mixed_loss(model(x),y)
            if not torch.isfinite(loss):raise ValueError('Nonfinite smoke loss')
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):raise ValueError('Nonfinite smoke gradient')
            optimizer.step();losses[label]=float(loss.detach())
        if any(not torch.equal(v,model.state_dict()[k]) for k,v in original_bn.items()):raise ValueError('Encoder BN statistics updated')
        # Small real-data roundtrip only; no formal training checkpoint is modified.
        save_checkpoint(folder/'roundtrip.pth',{'model':cpu_tree(model.state_dict())})
        clone=ARCH.UNet().to(device);clone.load_state_dict(torch.load(folder/'roundtrip.pth',map_location='cpu',weights_only=True)['model'],strict=True)
        model.eval();clone.eval()
        with torch.inference_mode():
            if not torch.equal(model(x),clone(x)):raise ValueError('Checkpoint roundtrip mismatch')
        evaluate(clone,'e13',[records['validation'][0],next(r for r in records['validation'] if not r['is_positive'])],device,mask_size=256,folder=folder/'validation_fixture')
        result={'status':'PASS','losses':losses,'encoder_bn_stats':'unchanged','checkpoint_roundtrip':'exact',
            'plan_signature':plan['signature'],'note':'Disposable three-step smoke only; no formal training or C6 inference'}
        write(folder/'smoke.json',result);print(json.dumps(result,indent=2))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['download','prepare','smoke','train','select','evaluate','report','status'])
    p.add_argument('--arm',choices=['mixed50','positive'],default='mixed50')
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--tta',choices=['rotate4','identity'],default='rotate4')
    args=p.parse_args()
    if args.command=='download':download_encoder()
    elif args.command=='prepare':prepare()
    elif args.command=='smoke':smoke()
    elif args.command=='train':train(args.arm,args.workers)
    elif args.command=='select':select(args.arm)
    elif args.command=='evaluate':evaluate_test(args.arm,args.tta)
    elif args.command=='report':report(args.arm)
    else:status()

if __name__=='__main__':main()
