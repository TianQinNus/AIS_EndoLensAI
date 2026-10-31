"""E13: clean ImageNet-to-segmentation training with teammate's 320 recipe.

Earlier experiments are immutable inputs. No trained segmentation weight is
ever used to initialize an E13 training run.
"""
from pathlib import Path
import contextlib, csv, gc, hashlib, importlib.util, json, math, os, random, sys, time
from datetime import datetime, timezone
import numpy as np
from PIL import Image, ImageEnhance
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

ROOT=Path(__file__).resolve().parent.parent
PACKAGE=Path(__file__).resolve().parent
OUTPUT=ROOT/'runs/e13_joint_seed42'
DATA=ROOT/'runs/e12_negative_data_seed42'
LEGACY=ROOT/'e12_negative_windows'
STARTER=LEGACY/'resources/Best_UNet_Negative_Starter_20261027'
ENCODER=PACKAGE/'resources/resnet50-11ad3fa6.pth'
ENCODER_SHA='11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca'
ENCODER_URL='https://download.pytorch.org/models/resnet50-11ad3fa6.pth'
sys.path.insert(0,str(LEGACY))
import common as legacy
from metrics import per_image, summarize

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result
ARCH=module('e13_teammate_architecture',STARTER/'model/resnet50_model.py')
HISTORICAL=module('e13_teammate_training',STARTER/'training_reference/historical_data_train.py')
sha=legacy.sha
read=legacy.read_json
write=legacy.write_json
signature=legacy.signature
csv_write=legacy.write_csv
exclusive_lock=legacy.exclusive_lock
safe_path=legacy.safe_path

RECIPE={
    'epochs':100,'draws_per_epoch':1160,'seed':42,'input_size':320,
    'validation_mask_size':256,'microbatch':2,'effective_batch':8,
    'lr':1e-4,'min_lr':1e-5,'weight_decay':1e-4,'precision':'FP32; TF32 disabled; no AMP',
    'architecture':'Supplied ResNet50 encoder + GN8 U-Net decoder 256/128/64/32/16',
    'initialization':'Official ImageNet V2 encoder; seed42 random decoder; no segmentation checkpoint',
    'normalization':'RGB /255; embedded ImageNet mean/std once',
    'encoder_bn':'Frozen running statistics; affine trainable; encoder and decoder trainable',
    'augmentation':'Resize320 first; H/V p.5; quarter-turn; brightness/contrast .8..1.2; shrink+zero-pad p.5 scale .4..8; saturation p.5 .5..1.5; gamma p.5 .7..1.5',
    'loss':'Original BCEWithLogits + per-image soft Dice(smooth1) + .2 inner boundary Dice(smooth1), INCLUDING empty targets',
    'positive_sampling':'Conditional original .5; C1-C4 .125 each; uniform within source',
    'negative_sampling':'Uniform over all1907 audited train-negative images; not sequence-balanced',
    'mixed_negative_mass':.5,'control_negative_mass':0.,
    'draw_policy':'1160 with replacement; stochastic class counts; fixed compute, not positive-exposure matched',
    'best_positive':'Maximum validation positive image-mean Dice, then earlier epoch',
    'best_tradeoff':'Minimum negative sequence-macro any-FPR, then minimum sequence-macro FP area, then maximum positive image Dice, among protected candidates',
    'positive_protection':{'max_image_dice_drop':.02,'max_source_dice_drop':.03,'max_low_count_increase':5,'max_zero_count_increase':3},
    'reference':'Supplied R50 evaluated with input320 / mask256 / identity; validation only',
    'evaluation_primary':'All compared weights input320 / native mask / rotate4 / threshold.5 / FP32',
    'evaluation_secondary':'Same locked weights input320 / native mask / identity / threshold.5',
    'test_role':'C6 historically reused exploratory evaluation; never selection or training',
}

def setup(device='cuda'):
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('Activate endolens-e2-win; CUDA is required.')
    return torch.device(device)

def seed_all(seed=42):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)

def cpu_tree(x):
    if isinstance(x,torch.Tensor):return x.detach().cpu().clone()
    if isinstance(x,dict):return {k:cpu_tree(v) for k,v in x.items()}
    if isinstance(x,list):return [cpu_tree(v) for v in x]
    if isinstance(x,tuple):return tuple(cpu_tree(v) for v in x)
    return x

def save_checkpoint(path,state):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp');torch.save(state,temp);os.replace(temp,path)

def say(folder,message):legacy.say(folder,message)

def code_hashes():
    paths=list(PACKAGE.glob('*.py'))+[STARTER/'model/resnet50_model.py',
        STARTER/'training_reference/historical_data_train.py',STARTER/'training_config.json',
        LEGACY/'common.py',LEGACY/'metrics.py',LEGACY/'modeling.py',
        ROOT/'lib/pvt.py',ROOT/'lib/pvtv2.py',ROOT/'e6_windows/pvt_timm_compat.py']
    paths=[p for p in paths if p.is_file()]
    return {str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in sorted(paths)}

def load_data(verify_images=False):
    bundle,records=legacy.load_data(False)
    train,valid,test=(records[k] for k in ['train','validation','test'])
    ids=[set(r['sample_id'] for r in rows) for rows in [train,valid,test]]
    if ids[0]&ids[1] or ids[0]&ids[2] or ids[1]&ids[2]:raise ValueError('Split overlap')
    if verify_images:
        # C6 pixel data and metrics are never opened by prepare or train.
        for role in ['train','validation']:
            for i,r in enumerate(records[role],1):
                for k in ['image','mask']+(['bbox'] if 'bbox' in r else []):
                    if sha(safe_path(r[k]))!=r[k+'_sha256']:raise ValueError('Changed data '+r['sample_id'])
                if i%500==0:print(f'[verify {role}] {i}/{len(records[role])}',flush=True)
    return bundle,records

def verify_plan():
    plan=read(OUTPUT/'plan.json')
    if signature({k:v for k,v in plan.items() if k!='signature'})!=plan['signature']:raise ValueError('Plan signature mismatch')
    if plan['recipe']!=RECIPE or plan['code_hashes']!=code_hashes():raise ValueError('Locked E13 code/recipe changed; preserve the run and use a new experiment ID')
    bundle,records=load_data(False)
    if plan['data_signature']!=bundle['signature']:raise ValueError('Data signature changed')
    if sha(ENCODER)!=ENCODER_SHA or sha(OUTPUT/'initial.pth')!=plan['initial_sha256']:raise ValueError('Initialization changed')
    if sha(OUTPUT/'reference_validation/summary.json')!=plan['reference_summary_sha256']:raise ValueError('Validation anchor changed')
    return plan,records

def draw_indices(records,arm,epoch,count=1160):
    mass={'mixed50':.5,'positive':0.}[arm]
    probs=np.zeros(len(records),np.float64)
    sources={s:[i for i,r in enumerate(records) if r['is_positive'] and r['dataset']==s]
        for s in ['original_training','PolypGen-C1','PolypGen-C2','PolypGen-C3','PolypGen-C4']}
    for s,indices in sources.items():
        if not indices:raise ValueError('Missing positive training source')
        probs[indices]=(1-mass)*(.5 if s=='original_training' else .125)/len(indices)
    negative=[i for i,r in enumerate(records) if not r['is_positive']]
    if not negative:raise ValueError('Missing training negatives')
    probs[negative]=mass/len(negative)
    if not np.isclose(probs.sum(),1):raise ValueError('Invalid sampling mass')
    rng=np.random.default_rng(np.random.SeedSequence([42,epoch,1313]))
    return rng.choice(len(records),count,replace=True,p=probs)

def augment_pair(image,mask):
    # Reuse the supplied exact geometric/brightness/contrast/shrink operations.
    image,mask=HISTORICAL.augment_pair(image,mask)
    if random.random()<.5:image=ImageEnhance.Color(image).enhance(random.uniform(.5,1.5))
    if random.random()<.5:
        gamma=random.uniform(.7,1.5)
        image=Image.fromarray(np.round((np.asarray(image)/255.)**gamma*255).astype(np.uint8))
    return image,mask

class DrawDataset(Dataset):
    def __init__(self,records,indices,epoch):self.records,self.indices,self.epoch=records,indices,epoch
    def __len__(self):return len(self.indices)
    def __getitem__(self,index):
        r=self.records[int(self.indices[index])]
        with Image.open(safe_path(r['image'])) as im:image=im.convert('RGB').resize((320,320),Image.Resampling.BILINEAR)
        with Image.open(safe_path(r['mask'])) as im:a=np.asarray(im.convert('L'))
        gt=a>(0 if a.max()<=1 else 127)
        if bool(gt.any())!=bool(r['is_positive']):raise ValueError('Label polarity changed '+r['sample_id'])
        mask=Image.fromarray(gt.astype(np.uint8)*255).resize((320,320),Image.Resampling.NEAREST)
        previous=random.getstate()
        try:
            random.seed((42*1000003+self.epoch)*1000003+index)
            image,mask=augment_pair(image,mask)
        finally:random.setstate(previous)
        x=np.asarray(image,np.float32).transpose(2,0,1)/255
        y=(np.asarray(mask)>127).astype(np.float32)[None]
        return torch.from_numpy(x.copy()),torch.from_numpy(y.copy())

def mixed_loss(logits,target):
    # Smooth1 makes the original Dice and boundary terms finite on zero masks.
    return HISTORICAL.loss_fn(logits.float(),target.float())

def input_tensor(image):
    a=np.asarray(image.convert('RGB').resize((320,320),Image.Resampling.BILINEAR),np.float32)/255
    return torch.from_numpy(np.ascontiguousarray(a.transpose(2,0,1)))

def evaluate(model,key,records,device,mask_size=None,tta='identity',folder=None,progress=True):
    model.eval();rows=[];started=time.perf_counter()
    from modeling import forward as delivered_forward
    def forward(x):return model(x) if key=='e13' else delivered_forward(model,key,x)
    with torch.inference_mode():
        for i,r in enumerate(records,1):
            with Image.open(safe_path(r['mask'])) as im:
                a=np.asarray(im.convert('L'));gt=a>(0 if a.max()<=1 else 127)
            if bool(gt.any())!=bool(r['is_positive']):raise ValueError('Changed label '+r['sample_id'])
            if mask_size is not None:gt=np.asarray(Image.fromarray(gt.astype(np.uint8)*255).resize((mask_size,mask_size),Image.Resampling.NEAREST))>127
            with Image.open(safe_path(r['image'])) as im:x=input_tensor(im).unsqueeze(0).to(device)
            if tta=='rotate4':z=sum(torch.rot90(forward(torch.rot90(x,k,(-2,-1))),-k,(-2,-1)) for k in range(4))/4
            elif tta=='identity':z=forward(x)
            else:raise ValueError('Unknown TTA')
            probability=F.interpolate(z.float(),size=gt.shape,mode='bilinear',align_corners=False).sigmoid()[0,0].cpu().numpy()
            rows.append(per_image(probability,gt,r))
            if progress and (i%200==0 or i==len(records)):print(f'[eval {key}/{tta}] {i}/{len(records)} {time.perf_counter()-started:.1f}s',flush=True)
    summary=summarize(rows)
    summary['protocol']={'input':320,'mask_size':mask_size or 'native','tta':tta,'threshold':.5,
        'image_minmax':False,'component_filter':False,'evaluation_seconds':time.perf_counter()-started}
    if folder is not None:
        write(folder/'summary.json',summary);write(folder/'per_image.json',rows);csv_write(folder/'per_image.csv',rows)
    return summary,rows

def protection(summary,reference):
    g=RECIPE['positive_protection'];reasons=[]
    if summary['positive']['dice']<reference['positive']['dice']-g['max_image_dice_drop']:reasons.append('positive_image_dice_drop_gt_0.02')
    for source,ref in reference['positive_sources'].items():
        if summary['positive_sources'][source]['dice']<ref['dice']-g['max_source_dice_drop']:reasons.append('source_dice_drop_gt_0.03:'+source)
    if summary['positive']['low_dice']>reference['positive']['low_dice']+g['max_low_count_increase']:reasons.append('low_dice_increase_gt_5')
    if summary['positive']['zero_dice']>reference['positive']['zero_dice']+g['max_zero_count_increase']:reasons.append('zero_dice_increase_gt_3')
    return reasons

def tradeoff_rank(s):return (-s['negative_sequence_macro_fpr_any'],-s['negative_sequence_macro_fp_area'],s['positive']['dice'])

def write_best(folder,state):
    for key in ['positive','tradeoff']:
        best=state['best_'+key]
        if best is not None:
            save_checkpoint(folder/('best_'+key+'.pth'),{'model':best['model'],'epoch':best['epoch'],
                'config':state['config'],'plan_signature':state['plan_signature'],'validation':best['summary']})
            write(folder/('best_'+key+'_validation.json'),best['summary'])
            csv_write(folder/('best_'+key+'_validation.csv'),best['rows'])

def rng_state():
    return {'python':random.getstate(),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()}
def restore_rng(state):
    random.setstate(state['python']);torch.set_rng_state(state['torch']);torch.cuda.set_rng_state_all(state['cuda'])

def train(arm='mixed50',workers=2):
    plan,records=verify_plan();load_data(True);device=setup();folder=OUTPUT/'training'/arm
    if workers<0:raise ValueError('workers must be nonnegative')
    reference=read(OUTPUT/'reference_validation/summary.json')
    config={**RECIPE,'arm':arm,'plan_signature':plan['signature'],'initial_sha256':plan['initial_sha256']}
    with exclusive_lock(LEGACY.parent/'runs/e12_negative_comparison_seed42/gpu.lock'),exclusive_lock(OUTPUT/'process.lock'):
        if (folder/'run_info.json').exists():
            completed=read(folder/'run_info.json')
            if completed.get('status')=='COMPLETED':
                if completed['config']!=config or completed['completed_epochs']!=100:raise ValueError('Completed run configuration mismatch')
                for key in ['positive','tradeoff']:
                    expected=completed['best_'+key+'_sha256']
                    if expected is not None and sha(folder/('best_'+key+'.pth'))!=expected:raise ValueError('Completed checkpoint changed')
                say(folder,'ALREADY_COMPLETED; no weights modified; no duplicate training');return
        seed_all();model=ARCH.UNet().to(device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=RECIPE['lr'],weight_decay=RECIPE['weight_decay'])
        path=folder/'last.pth'
        if path.exists():
            state=torch.load(path,map_location='cpu',weights_only=True)
            if state['config']!=config:raise ValueError('Resume configuration mismatch')
            model.load_state_dict(state['model'],strict=True);optimizer.load_state_dict(state['optimizer']);restore_rng(state['rng'])
            say(folder,f'RESUME_FROM_COMMITTED_EPOCH {state["epoch"]}; incomplete epoch will be replayed')
        else:
            if (folder/'run_info.json').exists():raise ValueError('Run metadata without checkpoint; inspect before restarting')
            init=torch.load(OUTPUT/'initial.pth',map_location='cpu',weights_only=True)
            model.load_state_dict(init['model'],strict=True)
            state={'epoch':0,'config':config,'plan_signature':plan['signature'],'history':[],
                'best_positive':None,'best_tradeoff':None,'model':cpu_tree(model.state_dict()),
                'optimizer':cpu_tree(optimizer.state_dict()),'rng':rng_state()}
            save_checkpoint(path,state)
        write_best(folder,state)
        csv_write(folder/'history.csv',state['history'])
        committed=state['epoch']
        try:
            for epoch in range(state['epoch']+1,RECIPE['epochs']+1):
                started=time.perf_counter();model.train();torch.cuda.reset_peak_memory_stats()
                lr=RECIPE['min_lr']+(RECIPE['lr']-RECIPE['min_lr'])*.5*(1+math.cos(math.pi*(epoch-1)/(RECIPE['epochs']-1)))
                for group in optimizer.param_groups:group['lr']=lr
                indices=draw_indices(records['train'],arm,epoch)
                ds=DrawDataset(records['train'],indices,epoch)
                loader=DataLoader(ds,batch_size=2,shuffle=False,num_workers=workers,pin_memory=True,
                    generator=torch.Generator().manual_seed(42+epoch))
                write(folder/'run_info.json',{'status':'TRAINING','pid':os.getpid(),'completed_epochs':committed,'epoch':epoch,'config':config})
                total=0.;seen=updates=0;optimizer.zero_grad(set_to_none=True)
                say(folder,f'EPOCH {epoch}/100 arm={arm} lr={lr:.8g}; ImageNet initialization; FP32')
                for step,(x,y) in enumerate(loader,1):
                    x,y=x.to(device,non_blocking=True),y.to(device,non_blocking=True)
                    group_start=((step-1)//4)*8;group_images=min(8,len(indices)-group_start)
                    loss=mixed_loss(model(x),y)
                    if not torch.isfinite(loss):raise ValueError('Nonfinite loss; preserve last committed checkpoint')
                    (loss*len(x)/group_images).backward();total+=float(loss.detach())*len(x);seen+=len(x)
                    if step%4==0 or step==len(loader):
                        optimizer.step();optimizer.zero_grad(set_to_none=True);updates+=1
                    if step%40==0 or step==len(loader):
                        say(folder,f'  train {seen}/1160 loss={total/seen:.5f} updates={updates}/145')
                summary,rows=evaluate(model,'e13',records['validation'],device,mask_size=256)
                reasons=protection(summary,reference)
                improved_pos=state['best_positive'] is None or summary['positive']['dice']>state['best_positive']['summary']['positive']['dice']
                improved_tradeoff=not reasons and (state['best_tradeoff'] is None or tradeoff_rank(summary)>tradeoff_rank(state['best_tradeoff']['summary']))
                if improved_pos or improved_tradeoff:
                    best={'model':cpu_tree(model.state_dict()),'epoch':epoch,'summary':summary,'rows':rows}
                    if improved_pos:state['best_positive']=best
                    if improved_tradeoff:state['best_tradeoff']=best
                neg=sum(not records['train'][int(i)]['is_positive'] for i in indices)
                row={'epoch':epoch,'lr':lr,'train_loss':total/seen,'positive_draws':seen-neg,'negative_draws':neg,
                    'actual_updates':updates,'val_positive_dice':summary['positive']['dice'],'val_positive_iou':summary['positive']['iou'],
                    'val_positive_macro_dice':summary['positive_source_macro_dice'],'val_low_dice':summary['positive']['low_dice'],
                    'val_near_zero_dice':summary['positive']['near_zero_dice'],'val_zero_dice':summary['positive']['zero_dice'],
                    'val_negative_frame_fpr':summary['negative_all']['fpr_any'],
                    'val_negative_seq_macro_fpr':summary['negative_sequence_macro_fpr_any'],
                    'val_negative_seq_macro_area':summary['negative_sequence_macro_fp_area'],
                    'eligible_tradeoff':not reasons,'rejection_reasons':';'.join(reasons),
                    'best_positive_epoch':state['best_positive']['epoch'],
                    'best_tradeoff_epoch':state['best_tradeoff']['epoch'] if state['best_tradeoff'] else None,
                    'draw_indices_sha256':hashlib.sha256(indices.astype('<i8').tobytes()).hexdigest(),
                    'peak_gpu_gib':torch.cuda.max_memory_allocated()/2**30,'seconds':time.perf_counter()-started}
                state['history'].append(row)
                state.update(epoch=epoch,model=cpu_tree(model.state_dict()),optimizer=cpu_tree(optimizer.state_dict()),
                    rng=rng_state(),last_validation=summary,last_rows=rows)
                save_checkpoint(path,state);committed=epoch
                # last.pth is the atomic canonical recovery record; all sidecars can be rebuilt.
                if improved_pos or improved_tradeoff:write_best(folder,state)
                csv_write(folder/'history.csv',state['history']);write(folder/'last_validation.json',summary)
                say(folder,'EPOCH_DONE '+json.dumps(row))
            write_best(folder,state)
            write(folder/'run_info.json',{'status':'COMPLETED','completed_epochs':state['epoch'],'config':config,
                'best_positive_epoch':state['best_positive']['epoch'],
                'best_tradeoff_epoch':state['best_tradeoff']['epoch'] if state['best_tradeoff'] else None,
                'best_positive_sha256':sha(folder/'best_positive.pth'),
                'best_tradeoff_sha256':sha(folder/'best_tradeoff.pth') if state['best_tradeoff'] else None})
            say(folder,'TRAINING_COMPLETED; C6 not read; next run select')
        except BaseException as error:
            write(folder/'interruption.json',{'error':repr(error),'last_committed_epoch':committed,
                'recovery':'Rerun exactly the same training command. Active process lock prevents duplicate jobs.'})
            raise
        finally:del model,optimizer;gc.collect();torch.cuda.empty_cache()
