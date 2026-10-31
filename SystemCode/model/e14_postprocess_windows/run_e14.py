"""E14 test-set postprocessing exploration of the locked E13 epoch94 model."""
from pathlib import Path
import argparse, csv, gc, hashlib, json, os, sys, time
from collections import defaultdict
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'e13_joint_windows'))
import experiment as e13
from postprocess import grid,components,retained,metrics,confirm_sequence,cv2
OUT=ROOT/'runs/e14_postprocess_seed42'
BASE=ROOT/'runs/e13_joint_seed42/test/mixed50/rotate4/e13_mixed50_best_positive'
READ=e13.read;WRITE=e13.write;SHA=e13.sha;CSV=e13.csv_write
BASE_ID='t0.50_a0.0000_c0.00'

def code_hashes():
    return {str(p.relative_to(ROOT)).replace('\\','/'):SHA(p) for p in sorted(Path(__file__).parent.glob('*.py'))}

def context():
    plan,records=e13.verify_plan()
    selection=READ(e13.OUTPUT/'training/mixed50/selection.json')
    entry=next(r for r in selection['entries'] if r['name']=='e13_mixed50_best_positive')
    if SHA(ROOT/entry['checkpoint'])!=entry['sha256'] or entry['epoch']!=94:raise ValueError('E13 epoch94 checkpoint changed')
    meta=READ(BASE/'run_info.json')
    if meta['status']!='COMPLETED' or meta['weight_sha256']!=entry['sha256']:raise ValueError('E13 comparison mismatch')
    for name,h in meta['outputs'].items():
        if SHA(BASE/name)!=h:raise ValueError('E13 comparison output changed')
    return plan,selection,entry,records['test']

def lock_plan():
    source,selection,entry,records=context()
    spec={'stage':'E14_TEST_SET_POSTPROCESSING_EXPLORATION','e13_plan_signature':source['signature'],
        'e13_selection_signature':selection['signature'],'weight_sha256':entry['sha256'],'checkpoint':entry['checkpoint'],
        'test_manifest_sha256':SHA(e13.DATA/'test.json'),'prior_baseline_summary_sha256':SHA(BASE/'summary.json'),
        'protocol':'E13 input320 / rotate4 averaged logits / FP32 / native logit resize / sigmoid once',
        'configs':grid(),'connectivity':8,'component_confidence':'Mean original probability inside thresholded connected component; uncalibrated score',
        'area_denominator':'Entire native image area; retain components >= cutoff; no largest-only rule',
        'temporal':{'confirmation_lengths':[2,3],'bbox_iou':.1,'matching':'causal greedy one-to-one adjacent-frame boxes',
            'ordering':'ascending numeric last filename token; reset at every missing ID and sequence boundary',
            'coverage':'10 negative-only C6 sequences; positive sequence sensitivity and time latency unavailable; no temporal Dice claim'},
        'report_operating_points':{'dice_drop_budgets':[.005,.01,.02],'zero_count_increase':1,
            'choice':'Among each descriptive positive-Dice budget, minimize negative frame any-FPR, then FP area, then maximize positive Dice'},
        'inference_cache':'FP32 320x320 mean logits; exact native GPU resize/sigmoid; baseline TP/FP/FN checked against every saved E13 row',
        'scope':'User requested test-set tuning. All scores and operating points are exploratory; no held-out confirmation or deployment-optimal claim.',
        'counts':{'positive':83,'negative':1760,'negative_sequence':1755,'negative_single':5},
        'code_hashes':code_hashes(),'opencv':cv2.__version__,'torch':str(torch.__version__)}
    if (OUT/'plan.json').exists():
        old=READ(OUT/'plan.json')
        if {k:v for k,v in old.items() if k not in ['signature','created_utc']}!=spec:raise ValueError('E14 plan/source changed; preserve this run and use a new revision')
        if e13.signature({k:v for k,v in old.items() if k!='signature'})!=old['signature']:raise ValueError('E14 signature mismatch')
        return old,entry,records
    spec['created_utc']=e13.legacy.utc();spec['signature']=e13.signature(spec);WRITE(OUT/'plan.json',spec)
    return spec,entry,records

def make_cache():
    plan,entry,records=lock_plan();device=e13.setup();expected={r['sample_id']:r for r in READ(BASE/'per_image.json')}
    with e13.exclusive_lock(ROOT/'runs/e12_negative_comparison_seed42/gpu.lock'),e13.exclusive_lock(OUT/'process.lock'):
        model=None;started=time.perf_counter();done=0;new=0
        for i,r in enumerate(records):
            folder=OUT/'cache';path=folder/f'{i:04d}.json';logitpath=folder/f'{i:04d}.npy'
            if path.exists():
                cached=READ(path)
                if cached['plan_signature']!=plan['signature'] or cached['sample_id']!=r['sample_id'] or SHA(logitpath)!=cached['logit_sha256']:raise ValueError('Cache mismatch')
                done+=1;continue
            if model is None:
                model=e13.ARCH.UNet().to(device)
                model.load_state_dict(torch.load(ROOT/entry['checkpoint'],map_location='cpu',weights_only=True)['model'],strict=True);model.eval()
            for kind in ['image','mask']:
                if SHA(e13.safe_path(r[kind]))!=r[kind+'_sha256']:raise ValueError('Input changed '+r['sample_id'])
            with Image.open(e13.safe_path(r['mask'])) as im:a=np.asarray(im.convert('L'));gt=a>(0 if a.max()<=1 else 127)
            if gt.shape!=(r['height'],r['width']) or bool(gt.any())!=r['is_positive']:raise ValueError('GT changed')
            with Image.open(e13.safe_path(r['image'])) as im:x=e13.input_tensor(im).unsqueeze(0).to(device)
            with torch.inference_mode():
                logits=sum(torch.rot90(model(torch.rot90(x,k,(-2,-1))),-k,(-2,-1)) for k in range(4))/4
                probability=F.interpolate(logits.float(),size=gt.shape,mode='bilinear',align_corners=False).sigmoid()[0,0].cpu().numpy()
            region_sets={f'{t:.2f}':components(probability,gt,t) for t in [.5,.6,.7,.8,.9]}
            baseline=metrics(region_sets['0.50'],int(gt.sum()),gt.size);old=expected[r['sample_id']]
            for key in ['tp','fp','fn','pred_pixels']:
                if baseline[key]!=old[key]:raise ValueError('Native FP32 baseline failed exact reproduction: '+r['sample_id']+' '+key)
            folder.mkdir(parents=True,exist_ok=True)
            temp=logitpath.with_suffix('.tmp')
            with temp.open('wb') as f:np.save(f,logits.cpu().numpy().astype(np.float32),allow_pickle=False)
            os.replace(temp,logitpath)
            WRITE(path,{'plan_signature':plan['signature'],'sample_id':r['sample_id'],'height':gt.shape[0],'width':gt.shape[1],
                'gt_pixels':int(gt.sum()),'logit_sha256':SHA(logitpath),'regions':region_sets,'baseline_exact':True})
            new+=1;done+=1
            if done%25==0 or done==len(records):
                print(f'[E14 cache] {done}/{len(records)} newly_computed={new} elapsed={time.perf_counter()-started:.1f}s baseline_exact=PASS',flush=True)
        if model is not None:del model;gc.collect();torch.cuda.empty_cache()
        WRITE(OUT/'cache_status.json',{'status':'COMPLETED','count':done,'plan_signature':plan['signature'],
            'exact_baseline_match_count':done,'seconds_this_invocation':time.perf_counter()-started})

def summarize(records,caches,config):
    rows=[]
    for r,c in zip(records,caches):
        kept=retained(c['regions'][f"{config['pixel_threshold']:.2f}"],config,c['height']*c['width'])
        m=metrics(kept,c['gt_pixels'],c['height']*c['width'])
        rows.append({'sample_id':r['sample_id'],'group_id':r['group_id'],'dataset':r['dataset'],
            'is_positive':r['is_positive'],'label_kind':r['label_kind'],**m})
    def mean(a,key):return float(np.mean([r[key] for r in a])) if a else None
    pos=[r for r in rows if r['is_positive']];neg=[r for r in rows if not r['is_positive']]
    groups=sorted({r['group_id'] for r in neg if r['label_kind']=='author_negative_only_sequence'})
    groupsumm={g:{'count':sum(r['group_id']==g for r in neg),'fpr':mean([r for r in neg if r['group_id']==g],'any_foreground'),
        'fp_area':mean([r for r in neg if r['group_id']==g],'pred_area_fraction')} for g in groups}
    small_ids={r['sample_id'] for r,c in zip(records,caches) if r['is_positive'] and c['gt_pixels']/(c['height']*c['width'])<.01}
    small=[r for r in pos if r['sample_id'] in small_ids]
    active=sum(config[k]>0 for k in ['min_component_area_fraction','min_component_mean_probability'])+int(config['pixel_threshold']>.5)
    family='baseline' if active==0 else 'combined' if active>1 else 'pixel_threshold' if config['pixel_threshold']>.5 else 'component_area' if config['min_component_area_fraction']>0 else 'component_confidence'
    summary={**config,'family':family,'positive_count':len(pos),'positive_dice':mean(pos,'dice'),'positive_iou':mean(pos,'iou'),
        'positive_low':sum(r['dice']<.5 for r in pos),'positive_near_zero':sum(r['dice']<.1 for r in pos),
        'positive_zero':sum(r['dice']==0 for r in pos),'positive_empty_predictions':sum(r['empty_prediction'] for r in pos),
        'positive_overlap_hit_rate':mean(pos,'overlap_hit'),'small_gt_count':len(small),'small_gt_dice':mean(small,'dice'),
        'negative_count':len(neg),'negative_fp_frames':sum(r['any_foreground'] for r in neg),
        'negative_frame_fpr':mean(neg,'any_foreground'),'negative_mean_fp_area':mean(neg,'pred_area_fraction'),
        'negative_seq_macro_fpr':float(np.mean([s['fpr'] for s in groupsumm.values()])),
        'negative_seq_macro_fp_area':float(np.mean([s['fp_area'] for s in groupsumm.values()])),
        'negative_area_ge_0_001_fpr':mean(neg,'area_ge_0_001'),'negative_area_ge_0_01_fpr':mean(neg,'area_ge_0_01')}
    return summary,rows,groupsumm

def operating_points(summaries):
    baseline=next(s for s in summaries if s['id']==BASE_ID);points={}
    for budget in [.005,.01,.02]:
        allowed=[s for s in summaries if s['positive_dice']>=baseline['positive_dice']-budget and s['positive_zero']<=baseline['positive_zero']+1]
        best=min(allowed,key=lambda s:(s['negative_frame_fpr'],s['negative_mean_fp_area'],-s['positive_dice'],s['id']))
        points[str(budget)]=best['id']
    return points

def temporal(records,caches,configs,selected_ids):
    rows=[];per_frames=[];group_rows=[];audit={}
    for ident in selected_ids:
        config=next(c for c in configs if c['id']==ident);groups=defaultdict(list)
        for r,c in zip(records,caches):
            if r['label_kind']!='author_negative_only_sequence':continue
            regions=retained(c['regions'][f"{config['pixel_threshold']:.2f}"],config,c['height']*c['width'])
            groups[r['group_id']].append({'sample_id':r['sample_id'],'frame_number':int(Path(r['image']).stem.split('_')[-1]),
                'regions':regions,'image_area':c['height']*c['width']})
        for k in [1,2,3]:
            metrics_by_group=[];allframes=[]
            for group,frames in sorted(groups.items()):
                frames.sort(key=lambda f:f['frame_number'])
                if len({f['frame_number'] for f in frames})!=len(frames):raise ValueError('Duplicate numeric frame ID')
                area={f['sample_id']:f['image_area'] for f in frames}
                confirmed=confirm_sequence(frames,k=k,min_iou=.1)
                values=[]
                for f in confirmed:
                    m=metrics(f['regions'],0,area[f['sample_id']])
                    row={'spatial_config':ident,'confirm_frames':k,'group':group,'sample_id':f['sample_id'],
                        'frame_number':f['frame_number'],'warmup':f['warmup'],'any_foreground':m['any_foreground'],
                        'pred_area_fraction':m['pred_area_fraction']}
                    values.append(row);allframes.append(row);per_frames.append(row)
                score={'spatial_config':ident,'confirm_frames':k,'sequence':group,'count':len(values),
                    'fp_frames':sum(f['any_foreground'] for f in values),'fpr':float(np.mean([f['any_foreground'] for f in values])),
                    'mean_fp_area':float(np.mean([f['pred_area_fraction'] for f in values]))}
                group_rows.append(score);metrics_by_group.append(score)
                audit[group]={'count':len(frames),'first':frames[0]['frame_number'],'last':frames[-1]['frame_number'],
                    'missing_id_boundaries':sum(b['frame_number']!=a['frame_number']+1 for a,b in zip(frames,frames[1:]))}
            after=[f for f in allframes if not f['warmup']]
            rows.append({'spatial_config':ident,'confirm_frames':k,'negative_sequence_frames':len(allframes),
                'negative_sequence_fp_frames':sum(f['any_foreground'] for f in allframes),
                'negative_sequence_frame_fpr':float(np.mean([f['any_foreground'] for f in allframes])),
                'negative_sequence_macro_fpr':float(np.mean([s['fpr'] for s in metrics_by_group])),
                'negative_mean_fp_area':float(np.mean([f['pred_area_fraction'] for f in allframes])),
                'after_warmup_frames':len(after),'after_warmup_fpr':float(np.mean([f['any_foreground'] for f in after])),
                'positive_dice':'NOT_EVALUABLE','positive_sensitivity':'NOT_EVALUABLE','latency_seconds':'UNKNOWN_FPS',
                'minimum_confirmation_delay_observed_frames':k-1})
    CSV(OUT/'temporal_comparison.csv',rows);CSV(OUT/'temporal_per_sequence.csv',group_rows)
    CSV(OUT/'temporal_per_frame.csv',per_frames);WRITE(OUT/'sequence_order_audit.json',audit)
    return rows

def analyze():
    plan,_,records=lock_plan();status=READ(OUT/'cache_status.json')
    if status['status']!='COMPLETED' or status['count']!=len(records):raise ValueError('Complete cache inference first')
    caches=[READ(OUT/'cache'/f'{i:04d}.json') for i in range(len(records))]
    if any(c['plan_signature']!=plan['signature'] or c['sample_id']!=r['sample_id'] for r,c in zip(records,caches)):raise ValueError('Mixed cache identity')
    configs=grid();summaries=[]
    for config in configs:
        s,_,_=summarize(records,caches,config);summaries.append(s)
    baseline=next(s for s in summaries if s['id']==BASE_ID);prior=READ(BASE/'summary.json')
    for key,target in [('positive_dice',prior['positive']['dice']),('positive_iou',prior['positive']['iou']),('negative_frame_fpr',prior['negative_all']['fpr_any'])]:
        if not np.isclose(baseline[key],target,rtol=0,atol=1e-12):raise ValueError('Aggregate baseline mismatch '+key)
    for s in summaries:
        s['dice_delta_vs_baseline']=s['positive_dice']-baseline['positive_dice']
        s['iou_delta_vs_baseline']=s['positive_iou']-baseline['positive_iou']
        s['fpr_delta_vs_baseline']=s['negative_frame_fpr']-baseline['negative_frame_fpr']
        s['exploratory_pareto']=not any(t['positive_dice']>=s['positive_dice'] and t['negative_frame_fpr']<=s['negative_frame_fpr'] and
            (t['positive_dice']>s['positive_dice'] or t['negative_frame_fpr']<s['negative_frame_fpr']) for t in summaries)
    CSV(OUT/'comparison.csv',summaries);WRITE(OUT/'comparison.json',summaries)
    points=operating_points(summaries)
    # Temporal checks use baseline and the three declared spatial operating points.
    chosen_ids=list(dict.fromkeys([BASE_ID]+list(points.values())))
    for ident in chosen_ids:
        config=next(c for c in configs if c['id']==ident);s,per,groups=summarize(records,caches,config)
        CSV(OUT/'operating_points'/ident/'per_image.csv',per);WRITE(OUT/'operating_points'/ident/'summary.json',s)
        WRITE(OUT/'operating_points'/ident/'negative_groups.json',groups)
    time_rows=temporal(records,caches,configs,chosen_ids)
    WRITE(OUT/'exploratory_operating_points.json',{'status':'TEST_SET_EXPLORATION_NOT_VALIDATED_SELECTION',
        'plan_signature':plan['signature'],'baseline':BASE_ID,'dice_drop_budget_to_config':points,
        'warning':'Test-informed choices require independent or development-only confirmation before release.'})
    write_report(summaries,points,time_rows)
    print(json.dumps({'baseline':baseline,'operating_points':[next(s for s in summaries if s['id']==i) for i in dict.fromkeys(points.values())],
        'temporal':time_rows},indent=2))

def write_report(summaries,points,temporal_rows):
    baseline=next(s for s in summaries if s['id']==BASE_ID)
    lines=['# E14: exploratory postprocessing of E13 epoch94','',
        '**Scope:** User-requested C6 test-set exploration. This is not independent evaluation or validation-only model selection. No model retraining.',
        '', 'Input320, rotate4 logit averaging, native-size resize, FP32. All 1,843 unfiltered predictions exactly reproduce the saved E13 TP/FP/FN/predicted-pixel counts.',
        '', '## Spatial single-factor comparisons','',
        '| Method | Pixel threshold | Min region area (%) | Min mean score | Positive Dice | Positive IoU | Zero Dice | Negative any-FPR | Mean negative FP area (%) |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    singles=[s for s in summaries if s['family']!='combined']
    def row(s):return f"| {s['family']} | {s['pixel_threshold']:.2f} | {s['min_component_area_fraction']*100:.3f} | {s['min_component_mean_probability']:.2f} | {s['positive_dice']:.6f} | {s['positive_iou']:.6f} | {s['positive_zero']} | {s['negative_frame_fpr']:.2%} | {s['negative_mean_fp_area']*100:.4f} |"
    lines += [row(s) for s in singles]
    lines += ['', '## Descriptive combined operating points','',
        'Choose minimum negative frame FPR within the declared Dice-drop budgets (0.005/0.01/0.02) and at most one extra zero-Dice image. This is a test-informed tradeoff illustration, not a deployable validated optimum.','',
        '| Allowed Dice drop | Configuration | Dice | IoU | Negative any-FPR | Zero Dice |', '|---:|---|---:|---:|---:|---:|']
    for budget,ident in points.items():
        s=next(s for s in summaries if s['id']==ident)
        lines.append(f"| {budget} | {ident} | {s['positive_dice']:.6f} | {s['positive_iou']:.6f} | {s['negative_frame_fpr']:.2%} | {s['positive_zero']} |")
    lines += ['', '## Causal temporal confirmation: negative-only feasibility check','',
        'Numeric filename ordering, missing-ID resets, one-to-one previous/current bounding-box IoU >=0.1. Retain only current regions observed in 2 or 3 consecutive frames. No lookahead or backfill.','',
        '| Spatial config | Consecutive frames | Negative sequence frame FPR | Sequence-macro FPR | FPR after warmup |', '|---|---:|---:|---:|---:|']
    for r in temporal_rows:
        lines.append(f"| {r['spatial_config']} | {r['confirm_frames']} | {r['negative_sequence_frame_fpr']:.2%} | {r['negative_sequence_macro_fpr']:.2%} | {r['after_warmup_fpr']:.2%} |")
    lines += ['', 'The 1,755 negative sequence frames come from 10 sequences. The 83 positive examples are single images, not positive video sequences. Therefore temporal positive Dice, sensitivity and real-time delay in seconds cannot be estimated here. Do not combine unaffected static-positive Dice with temporal-negative FPR as if both were evaluated in a common video pipeline.',
        '', '## Interpretation and files','',
        '- Any-FPR counts a verified negative image with any remaining foreground pixel. FP-area fraction is a separate endpoint.',
        '- Component area uses the entire native image area. It is not a largest-component-only rule. 8-connectivity is used.',
        '- Confidence is the component mean model score after pixel thresholding; it is not a calibrated probability of a real polyp.',
        '- Higher thresholds and filters only delete foreground. Positive Dice can rise if removed pixels were false foreground, or fall if genuine lesion pixels are removed.',
        '- `comparison.csv/json` contains all 150 spatial settings, including small-GT (<1% image area) scores and false-positive counts.',
        '- `operating_points/*/per_image.csv` contains baseline and selected illustrative settings. `temporal_per_frame.csv` preserves causal frame decisions.',
        '- `sequence_order_audit.json` records numeric frame ranges and gaps. Temporal tracking resets at every sequence boundary and missing numeric ID.',
        '- `cache/*.npy` preserves FP32 mean logits; `cache/*.json` preserves native connected-component sufficient statistics. Cached records are tied to the signed plan.',
        '- Confirm any preferred rule on development data and new patient/video-grouped positive and negative clips before web-model replacement.',
        '', 'No E13 training source, selection record, model weight or web deployment has been modified.']
    (OUT/'EXPERIMENT_SUMMARY.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['cache','analyze','all'])
    args=p.parse_args()
    if args.command in ['cache','all']:make_cache()
    if args.command in ['analyze','all']:analyze()

if __name__=='__main__':main()
