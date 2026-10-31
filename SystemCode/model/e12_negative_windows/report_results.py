"""Generate auditable tables and unsmoothed training curves after common tests."""
import csv
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from common import OUTPUT, DATA, read_json, write_json, write_csv, sha, signature


def row(entry, summary, split):
    p,n=summary['positive'],summary['negative_sequence']
    return {'model':entry['name'],'split':split,'selected_on_validation':False,
        'tta':summary['protocol']['tta'],
        'epoch':entry['epoch'],'positive_count':p['count'],'positive_dice':p['dice'],
        'positive_iou':p['iou'],'positive_mae':p['mae'],'positive_low_dice':p['low_dice'],
        'positive_near_zero':p['near_zero_dice'],'positive_zero':p['zero_dice'],
        'positive_source_macro_dice':summary['positive_source_macro_dice'],
        'negative_sequence_count':n['count'],'negative_sequence_frame_fpr_any':n['fpr_any'],
        'negative_sequence_macro_fpr_any':summary['negative_sequence_macro_fpr_any'],
        'negative_sequence_macro_fp_area':summary['negative_sequence_macro_fp_area'],
        'negative_sequence_frame_fpr_area_ge_0_001':n['fpr_area_ge_0_001'],
        'negative_single_count':summary['negative_single']['count'],
        'negative_single_fpr_any':summary['negative_single']['fpr_any'],
        'negative_all_count':summary['negative_all']['count'],
        'negative_all_fpr_any':summary['negative_all']['fpr_any'],
        'negative_all_mean_fp_area':summary['negative_all']['mean_fp_area_fraction'],
        'mixed_dice_supplement_only':summary['mixed_dice_supplement_only']}


def curves(folder):
    font=Path('C:/Windows/Fonts/arial.ttf')
    bold=Path('C:/Windows/Fonts/arialbd.ttf')
    colors=['#246499','#B57920','#28786B','#8A4D95']
    runs=[]
    for key in ['r34','r50']:
        for arm in ['positive','mixed']:
            path=OUTPUT/'training'/f'{key}_{arm}_seed42'/'history.csv'
            with path.open(encoding='utf-8',newline='') as f:
                runs.append((f'{key} {arm}',list(csv.DictReader(f))))
    for metric,title in [('positive_macro_dice','Validation positive source macro Dice'),
                         ('negative_seq_macro_fpr_any','Validation negative sequence macro any-foreground FPR'),
                         ('negative_seq_macro_fp_area','Validation negative sequence macro foreground-area fraction')]:
        im=Image.new('RGB',(1500,780),'white');d=ImageDraw.Draw(im)
        ft=ImageFont.truetype(str(font),25);fb=ImageFont.truetype(str(bold),30)
        d.text((90,20),title,font=fb,fill='#1F2933')
        left,right,top,bottom=100,1440,150,660
        for j in range(6):
            y=bottom-(bottom-top)*j/5
            d.line((left,y,right,y),fill='#DDE3E9',width=2)
            d.text((20,y-15),f'{j/5:.1f}',font=ft,fill='black')
        for j in [0,5,10,15,20,25,30]:
            x=left+(right-left)*j/30
            d.text((x-10,bottom+15),str(j),font=ft,fill='black')
        for i,(name,rows) in enumerate(runs):
            x=90+i*350
            d.line((x,105,x+35,105),fill=colors[i],width=5)
            d.text((x+45,88),name,font=ft,fill='black')
            points=[(left+(right-left)*int(r['round'])/30,
                     bottom-(bottom-top)*float(r[metric])) for r in rows]
            if len(points)>1:d.line(points,fill=colors[i],width=4)
        d.line((left,top,left,bottom,right,bottom),fill='black',width=2)
        d.text((left,bottom+65),'Recorded round; original observations, no smoothing',font=ft,fill='black')
        im.save(folder/(metric+'.png'))


def report(plan):
    decision=read_json(OUTPUT/'selection.json')
    if decision['plan_signature']!=plan['signature'] or signature({k:v for k,v in decision.items() if k!='signature'})!=decision['signature']:
        raise ValueError('Invalid locked selection')
    folder=OUTPUT/'report';folder.mkdir(parents=True,exist_ok=True)
    table=[];groups=[]
    for entry in decision['entries']:
        table.append(row(entry,entry['summary'],'validation'))
        p=OUTPUT/'test_identity'/entry['name']
        meta=read_json(p/'run_info.json')
        if meta.get('status')!='COMPLETED' or meta['selection_signature']!=decision['signature']:
            raise ValueError('Run all seven common tests first')
        for name in ['summary.json','per_image.csv','per_image.json']:
            if sha(p/name)!=meta['outputs'][name]:raise ValueError('Test output changed: '+name)
        s=read_json(p/'summary.json')
        table.append(row(entry,s,'C6_historical_test'))
        for name,metrics in s['negative_groups'].items():
            groups.append({'model':entry['name'],'group':name,**metrics})
    secondary=OUTPUT/'test_rotate4_secondary'
    secondary_complete=all((secondary/e['name']/'run_info.json').exists() for e in decision['entries'])
    if secondary_complete:
        for entry in decision['entries']:
            p=secondary/entry['name'];meta=read_json(p/'run_info.json')
            if meta.get('status')!='COMPLETED' or meta['selection_signature']!=decision['signature']:
                raise ValueError('Invalid secondary TTA comparison')
            for name in ['summary.json','per_image.csv','per_image.json']:
                if sha(p/name)!=meta['outputs'][name]:raise ValueError('Secondary output changed')
            table.append(row(entry,read_json(p/'summary.json'),'C6_secondary_rotate4'))
    for r in table:r['selected_on_validation']=r['model']==decision['selected_name']
    write_csv(folder/'comparison.csv',table)
    write_csv(folder/'test_negative_groups.csv',groups)
    write_json(folder/'locked_decision.json',decision)
    curves(folder)
    lines=['# E12 positive/negative comparison','',
        'This is an exploratory comparison of delivered model pipelines and matched continuations. '
        'The previously inspected C6 data are not an independent confirmation study.','',
        f'Validation-selected entry: **{decision["selected_name"]}**. Test scores do not change that choice.','',
        'All primary entries use RGB 352 x 352, FP32 inference without TTA, native-mask logit '
        'restoration before sigmoid, threshold 0.5, and no min-max or component filtering.','',
        '## Frozen data roles','',
        '| Split | Positive | Negative | Negative sequence | Negative single |',
        '|---|---:|---:|---:|---:|']
    for name,c in plan['counts'].items():
        lines.append(f'| {name} | {c["positive"]} | {c["negative"]} | {c["negative_sequence"]} | {c["negative_single"]} |')
    for split in ['validation','C6_historical_test']+(['C6_secondary_rotate4'] if secondary_complete else []):
        lines+=['',f'## {split}','',
            '| Model | Positive Dice | IoU | Low / zero | Negative sequence any-FPR (group mean) | Negative sequence FP area (group mean) |',
            '|---|---:|---:|---:|---:|---:|']
        for r in table:
            if r['split']==split:
                mark=' **selected**' if r['selected_on_validation'] else ''
                lines.append(f'| {r["model"]}{mark} | {r["positive_dice"]:.6f} | {r["positive_iou"]:.6f} | '
                    f'{r["positive_low_dice"]} / {r["positive_zero"]} | '
                    f'{r["negative_sequence_macro_fpr_any"]:.4%} | {r["negative_sequence_macro_fp_area"]:.6%} |')
    lines+=['','## Did negative training add benefit beyond continued positive training?','']
    for key,effect in decision['negative_training_effects'].items():
        lines.append(f'- {key}: declared validation gain gate = **{effect["declared_negative_training_gain_pass"]}**; '
            f'any-FPR delta vs frozen {effect["fpr_any_delta_vs_frozen"]:+.4f}; '
            f'vs positive control {effect["fpr_any_delta_vs_positive_control"]:+.4f}; '
            f'positive macro Dice delta vs control {effect["positive_macro_delta_vs_positive_control"]:+.6f}.')
    lines+=['','## Interpretation and handover','',
        '- Lower negative any-FPR means fewer negative frames produce nonempty masks at the unchanged 0.5 threshold. '
        'Area reduction alone is not a reduction in the number of candidate frames.',
        '- The 0.001 and 0.01 foreground-area event metrics in comparison.csv are descriptive, fixed thresholds; '
        'they do not delete predicted pixels or silently change candidate filtering.',
        '- Positive segmentation and negative false foreground remain separate. Correct empty masks can inflate mixed Dice; '
        'mixed Dice is supplementary and never selected a checkpoint.',
        '- Positive source protection and exact checkpoint selection rules are in plan.json. '
        'A best epoch of zero means no continuation checkpoint beat the eligible initial model.',
        '- If the declared negative-training gain gate fails, report failure and retain the prior release pending review; '
        'do not force a new trained model into the web application.',
        '- R50 was selected after historical C6 inspection and inherited nine later-excluded near duplicates. '
        'This comparison does not erase that exposure or establish a pure encoder-architecture effect.',
        '- Preparation revision v1.1 uses the strongest frozen U-Net as the final positive-protection anchor. '
        'PVT remains a comparison entry, but its original290 scores are not a clean unseen-validation requirement for all U-Nets. '
        'Initial validation results were viewed before this revision; old protocol/metadata are preserved. No new C6 inference or formal continuation preceded the revision.',
        '- The archived R50 Dice 0.883889 used 320 plus rotate4. It is not the same protocol as this primary 352 identity table.',
        '- New genuinely independent videos with both positive and negative intervals are still needed before claiming '
        'unseen-video performance. No frame-score is a calibrated clinical probability.','',
        '## Training trajectories','',
        '![Positive Dice](positive_macro_dice.png)','',
        '![Negative any-FPR](negative_seq_macro_fpr_any.png)','',
        '![Negative foreground area](negative_seq_macro_fp_area.png)','',
        '## Selected artifact','',f'Checkpoint: `{decision["selected_checkpoint"]}`','',
        f'SHA256: `{decision["selected_weight_sha256"]}`','',
        'Use run_e12.py infer to exercise the same input/output contract. Changing a web threshold or applying '
        'postprocessing produces a different operating point and requires validation.']
    (folder/'EXPERIMENT_SUMMARY.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('REPORT_CREATED',folder/'EXPERIMENT_SUMMARY.md',flush=True)
