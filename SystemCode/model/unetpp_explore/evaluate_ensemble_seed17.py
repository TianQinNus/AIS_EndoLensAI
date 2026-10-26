"""Validation-only selection of fixed ResNet34/50 U-Net logit ensembles."""
import csv
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from pretrained_model import UNet as R34
from resnet50_model import UNet as R50
from evaluate_tta import ROOT, SPLITS, sha, save, predict, score, summarize


def main():
    torch.set_num_threads(4)
    output=ROOT.parents[1]/'outputs/unet_ensemble_seed17_20261026'
    output.mkdir(parents=True,exist_ok=False)
    definitions=[('unet_pretrained384_20261025',R34),('unet_resnet50_20261025',R50),('unet_resnet50_seed17_20261026',R50)]
    models=[]; provenance=[]
    for folder,cls in definitions:
        run=ROOT.parents[1]/'outputs'/folder
        cp_path=run/'train/best.pt'
        cp=torch.load(cp_path,map_location='cpu',weights_only=True)
        mode='flip4'  # Fixed before evaluating new-seed ensemble candidates
        model=cls().to('mps');model.load_state_dict(cp['model']);model.eval()
        models.append((model,cp['config']['size'],mode))
        provenance.append(dict(checkpoint=str(cp_path),sha256=sha(cp_path),size=cp['config']['size'],mode=mode))
        for name in ['dev_train','validation']:
            assert sha(SPLITS/f'{name}.json')==cp['config'][name+'_sha256']
    records={n:json.loads((SPLITS/f'{n}.json').read_text()) for n in ['dev_train','validation','test']}
    assert len(records['test'])==798
    for rs in records.values():
        for r in rs:
            for key in ['image','mask']:assert sha(ROOT/r[key])==r[key+'_sha256']
    ids={n:{r['decoded_rgb_sha256'] for r in rs} for n,rs in records.items()}
    assert not(ids['dev_train'] & (ids['validation']|ids['test']) or ids['validation']&ids['test'])
    status=dict(status='RUNNING',models=provenance,source_sha256=sha(Path(__file__)),
                architecture_sha256={n:sha(Path(__file__).with_name(n)) for n in ['pretrained_model.py','resnet50_model.py']},
                manifests={n:sha(SPLITS/f'{n}.json') for n in records},
                candidates=[0.,.25,.5,.75,1.],weight_definition='weight on seed17; remainder equal blend of prior ResNet34 and seed42 ResNet50',
                selection='highest validation mean per-image Dice; tie prefer previous two-model ensemble',threshold=.5,
                evaluation='unchanged798 known images, native masks; not independent',torch_version=str(torch.__version__))
    save(output/'status.json',status)

    def evaluate(rs,weights,validation):
        rows={w:[] for w in weights}
        with torch.inference_mode():
            for i,r in enumerate(rs):
                with Image.open(ROOT/r['image']) as im:original=im.convert('RGB')
                with Image.open(ROOT/r['mask']) as im:raw=np.array(im.convert('L'))
                y=raw>(0 if raw.max()<=1 else 127)
                if validation:y=np.array(Image.fromarray(y).resize((256,256),Image.Resampling.NEAREST))
                logits=[]
                for model,size,mode in models:
                    im=original.resize((size,size),Image.Resampling.BILINEAR)
                    x=torch.from_numpy(np.array(im,dtype=np.float32).transpose(2,0,1).copy()/255)[None].to('mps')
                    logits.append(F.interpolate(predict(model,x,mode),y.shape,mode='bilinear',align_corners=False))
                for w in weights:
                    prob=((1-w)*(logits[0]+logits[1])/2+w*logits[2]).sigmoid()[0,0].cpu().numpy()
                    assert np.isfinite(prob).all()
                    dice,iou=score(prob,y)
                    rows[w].append(dict(sample_id=r['sample_id'],dataset=r.get('dataset','validation'),dice=dice,iou=iou))
                if (i+1)%50==0:print(f'{"validation" if validation else "known"} {i+1}/{len(rs)}',flush=True)
        return rows

    def write_rows(path,rows):
        with path.open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

    try:
        rows=evaluate(records['validation'],status['candidates'],True)
        summaries={w:summarize(v) for w,v in rows.items()}
        for w,v in rows.items():write_rows(output/f'validation_{w}.csv',v)
        save(output/'validation.json',summaries)
        selected=max(status['candidates'],key=lambda w:summaries[w]['ALL']['dice'])
        save(output/'locked_selection.json',dict(seed17_weight=selected,validation=summaries,models=provenance))
        status.update(status='EVALUATING',selected=selected);save(output/'status.json',status)
        rows=evaluate(records['test'],[selected],False)[selected]
        write_rows(output/'known_evaluation.csv',rows)
        summary=summarize(rows);save(output/'summary.json',summary)
        status.update(status='COMPLETED',known_evaluation=summary,target_reached=summary['ALL']['dice']>=.85)
        print(json.dumps(summary),flush=True)
    except BaseException as e:
        status.update(status='FAILED',error=repr(e));raise
    finally:save(output/'status.json',status)

if __name__=='__main__':main()
