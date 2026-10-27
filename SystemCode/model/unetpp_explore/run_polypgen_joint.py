"""User-specified joint-data training; wait for the already active job."""
import json,subprocess,sys,time
from pathlib import Path
from evaluate_tta import ROOT,save
out=ROOT.parents[1]/'outputs/polypgen_joint_20261026'
scripts=Path(__file__).resolve().parent
prior=ROOT.parents[1]/'outputs/unet_resnet50_photo_seed17_20261026/status.json'
status={'status':'WAITING','protocol':'2269 train,494 validation,83 C6; replacement1160 per epoch; 100 epochs'}
save(out/'status.json',status)
try:
    while json.loads(prior.read_text())['status'] not in ['COMPLETED','FAILED']:
        time.sleep(30)
    status['status']='TRAINING';save(out/'status.json',status)
    cp=ROOT.parents[1]/'outputs/unet_resnet50_photo_20261026/train/best.pt'
    with (out/'train.log').open('w') as f:
        subprocess.run([sys.executable,'-B',str(scripts/'train_polypgen_joint.py'),'--checkpoint',str(cp),'--output',str(out/'train'),'--photometric','--epochs','100','--lr','0.00001','--device','mps'],stdout=f,stderr=subprocess.STDOUT,check=True)
    status['status']='EVALUATING';save(out/'status.json',status)
    with (out/'evaluation.log').open('w') as f:
        subprocess.run([sys.executable,'-B',str(scripts/'evaluate_polypgen_joint.py'),'--checkpoint',str(out/'train/best.pt'),'--output',str(out/'evaluation'),'--device','mps'],stdout=f,stderr=subprocess.STDOUT,check=True)
    status.update(status='COMPLETED',c6=json.loads((out/'evaluation/summary.json').read_text()))
except BaseException as e:
    status.update(status='FAILED',error=repr(e));raise
finally:save(out/'status.json',status)
