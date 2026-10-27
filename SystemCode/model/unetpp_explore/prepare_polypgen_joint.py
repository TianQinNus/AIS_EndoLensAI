import json,hashlib
from pathlib import Path
from PIL import Image
src=Path('/Users/tianqin-gg/Desktop/EndoLensAI/Miscellaneous/model_artifacts/research_snapshot/e8_data_seed42')
root=Path('/Users/tianqin-gg/Desktop/AIS_EndoLensAI/SystemCode/model')
pg=Path('/Users/tianqin-gg/Desktop/EndoLensAI/SystemCode/model/PolypGen2021_MultiCenterData_v3')
out=Path('outputs/polypgen_joint_20261026/splits')
records={}
for name,files in {'dev_train':['original_train','new_train'],'validation':['validation'],'test':['holdout']}.items():
 rows=[]
 for f in files:
  for r in json.loads((src/(f+'.json')).read_text()):
   for k in ['image','mask']:
    old=r[k];p=pg/old.split('PolypGen2021_MultiCenterData_v3/')[1] if 'PolypGen2021_MultiCenterData_v3/' in old else root/old
    assert hashlib.sha256(p.read_bytes()).hexdigest()==r[k+'_sha256'],str(p)
    r[k]=str(p)
   rows.append(r)
 records[name]=rows
 (out/(name+'.json')).write_text(json.dumps(rows,indent=2))
assert [len(records[n]) for n in ['dev_train','validation','test']]==[2269,494,83]
ids={n:{r['decoded_rgb_sha256'] for r in rs} for n,rs in records.items()}
assert not(ids['dev_train']&(ids['validation']|ids['test']) or ids['validation']&ids['test'])
from collections import Counter
report={'counts':{n:dict(Counter(r['dataset'] for r in rs)) for n,rs in records.items()},'source':str(src),'source_hashes':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in src.glob('*.json')},'file_hashes_verified':True,'decoded_hash_sets_disjoint':True,'initialization_caveat':'Initial model previously trained on 1160 including 9 now-quarantined near duplicates; not a clean retraining claim.'}
(out/'audit.json').write_text(json.dumps(report,indent=2));print(report['counts'])
