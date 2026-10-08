"""Meaningful checks for fusion arithmetic, protocol rejection and an end-to-end smoke."""
import argparse
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
import run_v3 as v

class V3Tests(unittest.TestCase):
    def test_equal_probability_and_threshold_boundary(self):
        a=np.array([[.9,.2],[.4,.1]],dtype=np.float32)
        b=np.array([[.1,.8],[.2,.3]],dtype=np.float32)
        p=v.fuse(a,b)
        self.assertEqual(p.dtype,np.float32)
        np.testing.assert_allclose(p,np.array([[.5,.5],[.3,.2]],dtype=np.float32))
        gt=np.array([[1,0],[1,0]],dtype=bool)
        m=v.metrics(p,gt)
        self.assertEqual((m['tp'],m['fp'],m['fn']),(1,1,1))
        self.assertEqual(m['dice'],.5);self.assertEqual(m['iou'],1/3)
        with self.assertRaises(ValueError):v.fuse(a,b[:1])

    def test_probability_integrity_and_empty_rules(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'p.npz'
            for a in [np.array([[np.nan]],dtype=np.float32),np.array([[1.1]],dtype=np.float32),np.zeros((1,1),dtype=np.float64)]:
                np.savez(p,probability=a)
                with self.assertRaises(ValueError):v.load_prob(p,(1,1))
        self.assertEqual(v.metrics(np.zeros((2,2),dtype=np.float32),np.zeros((2,2),dtype=bool))['dice'],1)
        self.assertEqual(v.metrics(np.ones((2,2),dtype=np.float32),np.zeros((2,2),dtype=bool))['dice'],0)

    def fixture(self,root):
        e0=root/'runs/e0';e2=root/'runs/e2';ev=e2/'eval'
        e0.mkdir(parents=True);ev.mkdir(parents=True);(e2/'checkpoints').mkdir()
        cp=e2/'checkpoints/final.pth';cp.write_bytes(b'fixture checkpoint, never a real model')
        manifests=[];reference=[[],[]]
        for ds,count in v.COUNTS.items():
            for i in range(count):
                sid=ds+'/'+str(i);rgb=np.full((2,3,3),i%255,dtype=np.uint8);gt=np.array([[1,1,0],[0,0,0]],dtype=bool)
                image=Path('data')/(sid+'.png');mask=Path('gt')/(sid+'.png')
                for p in [root/image,root/mask]:p.parent.mkdir(parents=True,exist_ok=True)
                Image.fromarray(rgb).save(root/image);Image.fromarray(gt.astype(np.uint8)*255).save(root/mask)
                r=dict(sample_id=sid,dataset=ds,image=image.as_posix(),mask=mask.as_posix(),height=2,width=3,gt_foreground_pixels=2,image_sha256=v.digest(root/image),mask_sha256=v.digest(root/mask),decoded_rgb_sha256=v.hashlib.sha256(rgb.tobytes()).hexdigest())
                manifests.append(r)
                for j,d in enumerate([e0,ev]):
                    p=np.where(gt,.8 if j==0 else .6,.1 if j==0 else .2).astype(np.float32)
                    pp=Path('probabilities')/(sid+'.npz');mp=Path('masks')/(sid+'.png')
                    for f in [d/pp,d/mp]:f.parent.mkdir(parents=True,exist_ok=True)
                    np.savez_compressed(d/pp,probability=p);Image.fromarray((p>=.5).astype(np.uint8)*255).save(d/mp)
                    reference[j].append(dict(sample_id=sid,dataset=ds,height=2,width=3,**v.metrics(p,gt),probability_path=pp.as_posix(),mask_path=mp.as_posix()))
        v.write_json(e0/'manifest.json',manifests);v.write_json(e2/'test_manifest.json',manifests)
        v.write_csv(e0/'per_image.csv',reference[0]);v.write_csv(ev/'per_image.csv',reference[1])
        v.write_json(e0/'run_info.json',dict(status='COMPLETED',run_kind='FULL_E0',processed_count=798,threshold=.5,protocol={'id':'endolens-e0-probability-v1'},manifest_sha256=v.digest(e0/'manifest.json'),checkpoint_sha256='fixture-e0'))
        v.write_json(e2/'run_info.json',dict(status='TRAINED',run_kind='E2_FINAL',final_checkpoint_sha256=v.digest(cp)))
        v.write_json(ev/'run_info.json',dict(status='COMPLETED',processed_count=798,checkpoint_sha256=v.digest(cp),protocol=dict(id='endolens-e2-probability-v1',threshold=.5,per_image_minmax=False)))
        return e0,e2,ev

    def test_end_to_end_smoke_and_reject_corruption(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);e0,e2,ev=self.fixture(root)
            args=argparse.Namespace(project=str(root),e0_run=str(e0),e2_run=str(e2),e2_eval=str(ev),output='runs/smoke',smoke=True,audit_only=False)
            with contextlib.redirect_stdout(io.StringIO()):v.run(args)
            out=root/'runs/smoke'
            info=v.read_json(out/'run_info.json');self.assertEqual(info['run_kind'],'SMOKE_ONLY');self.assertEqual(info['processed_count'],5)
            summary=v.rows(out/'summary.csv');self.assertEqual(float(summary[0]['fusion_dice']),1.)
            self.assertTrue((out/'comparison.html').is_file())
            self.assertEqual(len(list((out/'probabilities').rglob('*.npz'))),5)
            with self.assertRaises(FileExistsError):v.run(args)
            # A test file replacement must stop the run, not silently change the experiment.
            Image.fromarray(np.zeros((2,3),dtype=np.uint8)).save(root/'gt/Kvasir/0.png')
            args.output='runs/bad'
            with self.assertRaises(ValueError):v.run(args)
            self.assertEqual(v.read_json(root/'runs/bad/run_info.json')['status'],'FAILED')

    def test_safe_paths_and_duplicate_ids(self):
        with self.assertRaises(ValueError):v.by_id([{'sample_id':'x'},{'sample_id':'x'}])
        root=Path.cwd().resolve()
        for p in ['../escape','C:/escape','sub\\file']:
            with self.assertRaises(ValueError):v.under(root,p)

if __name__=='__main__':unittest.main(verbosity=2)
