"""E4-specific CPU checks; no formal CPU training is enabled."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import run_e4 as e4
from data import TrainingDataset

class E4Tests(unittest.TestCase):
    def test_only_augmentation_changes(self):
        base=dict(e4.config(),augmentation='flips')
        e4.assert_control(base,e4.config())
        bad=e4.config();bad['lr']=2e-4
        with self.assertRaises(ValueError):e4.assert_control(base,bad)
        badbase=dict(base,epochs=80)
        with self.assertRaises(ValueError):e4.assert_control(badbase,e4.config())

    def test_strong_geometry_mask_and_determinism(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);mask=np.zeros((40,40),np.uint8);mask[2:23,5:18]=255;mask[20:35,24:37]=255
            Image.fromarray(np.stack([mask]*3,-1)).save(root/'x.png');Image.fromarray(mask).save(root/'m.png')
            rec={'sample_id':'fixture','image':'x.png','mask':'m.png'}
            for epoch in range(8):
                ds=TrainingDataset(root,[rec],40,42,epoch,'strong');x,y=ds[0];xx,yy=ds[0]
                self.assertTrue(torch.equal(x,xx));self.assertTrue(torch.equal(y,yy))
                rng=np.random.default_rng(np.random.SeedSequence([42,epoch,0]));expected=Image.fromarray(mask)
                for op in [Image.Transpose.FLIP_LEFT_RIGHT,Image.Transpose.FLIP_TOP_BOTTOM]:
                    if rng.random()<.5:expected=expected.transpose(op)
                expected=expected.rotate(float(rng.uniform(-15,15)),resample=Image.Resampling.NEAREST)
                np.testing.assert_array_equal(y[0].numpy(),(np.asarray(expected)>127).astype(np.float32))
                self.assertEqual(set(np.unique(y.numpy())),{0.,1.})
                self.assertTrue(torch.isfinite(x).all())

    def test_smoke_config_not_formal(self):
        s=e4.config(True);self.assertTrue(s['smoke']);self.assertEqual(s['epochs'],1)
        self.assertEqual(s['schedule_horizon'],100)
        with self.assertRaises(ValueError):e4.assert_control(dict(e4.config(),augmentation='flips'),s)

if __name__=='__main__':unittest.main(verbosity=2)
