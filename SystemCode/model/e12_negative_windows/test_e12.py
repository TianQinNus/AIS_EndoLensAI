"""Meaningful checks for empty targets, sampling, provenance and selection."""
import copy
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from common import ROOT, exclusive_lock, safe_path, canonical_path
from metrics import per_image, summarize, positive_protection, checkpoint_rank
from modeling import mixed_loss
from training import sampling_probabilities, draw_indices
from prepare_data import NearIndex, NEGATIVE_CENTERS, verified_empty_single, bbox_for_single


def record(name='image', positive=True, source='original_validation', group='positive'):
    return {'sample_id':name,'is_positive':positive,'dataset':source,
            'group_id':group,'label_kind':'positive_mask' if positive else 'author_negative_only_sequence',
            'center':'C5'}


class EmptyMaskLossTests(unittest.TestCase):
    def test_all_negative_equals_bce_and_has_finite_foreground_penalty(self):
        z=torch.zeros(2,1,4,4,requires_grad=True)
        y=torch.zeros_like(z)
        loss,bce,dice=mixed_loss(z,y)
        self.assertTrue(torch.equal(loss,bce))
        self.assertEqual(float(dice.detach()),0.)
        loss.backward()
        self.assertTrue(torch.isfinite(z.grad).all())
        self.assertTrue((z.grad>0).all())

    def test_all_positive_preserves_base_bce_dice(self):
        z=torch.full((2,1,3,3),.2)
        y=torch.ones_like(z)
        loss,bce,dice=mixed_loss(z,y)
        self.assertAlmostEqual(float(loss),float(bce+dice),places=6)
        self.assertGreater(float(dice),0)

    def test_microbatch_accumulation_independent_of_class_composition(self):
        torch.manual_seed(9)
        initial=torch.randn(4,1,4,4)
        y=torch.zeros_like(initial);y[0,:,1:3,1:3]=1;y[3,:,1:3,1:3]=1
        full=initial.clone().requires_grad_();mixed_loss(full,y)[0].backward()
        split=initial.clone().requires_grad_()
        # Includes a fully negative microbatch. Weighted accumulation must agree.
        for begin,end in [(0,1),(1,3),(3,4)]:
            (mixed_loss(split[begin:end],y[begin:end])[0]*(end-begin)/4).backward()
        torch.testing.assert_close(full.grad,split.grad,rtol=1e-6,atol=1e-7)


class SamplingTests(unittest.TestCase):
    def setUp(self):
        self.rows=[]
        for source,n in [('original_training',11),('PolypGen-C1',2),('PolypGen-C2',3),('PolypGen-C3',5),('PolypGen-C4',7)]:
            self.rows.extend({'dataset':source,'is_positive':True} for _ in range(n))
        self.rows.extend({'dataset':'negative','is_positive':False} for _ in range(13))

    def test_mixed_mass_and_conditional_source_weights(self):
        p=sampling_probabilities(self.rows,'mixed')
        self.assertAlmostEqual(sum(v for r,v in zip(self.rows,p) if not r['is_positive']),.25)
        self.assertAlmostEqual(sum(v for r,v in zip(self.rows,p) if r['dataset']=='original_training'),.375)
        for center in range(1,5):
            values=[v for r,v in zip(self.rows,p) if r['dataset']==f'PolypGen-C{center}']
            self.assertAlmostEqual(sum(values),.09375)
            self.assertEqual(len(set(values)),1)

    def test_positive_control_never_draws_negative(self):
        p=sampling_probabilities(self.rows,'positive')
        self.assertEqual(sum(v for r,v in zip(self.rows,p) if not r['is_positive']),0)
        ids=draw_indices(self.rows,'positive',42,1)
        self.assertTrue(all(self.rows[int(i)]['is_positive'] for i in ids))

    def test_draws_reproducible_with_replacement_not_quotas(self):
        a=draw_indices(self.rows,'mixed',42,1)
        np.testing.assert_array_equal(a,draw_indices(self.rows,'mixed',42,1))
        self.assertFalse(np.array_equal(a,draw_indices(self.rows,'mixed',42,2)))
        self.assertEqual(len(a),1160)
        self.assertLess(len(set(a)),len(a))


class MetricTests(unittest.TestCase):
    def test_negative_empty_and_false_foreground(self):
        r=record(positive=False,group='seq11_neg')
        gt=np.zeros((10,10),bool)
        a=per_image(np.zeros((10,10),np.float32),gt,r)
        p=np.zeros((10,10),np.float32);p[0,0]=.9
        b=per_image(p,gt,r)
        self.assertEqual(a['dice'],1)
        self.assertEqual(b['dice'],0)
        self.assertEqual(b['any_foreground'],1)
        self.assertEqual(b['pred_area_fraction'],.01)

    def test_mixed_dice_inflation_cannot_change_positive_dice(self):
        gt=np.ones((2,2),bool)
        p=per_image(np.zeros((2,2),np.float32),gt,record())
        n=per_image(np.zeros((2,2),np.float32),~gt,record('n',False,group='seq11_neg'))
        s=summarize([p]+[n]*9)
        self.assertEqual(s['positive']['dice'],0)
        self.assertAlmostEqual(s['mixed_dice_supplement_only'],.9)

    def test_sequence_equal_differs_from_frame_equal(self):
        gt=np.zeros((2,2),bool)
        yes=per_image(np.ones((2,2),np.float32),gt,record('a',False,group='seq11_neg'))
        no=per_image(np.zeros((2,2),np.float32),gt,record('b',False,group='seq12_neg'))
        s=summarize([yes]+[no]*9)
        self.assertAlmostEqual(s['negative_all']['fpr_any'],.1)
        self.assertAlmostEqual(s['negative_sequence_macro_fpr_any'],.5)

    def test_positive_sources_are_equal_not_count_weighted(self):
        gt=np.ones((2,2),bool)
        a=per_image(np.ones((2,2),np.float32),gt,record())
        b=per_image(np.zeros((2,2),np.float32),gt,record('b',source='PolypGen-C5'))
        s=summarize([a]*9+[b])
        self.assertAlmostEqual(s['positive']['dice'],.9)
        self.assertAlmostEqual(s['positive_source_macro_dice'],.5)

    def test_empty_everywhere_cannot_pass_positive_protection(self):
        ref={'positive_source_macro_dice':.8,'positive_sources':{'a':{'dice':.8}},
             'positive':{'low_dice':3,'zero_dice':1}}
        bad=copy.deepcopy(ref);bad['positive_source_macro_dice']=0;bad['positive_sources']['a']['dice']=0
        self.assertTrue(positive_protection(bad,ref))

    def test_negative_area_breaks_equal_any_fpr_tie(self):
        a={'negative_sequence_macro_fpr_any':1.,'negative_sequence_macro_fp_area':.2,'positive_source_macro_dice':.9}
        b={**a,'negative_sequence_macro_fp_area':.02,'positive_source_macro_dice':.895}
        self.assertGreater(checkpoint_rank(b),checkpoint_rank(a))


class AuditAndLockTests(unittest.TestCase):
    def test_zero_mask_not_inferred_from_missing_file(self):
        from PIL import Image
        with tempfile.TemporaryDirectory(dir=ROOT/'runs') as temp:
            root=Path(temp);image=root/'image.png';mask=root/'mask.png';bbox=root/'box.txt'
            Image.new('RGB',(8,8)).save(image);Image.new('L',(8,8),0).save(mask)
            self.assertFalse(verified_empty_single(image,mask,bbox))
            bbox.write_text('polyp 1 1 3 3\n')
            self.assertFalse(verified_empty_single(image,mask,bbox))
            bbox.write_text('\n')
            self.assertTrue(verified_empty_single(image,mask,bbox))
            Image.new('L',(8,8),255).save(mask)
            self.assertFalse(verified_empty_single(image,mask,bbox))

    def test_c2_unsuffixed_author_bbox_pairing(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'runs') as temp:
            root=Path(temp);(root/'bbox_C2').mkdir()
            expected=root/'bbox_C2/example.txt';expected.write_text('\n')
            self.assertEqual(bbox_for_single(root,2,root/'example_mask.jpg','example'),expected)

    def test_sequence_center_roles_keep_all_c6_out_of_training(self):
        self.assertTrue(all(NEGATIVE_CENTERS[i]=='C6' for i in range(14,24)))
        self.assertTrue(all(NEGATIVE_CENTERS[i]=='C5' for i in range(11,14)))
        self.assertTrue(all(NEGATIVE_CENTERS[i] in ['C1','C4'] for i in range(1,11)))

    def test_path_escape_rejected(self):
        with self.assertRaises(ValueError):safe_path('../outside')
        self.assertTrue(safe_path('runs/e12_negative_comparison_seed42').is_relative_to(ROOT))

    def test_extended_windows_path_spelling(self):
        path=ROOT/'runs'
        self.assertEqual(canonical_path(path),path.resolve())

    def test_live_lock_cannot_be_stolen(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'runs') as temp:
            path=Path(temp)/'lock.json'
            with exclusive_lock(path):
                with self.assertRaises(RuntimeError):
                    with exclusive_lock(path):pass
            self.assertFalse(path.exists())

    def test_dhash_alone_does_not_imply_duplicate(self):
        index=NearIndex()
        a={'sample_id':'a','dhash64':'0000000000000000','decoded_rgb_sha256':'a','width':10,'height':10}
        index.add(a,np.zeros((16,16,3),np.uint8))
        b={**a,'sample_id':'b','decoded_rgb_sha256':'b'}
        self.assertEqual(index.matches(b,np.full((16,16,3),255,np.uint8)),[])
        self.assertEqual(len(index.matches(b,np.ones((16,16,3),np.uint8))),1)


if __name__=='__main__':
    unittest.main(verbosity=2)
