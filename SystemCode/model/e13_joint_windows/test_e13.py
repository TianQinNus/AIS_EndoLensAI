"""Meaningful CPU checks for empty-target math, sampling and fixed data roles."""
import copy, random, unittest
import numpy as np
import torch
from PIL import Image
import experiment as e

class JointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):e.setup('cpu');cls.bundle,cls.records=e.load_data(False)

    def test_loss_matches_original_and_empty_gradients(self):
        for kind in ['positive','negative','mixed']:
            torch.manual_seed(7);z=torch.randn(2,1,16,16,requires_grad=True);y=torch.zeros_like(z)
            if kind!='negative':y[0,:,3:9,4:12]=1
            if kind=='positive':y[1,:,2:8,2:8]=1
            loss=e.mixed_loss(z,y)
            self.assertTrue(torch.equal(loss,e.HISTORICAL.loss_fn(z,y)))
            loss.backward();self.assertTrue(torch.isfinite(z.grad).all())
            if kind=='negative':self.assertGreater(float(z.grad.sum()),0)

    def test_effective_batch_accumulation(self):
        torch.manual_seed(8);net=torch.nn.Conv2d(3,1,1);clone=copy.deepcopy(net)
        x=torch.randn(8,3,12,12);y=torch.zeros(8,1,12,12);y[:4,:,3:7,3:7]=1
        e.mixed_loss(net(x),y).backward()
        for start in range(0,8,2):(e.mixed_loss(clone(x[start:start+2]),y[start:start+2])*.25).backward()
        for a,b in zip(net.parameters(),clone.parameters()):self.assertTrue(torch.allclose(a.grad,b.grad,atol=2e-7,rtol=1e-5))

    def test_sampling_and_split_boundaries(self):
        train=self.records['train'];totals={}
        for arm in ['mixed50','positive']:
            indices=np.concatenate([e.draw_indices(train,arm,k) for k in range(1,101)])
            mass=sum(not train[int(i)]['is_positive'] for i in indices)/len(indices)
            self.assertLess(abs(mass-(.5 if arm=='mixed50' else 0)),.01)
            self.assertTrue(np.array_equal(e.draw_indices(train,arm,3),e.draw_indices(train,arm,3)))
            self.assertFalse(np.array_equal(e.draw_indices(train,arm,3),e.draw_indices(train,arm,4)))
            positives=[train[int(i)]['dataset'] for i in indices if train[int(i)]['is_positive']]
            for source,expected in [('original_training',.5),('PolypGen-C1',.125),('PolypGen-C2',.125),('PolypGen-C3',.125),('PolypGen-C4',.125)]:
                self.assertLess(abs(positives.count(source)/len(positives)-expected),.01)
            totals[arm]=mass
        self.assertEqual(self.bundle['counts']['train']['negative'],1907)
        for role in ['train','validation','test']:
            centers={r['center'] for r in self.records[role]}
            if role=='train':self.assertFalse({'C5','C6'}&centers)
            if role=='validation':self.assertNotIn('C6',centers)
            if role=='test':self.assertEqual(centers,{'C6'})

    def test_real_negative_augmentation_stays_empty_and_deterministic(self):
        train=self.records['train'];negative=next(i for i,r in enumerate(train) if not r['is_positive'])
        ds=e.DrawDataset(train,np.array([negative]),7);x,y=ds[0];x2,y2=ds[0]
        self.assertEqual(tuple(x.shape),(3,320,320));self.assertEqual(tuple(y.shape),(1,320,320))
        self.assertEqual(int(y.sum()),0);self.assertTrue(torch.equal(x,x2));self.assertTrue(torch.equal(y,y2))

    def test_augmentation_exact_reference_operations(self):
        image=Image.fromarray(np.arange(32*32*3,dtype=np.uint8).reshape(32,32,3));mask=Image.new('L',(32,32),255)
        random.seed(123);a,b=e.augment_pair(image,mask)
        random.seed(123);c,d=e.HISTORICAL.augment_pair(image,mask)
        from PIL import ImageEnhance
        if random.random()<.5:c=ImageEnhance.Color(c).enhance(random.uniform(.5,1.5))
        if random.random()<.5:c=Image.fromarray(np.round((np.asarray(c)/255.)**random.uniform(.7,1.5)*255).astype(np.uint8))
        self.assertTrue(np.array_equal(a,c));self.assertTrue(np.array_equal(b,d))

    def test_initialization_pretrained_encoder_random_decoder(self):
        self.assertEqual(e.sha(e.ENCODER),e.ENCODER_SHA)
        e.seed_all();a=e.ARCH.UNet(pretrained_path=e.ENCODER)
        e.seed_all();b=e.ARCH.UNet(pretrained_path=e.ENCODER)
        self.assertTrue(all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items()))
        official=torch.load(e.ENCODER,map_location='cpu',weights_only=True)
        self.assertTrue(all(torch.equal(v,official[k]) for k,v in a.encoder.state_dict().items()))
        a.train();self.assertTrue(a.head.training)
        for m in a.encoder.modules():
            if isinstance(m,torch.nn.BatchNorm2d):self.assertFalse(m.training);self.assertTrue(m.weight.requires_grad)
        self.assertEqual(sum(p.numel() for p in a.parameters()),32520977)

    def test_tradeoff_protects_positives_and_does_not_rank_mixed_dice(self):
        ref={'positive':{'dice':.85,'low_dice':40,'zero_dice':10},'positive_sources':{'A':{'dice':.9},'B':{'dice':.8}}}
        good=copy.deepcopy(ref);good.update(negative_sequence_macro_fpr_any=.2,negative_sequence_macro_fp_area=.01)
        self.assertFalse(e.protection(good,ref))
        bad=copy.deepcopy(good);bad['positive']['zero_dice']=14;self.assertTrue(e.protection(bad,ref))
        bad=copy.deepcopy(good);bad['positive']['dice']=.82;self.assertTrue(e.protection(bad,ref))
        other=copy.deepcopy(good);other['negative_sequence_macro_fpr_any']=.3
        self.assertGreater(e.tradeoff_rank(good),e.tradeoff_rank(other))

if __name__=='__main__':unittest.main(verbosity=2)
