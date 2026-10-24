#!/usr/bin/env python3
"""CPU unit/integration tests. These are NOT formal E2 training runs."""
import hashlib
import json
import tempfile
import shutil
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image
import torch
from data import split_training, TrainingDataset, signature
from engine import require_cuda, execute_training, load_checkpoint, lr_at_epoch
from models import make_model, read_pretrained, segmentation_loss, ResNet34UNet
from e0_protocol import binary_metrics
from run_e2 import select_recipe, evaluate_final


class E2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        warnings.filterwarnings("ignore", category=FutureWarning)

    def test_split_deterministic_and_duplicate_grouping(self):
        records = [{"sample_id": f"TrainDataset/{i:03}", "stem": str(i) if i % 2 else f"abc{i}",
                    "height": 10, "width": 10, "decoded_rgb_sha256": str(i), "gt_foreground_pixels": 10} for i in range(100)]
        a, b = split_training(records)
        c, d = split_training(list(reversed(records)))
        self.assertEqual((a, b), (c, d))
        self.assertEqual((len(a), len(b)), (80, 20))
        records[1]["decoded_rgb_sha256"] = records[0]["decoded_rgb_sha256"]
        a, b = split_training(records)
        self.assertFalse({r['decoded_rgb_sha256'] for r in a} & {r['decoded_rgb_sha256'] for r in b})
        metadata = {r['sample_id']: {'source': 'test', 'group_id': str(i // 5)} for i, r in enumerate(records)}
        a, b = split_training(records, metadata=metadata)
        self.assertFalse({r['group_id'] for r in a} & {r['group_id'] for r in b})

    def test_paired_augmentation_alignment(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            a = np.zeros((32,32), np.uint8); a[3:15,5:22] = 255
            Image.fromarray(np.stack([a]*3,-1)).save(p/'x.png')
            Image.fromarray(a).save(p/'y.png')
            record={'sample_id':'x','image':'x.png','mask':'y.png'}
            for epoch in range(8):
                x,y=TrainingDataset(p,[record],32,42,epoch)[0]
                raw=x[0]*.229+.485
                self.assertTrue(torch.equal(raw>.5,y[0].bool()))

    def test_metric_empty_and_disjoint(self):
        empty=np.zeros((4,4),bool); one=empty.copy();one[0,0]=True
        other=empty.copy();other[3,3]=True
        self.assertEqual(binary_metrics(empty,empty)['dice'],1)
        self.assertEqual(binary_metrics(empty,one)['dice'],0)
        self.assertEqual(binary_metrics(one,other)['dice'],0)

    def test_cuda_unavailable_never_falls_back(self):
        with patch('torch.cuda.is_available',return_value=False):
            with self.assertRaisesRegex(RuntimeError,'No CPU fallback'):
                require_cuda()

    def test_resnet_unet_shape_grad_and_initialization(self):
        torch.manual_seed(1)
        model=make_model({'experiment':'random'}).train()
        x=torch.randn(2,3,64,64)
        logits=model(x)
        self.assertEqual(tuple(logits.shape),(2,1,64,64))
        loss,_,_=segmentation_loss(logits,torch.zeros_like(logits));loss.backward()
        self.assertTrue(torch.isfinite(model.encoder.conv1.weight.grad).all())
        self.assertTrue(torch.isfinite(model.head.weight.grad).all())
        state={k:v.detach().clone() for k,v in model.encoder.state_dict().items()}
        state['conv1.weight'].fill_(.01)
        with patch('models.read_pretrained',return_value=(state,'synthetic-test-only')):
            initialized=ResNet34UNet('synthetic-fixture')
        self.assertTrue(torch.equal(initialized.encoder.conv1.weight,state['conv1.weight']))
        # Loading a trained checkpoint never attempts to obtain ImageNet weights.
        with patch('models.read_pretrained',side_effect=AssertionError('unexpected initialization')):
            reloaded=make_model({'experiment':'pretrained'},initialization=False)
            reloaded.load_state_dict(initialized.state_dict(),strict=True)
        self.assertTrue(torch.equal(reloaded.encoder.conv1.weight,initialized.encoder.conv1.weight))
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'bad.pth';path.write_bytes(b'not official weights')
            with self.assertRaisesRegex(ValueError,'hash'):
                read_pretrained(path)

    def test_partial_accumulation_matches_full_batch(self):
        torch.manual_seed(9)
        config={'experiment':'original','base_channels':2}
        a,b=make_model(config),make_model(config);b.load_state_dict(a.state_dict())
        x=torch.randn(5,3,32,32); y=(torch.rand(5,1,32,32)>.8).float()
        segmentation_loss(a(x),y)[0].backward()
        for start in (0,2,4):
            z=x[start:start+2]; target=y[start:start+2]
            (segmentation_loss(b(z),target)[0]*len(z)/5).backward()
        for p,q in zip(a.parameters(),b.parameters()):
            torch.testing.assert_close(p.grad,q.grad,atol=2e-7,rtol=2e-5)

    def test_final_schedule_keeps_development_horizon(self):
        cfg={'lr':1e-4,'warmup_epochs':5,'schedule_horizon':100,'epochs':100}
        final=dict(cfg,epochs=34)
        for epoch in range(1,35):
            self.assertEqual(lr_at_epoch(cfg,epoch),lr_at_epoch(final,epoch))
        self.assertAlmostEqual(lr_at_epoch(cfg,100),1e-5)

    def test_smoke_cannot_select_final_recipe(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);run=root/'smoke';run.mkdir()
            (run/'run_info.json').write_text(json.dumps({'status':'TRAINED','run_kind':'GPU_SMOKE',
                                                       'config':{'smoke':True}}))
            with self.assertRaisesRegex(ValueError,'formal development'):
                select_recipe(root,['smoke'],root/'selection.json')
            self.assertFalse((root/'selection.json').exists())

    def test_development_test_evaluation_is_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);run=root/'development';run.mkdir()
            (run/'run_info.json').write_text(json.dumps({'status':'TRAINED','run_kind':'E2_DEVELOPMENT'}))
            with self.assertRaisesRegex(ValueError,'Development/smoke'):
                evaluate_final(root,run,'cpu','missing_e0',None)
            self.assertFalse(list(run.glob('evaluation_*')))

    def test_epoch_resume_matches_uninterrupted_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);records=[]
            rng=np.random.default_rng(5)
            for i in range(8):
                rgb=rng.integers(0,256,(32,32,3),dtype=np.uint8)
                mask=(rgb[:,:,0]>180).astype(np.uint8)*255
                Image.fromarray(rgb).save(root/f'x{i}.png');Image.fromarray(mask).save(root/f'y{i}.png')
                records.append({'sample_id':f'toy/{i}','image':f'x{i}.png','mask':f'y{i}.png'})
            data={'dev_train':records[:6],'validation':records[6:],'full_train':records,'test':[]}
            bundle={'bundle_signature':signature(data)}
            config={'experiment':'original','base_channels':2,'phase':'development','epochs':2,
                    'schedule_horizon':2,'warmup_epochs':0,'input_size':32,'batch_size':2,'accumulation':2,
                    'lr':1e-4,'weight_decay':1e-4,'seed':42,'amp':False,'augmentation':'flips','smoke':False}
            execute_training(root,root/'continuous',config,bundle,data,torch.device('cpu'),0,_test_only=True)
            interrupted=dict(config,test_interrupt_epoch=1)
            with self.assertRaises(KeyboardInterrupt):
                execute_training(root,root/'interrupted',interrupted,bundle,data,torch.device('cpu'),0,_test_only=True)
            # Simulate interruption after writing a newer best but before the
            # last-epoch checkpoint commit. Resume must restore the backup.
            best=root/'interrupted/checkpoints/best.pth'
            shutil.copyfile(best,root/'interrupted/checkpoints/best_backup.pth')
            torch.save({'uncommitted_new_best':True},best)
            execute_training(root,root/'interrupted',interrupted,bundle,data,torch.device('cpu'),0,resume=True,_test_only=True)
            a=load_checkpoint(root/'continuous/checkpoints/final.pth')
            b=load_checkpoint(root/'interrupted/checkpoints/final.pth')
            for key,value in a['model_state'].items():
                torch.testing.assert_close(value,b['model_state'][key],rtol=0,atol=0)
            with self.assertRaisesRegex(RuntimeError,'blocked'):
                execute_training(root,root/'forbidden',config,bundle,data,torch.device('cpu'),0)


if __name__=='__main__':
    unittest.main(verbosity=2)
