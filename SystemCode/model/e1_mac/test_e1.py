"""Meaningful CPU checks for paired augmentation, accumulation, and safe checkpoints."""
from pathlib import Path
import tempfile
import unittest
import copy
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from PIL import Image
import torch
from unet import UNet, segmentation_loss
import run_e1
from run_e1 import TrainingDataset, atomic_save, load_checkpoint
from e0_protocol import binary_metrics


class E1Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)

    def test_unet_shape_gradient_and_roundtrip(self):
        model = UNet(4)
        x = torch.randn(2, 3, 64, 64)
        y = (torch.rand(2, 1, 64, 64) > 0.8).float()
        logits = model(x)
        self.assertEqual(logits.shape, y.shape)
        loss, _, _ = segmentation_loss(logits, y)
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        optimizer.step()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.pth"
            atomic_save({"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "epoch": 1}, path)
            state = load_checkpoint(path)
            other = UNet(4)
            other.load_state_dict(state["model_state"], strict=True)
            self.assertTrue(torch.equal(model(x), other(x)))
            self.assertEqual(state["epoch"], 1)

    def test_partial_group_accumulation_matches_full_batch(self):
        # Five samples split 2+2+1; BCE+Dice loss uses per-image means.
        x = torch.randn(5, 3, 32, 32)
        y = (torch.rand(5, 1, 32, 32) > .5).float()
        full = UNet(2)
        split = copy.deepcopy(full)
        segmentation_loss(full(x), y)[0].backward()
        for start in range(0, 5, 2):
            xx, yy = x[start:start+2], y[start:start+2]
            (segmentation_loss(split(xx), yy)[0] * len(xx) / 5).backward()
        for a, b in zip(full.parameters(), split.parameters()):
            torch.testing.assert_close(a.grad, b.grad, rtol=5e-4, atol=1e-6)

    def test_augmentation_preserves_image_mask_alignment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mask = np.zeros((32, 32), dtype=np.uint8)
            mask[3:14, 9:25] = 255
            Image.fromarray(np.stack((mask, mask, mask), axis=-1)).save(root / "image.png")
            Image.fromarray(mask).save(root / "mask.png")
            records = [{"image": "image.png", "mask": "mask.png"}]
            for epoch in range(8):
                dataset = TrainingDataset(root, records, 32, 42, epoch)
                x, y = dataset[0]
                recovered = x[0].numpy() * .229 + .485
                np.testing.assert_array_equal(recovered > .5, y[0].numpy() > .5)
                x2, y2 = dataset[0]
                self.assertTrue(torch.equal(x, x2) and torch.equal(y, y2))

    def test_e0_metric_empty_and_disjoint_rules(self):
        empty = np.zeros((4, 4), bool)
        full = ~empty
        self.assertEqual(binary_metrics(empty, empty)["dice"], 1)
        self.assertEqual(binary_metrics(full, empty)["iou"], 0)
        gt = empty.copy(); gt[0, 0] = True
        pred = empty.copy(); pred[3, 3] = True
        self.assertEqual(binary_metrics(pred, gt)["dice"], 0)

    def test_epoch_resume_matches_uninterrupted_training(self):
        # A small synthetic dataset tests the actual training/checkpoint engine,
        # including a partial accumulation group and an interruption after save.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "dataset/TrainDataset/images").mkdir(parents=True)
            records = []
            rng = np.random.default_rng(5)
            for i in range(3):
                rgb = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
                gt = (rgb[:, :, 0] > 128).astype(np.uint8) * 255
                Image.fromarray(rgb).save(root / f"x{i}.png")
                Image.fromarray(gt).save(root / f"y{i}.png")
                records.append({"image": f"x{i}.png", "mask": f"y{i}.png"})
            def fake_audit(project, e0, output):
                run_e1.write_json(output / "train_manifest.json", records)
                run_e1.write_json(output / "test_manifest.json", [])
                return records, [], root / "e0"
            def args(output=None, resume=None):
                return SimpleNamespace(project=str(root), device="cpu", output=str(output) if output else None,
                    resume=str(resume) if resume else None, epochs=2, batch_size=2, accumulation=4,
                    lr=1e-4, weight_decay=1e-4, seed=42, input_size=32, base_channels=2,
                    smoke=False, workers=0, log_every=20, train_only=True, warmup=0, e0_run=None)
            uninterrupted = root / "uninterrupted"
            interrupted = root / "interrupted"
            with patch.object(run_e1, "audit_into", side_effect=fake_audit), patch.object(run_e1, "prepare_data", return_value=(records, [], {}, {"passed": True})):
                run_e1.run_training(args(uninterrupted))
                real_save = run_e1.atomic_save
                def interrupt_after_save(value, path):
                    real_save(value, path)
                    if path.name == "last.pth" and value["epoch"] == 1:
                        raise KeyboardInterrupt("simulated interruption")
                with patch.object(run_e1, "atomic_save", side_effect=interrupt_after_save):
                    with self.assertRaises(KeyboardInterrupt):
                        run_e1.run_training(args(interrupted))
                run_e1.run_training(args(resume=interrupted))
            full = load_checkpoint(uninterrupted / "checkpoints/unet_final.pth")
            resumed = load_checkpoint(interrupted / "checkpoints/unet_final.pth")
            self.assertEqual(resumed["epoch"], 2)
            for name in full["model_state"]:
                self.assertTrue(torch.equal(full["model_state"][name], resumed["model_state"][name]), name)


if __name__ == "__main__":
    unittest.main(verbosity=2)
