#!/usr/bin/env python3
"""Real CUDA forward/backward/AdamW probe. Never silently use CPU."""
import argparse
import json
from pathlib import Path
import torch
from engine import environment, require_cuda, sync
from models import make_model, segmentation_loss


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", choices=("cuda",), default="cuda")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--input-size", type=int, default=352)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--output", help="Optional JSON device proof")
    a = p.parse_args()
    if min(a.batch_size, a.input_size) < 1:
        p.error("Batch size and input size must be positive")
    device = require_cuda()
    model = make_model({"experiment": "random"}).to(device).train()
    x = torch.rand(a.batch_size, 3, a.input_size, a.input_size, device=device)
    y = (torch.rand(a.batch_size, 1, a.input_size, a.input_size, device=device) > .8).float()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=not a.no_amp)
    before = model.head.weight.detach().clone()
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=not a.no_amp):
            logits = model(x)
        loss, _, _ = segmentation_loss(logits, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    sync(device)
    if not (loss.isfinite() and logits.is_cuda and x.is_cuda and next(model.parameters()).is_cuda and not torch.equal(before, model.head.weight)):
        raise RuntimeError("CUDA model probe failed")
    result = {"passed": True, "environment": environment(device), "model_device": str(next(model.parameters()).device),
              "input_device": str(x.device), "output_device": str(logits.device),
              "loss": float(loss), "amp": not a.no_amp, "forward_dtype": str(logits.dtype),
              "peak_cuda_allocated_mb": torch.cuda.max_memory_allocated(device) / 2**20,
              "probe": "ResNet34-U-Net at actual resolution, forward + BCE/Dice backward + AdamW parameter update",
              "disclaimer": "A device probe, not an experiment score"}
    print(json.dumps(result, indent=2))
    if a.output:
        path = Path(a.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("PASS: REAL CUDA FORWARD, BACKWARD AND PARAMETER UPDATE. NO CPU FALLBACK.")


if __name__ == "__main__":
    main()
