"""Check actual availability and exercise U-Net forward/backward on the chosen device."""
import argparse
import platform
import sys
import torch
from unet import UNet, segmentation_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("mps", "cpu", "cuda"), default="mps")
    args = parser.parse_args()
    print("Python:", sys.version.split()[0], "Architecture:", platform.machine())
    print("PyTorch:", torch.__version__)
    print("MPS built:", torch.backends.mps.is_built())
    print("MPS available:", torch.backends.mps.is_available())
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS unavailable. Use native arm64 Python and a current stable PyTorch in the local Terminal. See README_E1_MAC.md.")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable.")
    device = torch.device(args.device)
    torch.manual_seed(42)
    model = UNet(base_channels=8).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    x = torch.randn(2, 3, 64, 64, device=device)
    target = (torch.rand(2, 1, 64, 64, device=device) > 0.7).float()
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss, _, _ = segmentation_loss(logits, target)
        loss.backward()
        optimizer.step()
        if not torch.isfinite(loss).item():
            raise RuntimeError("Nonfinite loss")
    if device.type == "mps":
        torch.mps.synchronize()
    print(f"PASS: U-Net forward, BCE + Dice backward and AdamW on {device}; loss={loss.item():.6f}")
    print("This is a small device probe, not an E1 experiment result.")


if __name__ == "__main__":
    main()
