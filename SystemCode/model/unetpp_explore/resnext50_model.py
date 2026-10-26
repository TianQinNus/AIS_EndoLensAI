"""E2 models: original U-Net and a documented ResNeXt50-U-Net variant.

Torchvision supplies the published ResNeXt50 encoder. This package adapts a
U-Net decoder to its multiscale features. No original architecture claim.
"""
from pathlib import Path
import hashlib
import torch
from torch import nn
import torch.nn.functional as F
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]/'outputs/runtime'))
from torchvision.models import resnext50_32x4d

WEIGHT_FILENAME = "resnext50_32x4d-1a0047aa.pth"
WEIGHT_URL = "https://download.pytorch.org/models/" + WEIGHT_FILENAME


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_pretrained(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing ImageNet weights: {path}. Run download-weights first.")
    digest = sha256(path)
    # Official torchvision IMAGENET1K_V2 filename carries this hash prefix.
    if not digest.startswith("1a0047aa"):
        raise ValueError("ResNeXt50 weight hash does not match official 1a0047aa prefix")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or "conv1.weight" not in state:
        raise ValueError("Invalid official ResNeXt50 state dictionary")
    return {k: v for k, v in state.items() if not k.startswith("fc.")}, digest


class DecoderBlock(nn.Module):
    def __init__(self, cin, skip_channels, cout):
        super().__init__()
        # GroupNorm uses each sample's activations, not batch running statistics.
        self.conv = nn.Sequential(
            nn.Conv2d(cin + skip_channels, cout, 3, padding=1, bias=False),
            nn.GroupNorm(8, cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.GroupNorm(8, cout), nn.ReLU(inplace=True),
        )

    def forward(self, x, skip=None, size=None):
        target = skip.shape[-2:] if skip is not None else size
        x = F.interpolate(x, size=target, mode="bilinear", align_corners=False)
        if skip is not None:
            x = torch.cat((x, skip), dim=1)
        return self.conv(x)


class ResNeXt50UNet(nn.Module):
    def __init__(self, pretrained_path=None):
        super().__init__()
        # Never download silently, particularly when loading a trained model.
        self.encoder = resnext50_32x4d(weights=None)
        self.encoder.fc = nn.Identity()
        self.initial_weight_sha256 = None
        if pretrained_path is not None:
            state, self.initial_weight_sha256 = read_pretrained(pretrained_path)
            self.encoder.load_state_dict(state, strict=True)
        self.decoder = nn.ModuleList([
            DecoderBlock(2048, 1024, 256), DecoderBlock(256, 512, 128),
            DecoderBlock(128, 256, 64), DecoderBlock(64, 64, 32),
            DecoderBlock(32, 0, 16),
        ])
        self.head = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        original_size = x.shape[-2:]
        e = self.encoder
        f0 = e.relu(e.bn1(e.conv1(x)))                  # H/2, 64
        f1 = e.layer1(e.maxpool(f0))                   # H/4, 256
        f2 = e.layer2(f1)                              # H/8, 512
        f3 = e.layer3(f2)                              # H/16, 1024
        f4 = e.layer4(f3)                              # H/32, 2048
        z = f4
        for block, skip in zip(self.decoder[:4], (f3, f2, f1, f0)):
            z = block(z, skip)
        z = self.decoder[4](z, size=original_size)
        return self.head(z)                            # logits, full resolution



class UNet(ResNeXt50UNet):
    """RGB /255 interface; embedded ImageNet normalization and frozen BN stats."""
    def __init__(self, base=32, pretrained_path=None):
        super().__init__(pretrained_path)
        self.register_buffer('input_mean', torch.tensor([.485, .456, .406]).view(1,3,1,1))
        self.register_buffer('input_std', torch.tensor([.229, .224, .225]).view(1,3,1,1))

    def forward(self, x):
        return super().forward((x-self.input_mean)/self.input_std)

    def train(self, mode=True):
        super().train(mode)
        for module in self.encoder.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
        return self
