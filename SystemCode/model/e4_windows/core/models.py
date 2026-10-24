"""E2 models: original U-Net and a documented ResNet34-U-Net variant.

Torchvision supplies the published ResNet34 encoder. This package adapts a
U-Net decoder to its multiscale features. No original architecture claim.
"""
from pathlib import Path
import hashlib
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet34
from original_unet import UNet

WEIGHT_FILENAME = "resnet34-b627a593.pth"
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
    # Official torchvision IMAGENET1K_V1 filename carries this hash prefix.
    if not digest.startswith("b627a593"):
        raise ValueError("ResNet34 weight hash does not match official b627a593 prefix")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or "conv1.weight" not in state:
        raise ValueError("Invalid official ResNet34 state dictionary")
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


class ResNet34UNet(nn.Module):
    def __init__(self, pretrained_path=None):
        super().__init__()
        # Never download silently, particularly when loading a trained model.
        self.encoder = resnet34(weights=None)
        self.encoder.fc = nn.Identity()
        self.initial_weight_sha256 = None
        if pretrained_path is not None:
            state, self.initial_weight_sha256 = read_pretrained(pretrained_path)
            self.encoder.load_state_dict(state, strict=True)
        self.decoder = nn.ModuleList([
            DecoderBlock(512, 256, 256), DecoderBlock(256, 128, 128),
            DecoderBlock(128, 64, 64), DecoderBlock(64, 64, 32),
            DecoderBlock(32, 0, 16),
        ])
        self.head = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        original_size = x.shape[-2:]
        e = self.encoder
        f0 = e.relu(e.bn1(e.conv1(x)))                  # H/2, 64
        f1 = e.layer1(e.maxpool(f0))                   # H/4, 64
        f2 = e.layer2(f1)                              # H/8, 128
        f3 = e.layer3(f2)                              # H/16, 256
        f4 = e.layer4(f3)                              # H/32, 512
        z = f4
        for block, skip in zip(self.decoder[:4], (f3, f2, f1, f0)):
            z = block(z, skip)
        z = self.decoder[4](z, size=original_size)
        return self.head(z)                            # logits, full resolution


def make_model(config, project=None, initialization=True):
    if config["experiment"] == "original":
        return UNet(config.get("base_channels", 64))
    weights = None
    if config["experiment"] == "pretrained" and initialization:
        if project is None:
            raise ValueError("Project path required for pretrained initialization")
        weights = Path(project) / config["encoder_weights"]
    return ResNet34UNet(weights)


def segmentation_loss(logits, target):
    # FP32 reductions are important when the forward pass uses float16 AMP.
    logits, target = logits.float(), target.float()
    bce = F.binary_cross_entropy_with_logits(logits, target)
    probability = logits.sigmoid()
    axes = (1, 2, 3)
    inter = (probability * target).sum(axes)
    denominator = probability.sum(axes) + target.sum(axes)
    dice = 1 - ((2 * inter + 1) / (denominator + 1)).mean()
    return bce + dice, bce, dice
