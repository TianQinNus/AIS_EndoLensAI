"""Standard padded U-Net: four down/up levels, no pretrained encoder or BatchNorm."""
import torch
from torch import nn


class DoubleConv(nn.Sequential):
    def __init__(self, cin, cout):
        super().__init__(
            nn.Conv2d(cin, cout, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1), nn.ReLU(inplace=True),
        )


class Up(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.up = nn.ConvTranspose2d(cin, cout, 2, stride=2)
        self.conv = DoubleConv(2 * cout, cout)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            raise ValueError("U-Net input height/width must be divisible by 16")
        return self.conv(torch.cat((skip, x), dim=1))


class UNet(nn.Module):
    def __init__(self, base_channels=64):
        super().__init__()
        b = base_channels
        self.inc = DoubleConv(3, b)
        self.pool = nn.MaxPool2d(2)
        self.down = nn.ModuleList([DoubleConv(b * 2**i, b * 2**(i + 1)) for i in range(4)])
        self.up = nn.ModuleList([Up(b * 2**i, b * 2**(i - 1)) for i in range(4, 0, -1)])
        self.out = nn.Conv2d(b, 1, 1)

    def forward(self, x):
        features = [self.inc(x)]
        for block in self.down:
            features.append(block(self.pool(features[-1])))
        x = features[-1]
        for block, skip in zip(self.up, reversed(features[:-1])):
            x = block(x, skip)
        return self.out(x)  # logits; sigmoid belongs in loss/evaluation


def segmentation_loss(logits, target):
    """BCE + per-image soft Dice loss, each coefficient 1, smoothing 1."""
    bce = nn.functional.binary_cross_entropy_with_logits(logits, target)
    probability = torch.sigmoid(logits)
    axes = (1, 2, 3)
    intersection = (probability * target).sum(dim=axes)
    denominator = probability.sum(dim=axes) + target.sum(dim=axes)
    dice = (2 * intersection + 1.0) / (denominator + 1.0)
    dice_loss = (1.0 - dice).mean()
    return bce + dice_loss, bce, dice_loss
