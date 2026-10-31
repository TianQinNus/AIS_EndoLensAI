"""Common 352-RGB/logit interface; no extra sigmoid or image min-max scaling."""
import importlib.util
import os
import sys
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from common import ROOT, STARTER, WEIGHTS, EXPECTED_WEIGHTS, sha

SIZE = 352
MEAN = (.485, .456, .406)
STD = (.229, .224, .225)


def setup(device='cuda'):
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA required; activate endolens-e2-win. No CPU training fallback.')
    result = torch.device('cuda:0' if device == 'cuda' else 'cpu')
    if result.type == 'cuda':
        x = torch.randn(8, 8, device=result)
        (x@x).sum().item()
    return result


def build(model_key, device, checkpoint=None, verify=True):
    if model_key not in WEIGHTS:
        raise ValueError('Unknown model: ' + model_key)
    path = WEIGHTS[model_key] if checkpoint is None else checkpoint
    if verify and checkpoint is None and sha(path) != EXPECTED_WEIGHTS[model_key]:
        raise ValueError('Frozen weight mismatch')
    state = torch.load(path, map_location='cpu', weights_only=True)
    if model_key == 'r34':
        core = str(ROOT/'e4_windows/core')
        if core not in sys.path:
            sys.path.insert(0, core)
        from models import ResNet34UNet
        model = ResNet34UNet()
        weights = state.get('model_state', state.get('model'))
    elif model_key == 'r50':
        spec = importlib.util.spec_from_file_location('e12_reviewed_r50', STARTER/'model/resnet50_model.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        model = module.UNet()
        weights = state['model']
    else:
        sys.path.insert(0, str(ROOT/'e6_windows'))
        import pvt_timm_compat
        pvt_timm_compat.install_if_missing()
        sys.path.insert(0, str(ROOT))
        from lib.pvt import PolypPVT
        # Historical constructor loads the local backbone from a relative path.
        previous = os.getcwd()
        try:
            os.chdir(ROOT)
            model = PolypPVT()
        finally:
            os.chdir(previous)
        weights = state
    if weights is None:
        raise ValueError('Missing model state in checkpoint')
    model.load_state_dict(weights, strict=True)
    model.to(device).eval()
    return model


def train_mode(model):
    model.train()
    # Common policy for both U-Nets at physical batch 2; BN affine remains trainable.
    for module in model.encoder.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.eval()


def forward(model, model_key, x):
    if model_key != 'r50':
        mean = x.new_tensor(MEAN).view(1, 3, 1, 1)
        std = x.new_tensor(STD).view(1, 3, 1, 1)
        x = (x-mean)/std
    value = model(x)
    if model_key == 'pvt':
        if not isinstance(value, (tuple, list)) or len(value) != 2:
            raise ValueError('Expected the original two PVT output branches')
        value = value[0]+value[1]
    return value


def input_tensor(image, size=SIZE):
    a = np.asarray(image.convert('RGB').resize((size, size), Image.Resampling.BILINEAR), np.float32)/255
    return torch.from_numpy(np.ascontiguousarray(a.transpose(2, 0, 1)))


def probabilities(model, model_key, x, shape, tta='identity'):
    if tta == 'identity':
        logits = forward(model, model_key, x)
    elif tta == 'rotate4':
        logits = sum(torch.rot90(forward(model, model_key, torch.rot90(x, k, (-2, -1))), -k, (-2, -1))
                     for k in range(4))/4
    else:
        raise ValueError('Unknown predeclared inference mode')
    resized = F.interpolate(logits.float(), size=shape, mode='bilinear', align_corners=False)
    return torch.sigmoid(resized)[0, 0].cpu().numpy().astype(np.float32)


def mixed_loss(logits, target):
    """Mean over images of BCE + I(nonempty GT)*soft Dice. Empty batches are safe."""
    z, y = logits.float(), target.float()
    dims = (1, 2, 3)
    bce = F.binary_cross_entropy_with_logits(z, y, reduction='none').mean(dims)
    p = z.sigmoid()
    dice = 1-(2*(p*y).sum(dims)+1)/(p.sum(dims)+y.sum(dims)+1)
    positive = (y.sum(dims)>0).to(z.dtype)
    return (bce+positive*dice).mean(), bce.mean(), (positive*dice).mean()
