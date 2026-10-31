import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from model.resnet50_model import UNet
p=argparse.ArgumentParser()
p.add_argument('--image',required=True)
p.add_argument('--output',required=True)
p.add_argument('--device',default='cpu',choices=['cpu','mps','cuda'])
a=p.parse_args()
root=Path(__file__).resolve().parent
cp=torch.load(root/'weights/best.pt',map_location='cpu',weights_only=True)
m=UNet().to(a.device);m.load_state_dict(cp['model'],strict=True);m.eval()
im=Image.open(a.image).convert('RGB');w,h=im.size
x=torch.from_numpy(np.asarray(im.resize((320,320),Image.Resampling.BILINEAR),dtype=np.float32).transpose(2,0,1).copy()/255)[None].to(a.device)
with torch.inference_mode():
 logits=torch.stack([torch.rot90(m(torch.rot90(x,k,(-2,-1))),-k,(-2,-1)) for k in range(4)]).mean(0)
 prob=F.interpolate(logits,(h,w),mode='bilinear',align_corners=False).sigmoid()
 mask=(prob[0,0]>=.5).cpu().numpy().astype('uint8')*255
Image.fromarray(mask).save(a.output)
