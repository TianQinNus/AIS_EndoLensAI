"""Deterministic region filters and causal confirmation, with no learned changes."""
from pathlib import Path
import sys
import numpy as np
try:
    import cv2
except ImportError:
    sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'.handoff_test_deps'))
    import cv2
cv2.setNumThreads(4)

THRESHOLDS=[.5,.6,.7,.8,.9]
AREAS=[0.,.0001,.0005,.001,.002,.005]
CONFIDENCES=[0.,.7,.8,.9,.95]

def grid():
    rows=[]
    for t in THRESHOLDS:
        for a in AREAS:
            for c in CONFIDENCES:
                rows.append({'id':f't{t:.2f}_a{a:.4f}_c{c:.2f}',
                    'pixel_threshold':t,'min_component_area_fraction':a,'min_component_mean_probability':c})
    return rows

def components(probability,gt,threshold):
    """8-connected foreground; every component keeps exact native-size counts."""
    mask=(probability>=threshold).astype(np.uint8)
    count,labels,stats,centers=cv2.connectedComponentsWithStats(mask,connectivity=8,ltype=cv2.CV_32S)
    if count==1:return []
    sums=np.bincount(labels.ravel(),weights=probability.ravel(),minlength=count)
    true=np.bincount(labels[gt],minlength=count)
    regions=[]
    for label in range(1,count):
        x,y,w,h,area=(int(x) for x in stats[label])
        regions.append({'area':area,'tp':int(true[label]),'mean_probability':float(sums[label]/area),
            'bbox_xyxy':[x,y,x+w,y+h],'centroid':[float(x) for x in centers[label]]})
    return regions

def retained(regions,config,image_area):
    return [r for r in regions if r['area']/image_area>=config['min_component_area_fraction']
        and r['mean_probability']>=config['min_component_mean_probability']]

def metrics(regions,gt_pixels,image_area):
    tp=sum(r['tp'] for r in regions);pred=sum(r['area'] for r in regions)
    fp=pred-tp;fn=gt_pixels-tp;den=pred+gt_pixels;union=tp+fp+fn
    return {'tp':tp,'fp':fp,'fn':fn,'pred_pixels':pred,'dice':2*tp/den if den else 1.,
        'iou':tp/union if union else 1.,'any_foreground':int(pred>0),
        'pred_area_fraction':pred/image_area,'area_ge_0_001':int(pred/image_area>=.001),
        'area_ge_0_01':int(pred/image_area>=.01),'overlap_hit':int(tp>0),
        'empty_prediction':int(pred==0)}

def bbox_iou(a,b):
    overlap=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-overlap
    return overlap/union if union else 0.

def confirm_sequence(frames,k=3,min_iou=.1):
    """Only adjacent numeric frame IDs; greedy 1:1 box matches; no lookahead.

    Frames: sorted dictionaries with frame_number and retained regions. Returned
    masks contain current-frame components only after k consecutive matches.
    No past frame is backfilled. Missing numeric IDs reset all streaks.
    """
    if k<1:raise ValueError('k must be positive')
    previous=[];last=None;out=[];segment_age=0
    for f in frames:
        number=f['frame_number']
        if last is None or number!=last+1:previous=[];segment_age=0
        segment_age+=1;current=f['regions'];streaks=[1]*len(current)
        pairs=[]
        for i,r in enumerate(current):
            for j,p in enumerate(previous):
                overlap=bbox_iou(r['bbox_xyxy'],p['region']['bbox_xyxy'])
                if overlap>=min_iou:pairs.append((overlap,i,j))
        used_current=set();used_previous=set()
        for _,i,j in sorted(pairs,reverse=True):
            if i not in used_current and j not in used_previous:
                streaks[i]=previous[j]['streak']+1;used_current.add(i);used_previous.add(j)
        kept=[r for r,s in zip(current,streaks) if s>=k]
        out.append({'sample_id':f['sample_id'],'frame_number':number,'regions':kept,'warmup':segment_age<k})
        previous=[{'region':r,'streak':s} for r,s in zip(current,streaks)];last=number
    return out
