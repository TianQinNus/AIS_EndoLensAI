"""Check region decisions against direct masks and causal temporal edge cases."""
import unittest
import numpy as np
from postprocess import components,retained,metrics,confirm_sequence,grid

def region(box=(0,0,4,4),area=16,tp=0,score=.9):
    return {'bbox_xyxy':list(box),'area':area,'tp':tp,'mean_probability':score,'centroid':[2.,2.]}

class Filters(unittest.TestCase):
    def test_baseline_matches_direct_binary_mask(self):
        rng=np.random.default_rng(42);p=rng.random((24,32),dtype=np.float32);gt=rng.random(p.shape)>.7
        for threshold in [.5,.6,.7,.8,.9]:
            r=components(p,gt,threshold);m=metrics(r,int(gt.sum()),gt.size);mask=p>=threshold
            self.assertEqual(m['tp'],int((mask&gt).sum()));self.assertEqual(m['fp'],int((mask&~gt).sum()))
            self.assertEqual(m['pred_pixels'],int(mask.sum()))

    def test_8_connectivity_area_and_mean_score(self):
        p=np.zeros((10,10),np.float32);gt=np.zeros((10,10),bool)
        p[1,1]=.8;p[2,2]=.9;p[7:9,7:9]=.6;gt[1,1]=True
        regions=components(p,gt,.5);self.assertEqual(sorted(r['area'] for r in regions),[2,4])
        config={'min_component_area_fraction':.03,'min_component_mean_probability':0.}
        kept=retained(regions,config,100);self.assertEqual(len(kept),1);self.assertEqual(kept[0]['area'],4)
        config={'min_component_area_fraction':.02,'min_component_mean_probability':.8}
        kept=retained(regions,config,100);self.assertEqual(len(kept),1);self.assertEqual(kept[0]['tp'],1)

    def test_filter_can_increase_dice_by_removing_wrong_region(self):
        regions=[region(tp=16),region((5,5,6,6),area=1,tp=0,score=.55)]
        config={'min_component_area_fraction':.02,'min_component_mean_probability':0.}
        before=metrics(regions,16,100);after=metrics(retained(regions,config,100),16,100)
        self.assertGreater(after['dice'],before['dice']);self.assertEqual(after['dice'],1.)

    def test_empty_predictions_and_threshold_comparison(self):
        self.assertEqual(metrics([],0,100)['dice'],1.)
        self.assertEqual(metrics([],4,100)['dice'],0.)
        p=np.zeros((3,3),np.float32);p[0,0]=.5
        self.assertEqual(len(components(p,np.zeros_like(p,bool),.5)),1)
        self.assertEqual(len(components(p,np.zeros_like(p,bool),.6)),0)

    def test_temporal_causal_streaks_and_gaps(self):
        frames=[{'sample_id':str(i),'frame_number':i,'regions':[region()]} for i in [1,2,3,10,11,12]]
        out=confirm_sequence(frames,k=3)
        self.assertEqual([len(f['regions']) for f in out],[0,0,1,0,0,1])
        self.assertEqual([f['warmup'] for f in out],[True,True,False,True,True,False])
        self.assertTrue(all(len(f['regions'])==1 for f in confirm_sequence(frames,k=1)))

    def test_temporal_different_locations_and_one_to_one(self):
        frames=[{'sample_id':'1','frame_number':1,'regions':[region()]},
            {'sample_id':'2','frame_number':2,'regions':[region(),region((0,0,3,3),9)]},
            {'sample_id':'3','frame_number':3,'regions':[region((50,50,54,54))]}]
        out=confirm_sequence(frames,k=2)
        self.assertEqual([len(f['regions']) for f in out],[0,1,0])

    def test_grid_contains_each_single_factor_and_baseline(self):
        settings=grid();self.assertEqual(len(settings),150)
        self.assertEqual(len({x['id'] for x in settings}),150)
        self.assertTrue(any(x['pixel_threshold']==.5 and x['min_component_area_fraction']==0 and x['min_component_mean_probability']==0 for x in settings))

if __name__=='__main__':unittest.main(verbosity=2)
