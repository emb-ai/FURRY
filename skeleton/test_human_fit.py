import copy
import unittest
import numpy as np
from human_fit import HumanLegFit, NAMES, EDGES

class HumanFitTest(unittest.TestCase):
    def setUp(self):
        self.xyz=np.array([[-.1,0,2],[.1,0,2],[-.1,.38,2.12],[.1,.38,2.12],[-.1,.77,2.03],[.1,.77,2.03]])
        self.lengths=np.array([np.linalg.norm(self.xyz[a]-self.xyz[b]) for a,b in EDGES])
        self.packet={'joints':{n:{'p':p.tolist(),'conf':.95,'src':'window'} for n,p in zip(NAMES,self.xyz)}}
        self.fit=HumanLegFit(self.lengths)

    def test_noisy_lengths_and_input_unchanged(self):
        p=copy.deepcopy(self.packet);p['joints']['lknee']['p'][2]+=.04
        before=copy.deepcopy(p);r=self.fit.fit(p)
        self.assertTrue(r['accepted']);self.assertEqual(p,before)
        x=np.array([r['points'][n] for n in NAMES])
        np.testing.assert_allclose([np.linalg.norm(x[a]-x[b]) for a,b in EDGES],self.lengths,atol=1e-4)

    def test_no_invented_missing_and_prior_labelled(self):
        p=copy.deepcopy(self.packet);p['joints']['lankle']['src']='missing'
        self.assertFalse(self.fit.fit(p)['accepted'])
        r=self.fit.fit(p,{'lankle':self.xyz[4]})
        self.assertTrue(r['accepted']);self.assertEqual(r['sources']['lankle'],'predicted')

    def test_gross_depth_outlier_rejected(self):
        p=copy.deepcopy(self.packet);p['joints']['lankle']['p'][2]+=1.5
        self.assertFalse(self.fit.fit(p)['accepted'])

    def test_rigid_frame_equivariance(self):
        p=copy.deepcopy(self.packet);p['joints']['lknee']['p'][2]+=.03
        r=self.fit.fit(p);R=np.array([[0,-1,0],[1,0,0],[0,0,1]]);t=np.array([1,2,-1])
        for j in p['joints'].values():j['p']=(R@np.array(j['p'])+t).tolist()
        s=self.fit.fit(p);self.assertTrue(s['accepted'])
        for n in NAMES:np.testing.assert_allclose(s['points'][n],R@np.array(r['points'][n])+t,atol=2e-4)

    def test_calibration_requires_evidence(self):
        with self.assertRaises(ValueError):HumanLegFit.calibrate([self.packet])
        np.testing.assert_allclose(HumanLegFit.calibrate([self.packet]*30).lengths,self.lengths)

if __name__=='__main__':unittest.main()
