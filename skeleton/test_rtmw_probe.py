import unittest
import numpy as np
from rtmw_probe import metric_points,monocular_prior

class DepthTest(unittest.TestCase):
    def test_metric_axes_and_invalid_patch(self):
        d=np.ones((30,30))*2
        p=metric_points(np.array([[15,10],[-1,10],[10,10]]),np.array([.9,.9,.1]),d,[100,100,10,10])
        self.assertEqual(list(p),[0]);np.testing.assert_allclose(p[0]['p'],[.1,0,2])
        d[7:14,12:19]=0
        self.assertEqual(metric_points(np.array([[15,10]]),[.9],d,[100,100,10,10]),{})

    def test_mixed_foreground_background_rejected(self):
        d=np.ones((30,30))*2;d[:,15:]=4
        self.assertEqual(metric_points(np.array([[15,10]]),[.9],d,[100,100,10,10]),{})

    def test_prior_needs_consistent_metric_anchors(self):
        xy=np.ones((17,2))*10;z=np.zeros(17);scores=np.ones(17)
        anchors={i:{'p':[0,0,2],'conf':.9} for i in (5,6,11,12)}
        p=monocular_prior(xy,scores,z,anchors,[100,100,10,10])
        np.testing.assert_allclose(p[15],[0,0,2])
        self.assertEqual(monocular_prior(xy,scores,z,{},[100,100,10,10]),{})
        anchors[12]['p'][2]=4
        self.assertEqual(monocular_prior(xy,scores,z,anchors,[100,100,10,10]),{})

if __name__=='__main__':unittest.main()
