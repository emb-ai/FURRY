import json
from pathlib import Path
import unittest
import torch
from clearance_reward import ClearanceConfig,velocity_clearance_cost,sole_geometry_cost,contact_point_slip


class ClearanceTests(unittest.TestCase):
    def cost(self,h,speed):
        h=torch.tensor([h],dtype=torch.float64)
        v=torch.zeros(h.shape+(2,),dtype=h.dtype);v[...,0]=torch.tensor(speed)
        return float(velocity_clearance_cost(h,v)[0])

    def test_low_transfer_is_penalized_and_lift_helps(self):
        costs=[self.cost([h,0],[1,0]) for h in [0,.001,.01,.03,.064,.065,.10]]
        self.assertTrue(all(a>b for a,b in zip(costs[:5],costs[1:6])))
        self.assertEqual(costs[-1],0);self.assertEqual(costs[-2],0)
        # Below the rejected v1 35mm gate still has a positive, smooth cost.
        self.assertGreater(costs[3],0)

    def test_stand_crouch_and_frozen_foot_escape(self):
        self.assertEqual(self.cost([0,0],[0,0]),0)
        # No torso or crouch pose appears; fixed feet have the same zero cost.
        self.assertEqual(self.cost([.001,0],[0,0]),0)
        # This limitation is deliberate evidence, not something a label masks.
        self.assertGreater(self.cost([.001,0],[1,0]),0)

    def test_centroid_rotation_and_glide_limitations(self):
        p=torch.tensor([[[[-.1,-.03,.005],[-.1,.03,.005],[.1,-.03,.005],[.1,.03,.005]],
                          [[-.1,-.03,.005],[-.1,.03,.005],[.1,-.03,.005],[.1,.03,.005]]]],dtype=torch.float64)
        previous=p.clone();angle=.02
        rot=torch.tensor([[torch.cos(torch.tensor(angle)), -torch.sin(torch.tensor(angle))],
                          [torch.sin(torch.tensor(angle)), torch.cos(torch.tensor(angle))]],dtype=p.dtype)
        p[0,0,:,:2]=p[0,0,:,:2]@rot.T
        self.assertAlmostEqual(float(sole_geometry_cost(p,previous,.005,.02)[0]),0,places=14)
        # About a planted toe, the centroid moves: cost is positive. No universal
        # exemption is claimed for grounded pivots or intentional skating.
        p[0,0,:,0]+=.002
        self.assertGreater(float(sole_geometry_cost(p,previous,.005,.02)[0]),0)

    def test_frame_invariance_and_reference_speed_counterexample(self):
        h=torch.tensor([[.001,0.]])
        v=torch.tensor([[[.8,.6],[0.,0.]]]);vr=torch.tensor([[[-.6,.8],[0.,0.]]])
        torch.testing.assert_close(velocity_clearance_cost(h,v)[0],velocity_clearance_cost(h,vr)[0])
        actual=torch.zeros_like(v)
        self.assertEqual(float(velocity_clearance_cost(h,actual)[0]),0)
        self.assertGreater(float(velocity_clearance_cost(h,v)[0]),0) # reference-speed alternative

    def test_contact_slip_keeps_all_contact_signal(self):
        p=torch.zeros((1,2,4,3));p[...,2]=.005;prev=p.clone();prev[...,0]-=.02
        args=[p,prev,.005,torch.zeros((1,2,3)),torch.ones((1,2))*400,.02,torch.tensor([True]),torch.tensor([[800.]])]
        self.assertAlmostEqual(float(contact_point_slip(*args)),2.)
        # Non-floor force at an airborne foot does not count as floor slip.
        args[0]=p+.1;args[1]=prev+.1
        self.assertEqual(float(contact_point_slip(*args)),0)

    def test_bounds_denominator_dt_and_invalid(self):
        expected=1-torch.exp(-.064**2*torch.tanh(torch.tensor(2.))/.05)
        self.assertAlmostEqual(self.cost([.001,0],[1,0]),float(expected),places=7)
        self.assertLessEqual(self.cost([-1e4,-1e4],[1e6,1e6]),1)
        self.assertAlmostEqual(-.02*self.cost([.001,0],[1,0]),-.001518996,places=5)
        self.assertTrue(torch.isnan(velocity_clearance_cost(torch.tensor([[float('nan')]]),torch.zeros((1,1,2)))[0]).all())
        with self.assertRaises(ValueError):ClearanceConfig(exponential_denominator_m2=0)


if __name__=='__main__':
    torch.set_num_threads(1)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ClearanceTests))
    (Path(__file__).parent/'tests.json').write_text(json.dumps(dict(tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),success=result.wasSuccessful()),indent=2)+'\n')
    raise SystemExit(not result.wasSuccessful())
