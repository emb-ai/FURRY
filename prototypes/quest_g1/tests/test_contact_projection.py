import unittest
import mujoco as mj
import numpy as np
from g1_sim.scene import make_model
from g1_sim.controller import DEFAULT, JOINTS
from g1_sim.contact_projection import SingleSupportProjection


class ContactProjectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model,_=make_model('stand')
        cls.pose=cls.model.qpos0.copy()
        cls.pose[[cls.model.joint(n).qposadr[0] for n in JOINTS]]=DEFAULT
        projector=SingleSupportProjection(cls.model)
        projector._fk(cls.pose)
        cls.pose[2]-=projector._floor_clearance().min()

    def test_feasible_unknown_reference_unchanged(self):
        p=SingleSupportProjection(self.model)
        result=p.project(self.pose,[-1,-1])
        self.assertTrue(result['feasible'])
        self.assertEqual(result['anchor_side'],-1)
        np.testing.assert_allclose(result['qpos'],self.pose,atol=1e-10)

    def test_floor_and_limits_solved_without_changing_other_reference_coordinates(self):
        p=SingleSupportProjection(self.model);pose=self.pose.copy();pose[2]-=.006
        result=p.project(pose,[-1,-1])
        self.assertTrue(result['feasible'])
        self.assertGreaterEqual(result['min_clearance_after_m'],-1e-6)
        others=np.setdiff1d(np.arange(self.model.nq),p.qadr)
        np.testing.assert_array_equal(result['qpos'][others],pose[others])
        joints=result['qpos'][p.qadr[1:]]
        self.assertTrue(np.all(joints>=p.limits[:,0]-1e-9))
        self.assertTrue(np.all(joints<=p.limits[:,1]+1e-9))

    def test_only_one_anchor_and_no_root_translation_lock(self):
        p=SingleSupportProjection(self.model)
        first=p.project(self.pose,[1,1],[.9,.8])
        pose=self.pose.copy();pose[0]+=.015
        second=p.project(pose,[1,1],[.9,.8])
        self.assertTrue(second['feasible'])
        self.assertEqual(second['anchor_side'],0)
        np.testing.assert_array_equal(second['qpos'][:2],pose[:2])
        np.testing.assert_array_equal(first['anchor_xy'],second['anchor_xy'])
        self.assertLess(second['anchor_error_m'],1e-6)
        self.assertGreater(np.max(np.abs(second['joint_delta_rad'])),.001)

    def test_unknown_and_reference_drift_release(self):
        p=SingleSupportProjection(self.model)
        p.project(self.pose,[1,-1])
        r=p.project(self.pose,[-1,-1]);self.assertEqual(r['anchor_side'],-1)
        self.assertEqual(r['release'],'reference_phase')
        p.project(self.pose,[1,-1]);pose=self.pose.copy();pose[0]+=.08
        r=p.project(pose,[1,-1]);self.assertEqual(r['anchor_side'],-1)
        self.assertEqual(r['release'],'reference_drift')

    def test_swing_is_never_anchored_and_reset_drops_anchor(self):
        p=SingleSupportProjection(self.model)
        r=p.project(self.pose,[0,0]);self.assertEqual(r['anchor_side'],-1)
        p.project(self.pose,[1,-1]);pose=self.pose.copy();pose[0]+=.02
        r=p.project(pose,[1,-1],reset=True)
        self.assertLess(np.max(np.abs(r['joint_delta_rad'])),1e-9)

    def test_position_jacobian_matches_finite_difference(self):
        p=SingleSupportProjection(self.model);q=self.pose.copy();p._fk(q)
        _,g,body,_=p.pads[0]
        analytic=p._point_jacobian(p.data.geom_xpos[g],body)
        numerical=np.zeros_like(analytic)
        for k,adr in enumerate(p.qadr):
            a=q.copy();b=q.copy();a[adr]+=1e-6;b[adr]-=1e-6
            p._fk(a);plus=p.data.geom_xpos[g].copy()
            p._fk(b);minus=p.data.geom_xpos[g].copy()
            numerical[:,k]=(plus-minus)/2e-6
        np.testing.assert_allclose(analytic,numerical,atol=1e-9,rtol=1e-8)


if __name__=='__main__':unittest.main()
