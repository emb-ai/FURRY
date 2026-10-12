import unittest
import numpy as np
from g1_sim.contact_phase import ContactPhaseConfig, Phase, label_reference_contacts, mujoco_reference_sole_points


def stationary(n=50,height=0.):
    t=np.arange(n)*.02
    points=np.zeros((n,2,4,3))
    points[:,:,:,0]=[-.05,-.05,.12,.12]
    points[:,:,:,1]=[.025,-.025,.03,-.03]
    points[:,:,:,2]=height+.005
    return t,points


class ContactPhaseTests(unittest.TestCase):
    def test_stance_requires_history_and_dwell(self):
        t,p=stationary();r=label_reference_contacts(t,p,.005)
        self.assertTrue(np.all(r['state'][:4]==Phase.UNKNOWN))
        self.assertTrue(np.all(r['state'][4:]==Phase.STANCE))
        self.assertTrue(np.isnan(r['desired_contact'][0]).all())
        self.assertTrue(np.all(r['desired_contact'][4:]==1))

    def test_stationary_air_is_swing_not_support(self):
        t,p=stationary(height=.08);r=label_reference_contacts(t,p,.005)
        self.assertTrue(np.all(r['state'][2:]==Phase.SWING))

    def test_moving_ground_is_unknown_not_fabricated_swing(self):
        t,p=stationary();p[:,:,:,0]+=t[:,None,None]*.5
        r=label_reference_contacts(t,p,.005)
        self.assertTrue(np.all(r['state']==Phase.UNKNOWN))

    def test_invalid_gap_reset_and_deep_penetration_clear_history(self):
        t,p=stationary();valid=np.ones((len(t),2),bool);valid[15]=False
        p[30,:,:,2]=-.03
        segment=np.zeros(len(t),int);segment[40:]=1
        r=label_reference_contacts(t,p,.005,valid,segment)
        for i in (15,16,30,31,40,41):self.assertTrue(np.all(r['state'][i]==Phase.UNKNOWN))
        t[20:]+=.10;r=label_reference_contacts(t,p,.005)
        self.assertTrue(np.all(r['state'][20]==Phase.UNKNOWN))

    def test_causal_prefix_and_foot_independence(self):
        t,p=stationary(80);p[30:,0,:,2]+=.08
        full=label_reference_contacts(t,p,.005)
        prefix=label_reference_contacts(t[:35],p[:35],.005)
        for k in full:np.testing.assert_allclose(full[k][:35],prefix[k],equal_nan=True)
        self.assertTrue(np.all(full['state'][40:,0]==Phase.SWING))
        self.assertTrue(np.all(full['state'][40:,1]==Phase.STANCE))

    def test_wrapped_yaw_and_bad_timestamps(self):
        t,p=stationary();yaw=np.zeros((len(t),2));yaw[:25]=np.pi-.001;yaw[25:]=-np.pi+.001
        r=label_reference_contacts(t,p,.005,foot_yaw=yaw)
        self.assertTrue(np.all(r['state'][10:]==Phase.STANCE))
        t[10]=t[9]
        with self.assertRaises(ValueError):label_reference_contacts(t,p,.005)

    def test_fk_reads_floor_colliders_excludes_self_box(self):
        import mujoco as mj
        xml='''<mujoco><worldbody><geom name="floor" type="plane" size="0 0 .1" contype="1" conaffinity="1"/>
        <body name="left_ankle_roll_link" pos="0 0 .04"><freejoint/>
        <geom name="sole_l" type="sphere" pos=".12 0 -.03" size=".01" contype="1" conaffinity="1"/>
        <geom name="self_box" type="box" size="1 1 1" contype="2" conaffinity="2" mass="0"/></body>
        <body name="right_ankle_roll_link" pos="0 1 .04"><freejoint/>
        <geom name="sole_r" type="sphere" pos=".12 0 -.03" size=".01" contype="1" conaffinity="1"/></body></worldbody></mujoco>'''
        model=mj.MjModel.from_xml_string(xml);q=np.tile(model.qpos0,(3,1))
        q[1,3:7]=[np.sqrt(.5),0,np.sqrt(.5),0]
        r=mujoco_reference_sole_points(model,q)
        np.testing.assert_allclose(r['sole_points'][0,:,:,2]-r['radii'],0,atol=1e-12)
        self.assertAlmostEqual(r['sole_points'][1,0,0,2]-r['radii'][0,0],-.09)
        self.assertEqual(r['geometry'][0]['geom_names'],['sole_l'])


if __name__=='__main__':unittest.main()
