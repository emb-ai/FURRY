"""Physical invariants for the Quest collision simplification."""
import unittest
import mujoco
import numpy as np
from g1_sim.scene import make_model
from g1_sim.collisions import primitive_collisions
from g1_sim.controller import Controller

class CollisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original, xml=make_model('lab')
        xml,cls.report=primitive_collisions(xml)
        cls.model=mujoco.MjModel.from_xml_string(xml)

    def test_no_mesh_collisions_and_unchanged_dynamics(self):
        a,b=self.original,self.model
        active=(b.geom_contype!=0)|(b.geom_conaffinity!=0)
        self.assertEqual(self.report['replaced_meshes'],38)
        self.assertFalse(np.any(b.geom_type[active]==mujoco.mjtGeom.mjGEOM_MESH))
        for field in ('body_mass','body_inertia','body_ipos','body_iquat','dof_armature',
                      'dof_damping','jnt_range','actuator_ctrlrange','geom_contype','geom_conaffinity','geom_friction'):
            np.testing.assert_allclose(getattr(a,field),getattr(b,field),atol=1e-12,rtol=0)

    def test_foot_ground_pads_unchanged_and_no_new_rest_intersections(self):
        a,b=self.original,self.model
        for side in ('left','right'):
            foot=a.body(side+'_ankle_roll_link').id
            ids=np.flatnonzero((a.geom_bodyid==foot)&(a.geom_contype!=0))
            self.assertEqual(len(ids),4)
            for field in ('geom_type','geom_size','geom_pos','geom_quat','geom_solref','geom_solimp'):
                np.testing.assert_array_equal(getattr(a,field)[ids],getattr(b,field)[ids])
        data=mujoco.MjData(b);mujoco.mj_forward(b,data)
        pelvis=b.body('pelvis').id
        self.assertFalse(any(b.geom_bodyid[c.geom1]>=pelvis and b.geom_bodyid[c.geom2]>=pelvis and c.dist<0 for c in data.contact))

    def test_balanced_stand_and_reset_with_new_collisions(self):
        m=self.model;d=mujoco.MjData(m);controller=Controller(m);controller.initialize(d);initial=d.qpos.copy();min_z=1.
        for _ in range(10000):
            controller.step(d);min_z=min(min_z,d.qpos[2])
        self.assertGreater(min_z,.70)
        self.assertLess(np.linalg.norm(d.qpos[:2]),.15)
        self.assertEqual(sum(w.number for w in d.warning),0)
        controller.initialize(d);np.testing.assert_array_equal(d.qpos,initial)
