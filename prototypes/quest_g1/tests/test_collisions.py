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

    def test_compact_housings_and_unchanged_dynamics(self):
        a,b=self.original,self.model
        active=(b.geom_contype!=0)|(b.geom_conaffinity!=0)
        self.assertEqual(self.report['replaced_meshes'],38)
        self.assertEqual(np.count_nonzero(b.geom_type[active]==mujoco.mjtGeom.mjGEOM_MESH), 6)
        hulls = [c for c in self.report['changes'] if c['shape']=='compact_convex_hull']
        self.assertEqual(len(hulls), 6)
        for hull in hulls:
            self.assertLessEqual(hull['max_plane_error_m'], .00015)
            self.assertGreater(hull['volume_ratio'], .995)
            self.assertLess(hull['vertices'], 500)
        for field in ('body_mass','body_inertia','body_ipos','body_iquat','dof_armature',
                      'dof_damping','jnt_range','actuator_ctrlrange','geom_contype','geom_conaffinity','geom_friction'):
            np.testing.assert_allclose(getattr(a,field),getattr(b,field),atol=1e-12,rtol=0)

    def test_compiled_hull_frames_and_support_surfaces(self):
        # Check the world-space surfaces after MuJoCo's mesh recentering and
        # principal-axis rotation, independently of the fitting algorithm.
        a, b = self.original, self.model
        da, db = mujoco.MjData(a), mujoco.MjData(b)
        mujoco.mj_forward(a, da); mujoco.mj_forward(b, db)
        directions = np.random.default_rng(7).normal(size=(512, 3))
        directions /= np.linalg.norm(directions, axis=1)[:, None]
        for change in self.report['changes']:
            if change['shape'] != 'compact_convex_hull':
                continue
            gid = b.geom(change['geom']).id
            supports = []
            for model, data in ((a, da), (b, db)):
                mid = model.geom_dataid[gid]
                adr, count = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
                v = model.mesh_vert[adr:adr+count].astype(float)
                world = np.einsum('ij,kj->ik', v, data.geom_xmat[gid].reshape(3, 3)) + data.geom_xpos[gid]
                supports.append(np.max(np.einsum('ij,kj->ik', world, directions), axis=0))
            gap = supports[0]-supports[1]
            self.assertGreaterEqual(float(gap.min()), -1e-7)
            self.assertLess(float(gap.max()), .0003)

    def test_wrist_contacts_match_source_through_legal_rotations(self):
        # The old bounding boxes collide hundreds of times in these poses.
        # Compare actual compiled collision distances to catch frame mistakes.
        a, b = self.original, self.model
        da, db = mujoco.MjData(a), mujoco.MjData(b)
        rng = np.random.default_rng(512)
        for _ in range(128):
            da.qpos[:] = a.qpos0
            for side in ('left', 'right'):
                for axis in ('pitch', 'yaw'):
                    joint = a.joint(f'{side}_wrist_{axis}_joint')
                    da.qpos[joint.qposadr[0]] = rng.uniform(*joint.range)
            db.qpos[:] = da.qpos
            mujoco.mj_forward(a, da); mujoco.mj_forward(b, db)
            for side in ('left', 'right'):
                ids = []
                for axis in ('roll', 'yaw'):
                    body = a.body(f'{side}_wrist_{axis}_link').id
                    ids.append(next(i for i in range(a.ngeom)
                                    if a.geom_bodyid[i] == body and a.geom_contype[i]
                                    and a.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH
                                    and a.mesh(a.geom_dataid[i]).name == a.body(body).name))
                source = mujoco.mj_geomDistance(a, da, *ids, 1., None)
                fixed = mujoco.mj_geomDistance(b, db, *ids, 1., None)
                if source >= -.0002:
                    self.assertGreaterEqual(fixed, -.0002)

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
