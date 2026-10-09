"""Same-articulation geometry, contact coverage and authored render-stream checks."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from g1_sim.controller import Controller
from g1_sim.geometry import ASSETS, BODY, HAND, FOOT_BOX, PROP
from g1_sim.scene import make_model
from g1_sim.visuals import export_visual_meshes, mesh_stream


class GeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.xml = make_model('stand')
        cls.source, _ = make_model('stand', geometry='source')

    def test_articulation_and_dynamics_are_unchanged(self):
        a, b = self.model, self.source
        self.assertEqual(a.nu, 43)
        self.assertEqual([a.joint(i).name for i in range(a.njnt)], [b.joint(i).name for i in range(b.njnt)])
        for attr in ('body_parentid', 'body_pos', 'body_quat', 'body_mass', 'body_inertia', 'body_ipos', 'body_iquat',
                     'jnt_type', 'jnt_axis', 'jnt_pos', 'jnt_range', 'dof_armature', 'dof_damping', 'dof_frictionloss',
                     'actuator_trnid', 'actuator_gainprm', 'actuator_biasprm', 'actuator_gear',
                     'actuator_ctrlrange', 'actuator_forcerange'):
            np.testing.assert_array_equal(getattr(a, attr), getattr(b, attr), err_msg=attr)
        self.assertEqual(a.opt.timestep, b.opt.timestep)
        self.assertEqual(a.opt.solver, b.opt.solver)
        self.assertEqual(a.opt.iterations, b.opt.iterations)

    def test_body_primitives_and_independent_hand_hulls(self):
        m = self.model
        body = np.flatnonzero(np.isin(m.geom_contype, [BODY, FOOT_BOX]))
        hands = np.flatnonzero(m.geom_contype == HAND)
        self.assertEqual(Counter(map(int, m.geom_type[body])), {
            int(mujoco.mjtGeom.mjGEOM_CAPSULE): 15, int(mujoco.mjtGeom.mjGEOM_SPHERE): 10,
            int(mujoco.mjtGeom.mjGEOM_BOX): 2})
        self.assertEqual(len(hands), 16)
        self.assertEqual(len(set(m.geom_bodyid[hands])), 16)
        self.assertTrue(np.all(m.geom_group[np.r_[body, hands]] == 3))
        self.assertTrue(all(m.mesh_facenum[m.geom_dataid[i]] <= 384 for i in hands))
        self.assertLessEqual(sum(m.mesh_facenum[m.geom_dataid[i]] for i in hands), 6144)
        self.assertEqual(sum('_hand_' in m.joint(i).name for i in range(m.njnt)), 14)

    def test_hand_hulls_are_closed_convex_and_small(self):
        description = json.loads((ASSETS/'collision.json').read_text())
        for hand in description['hands']:
            with self.subTest(hand=hand['name']):
                points = np.array(hand['vertices'])
                source = self.source
                mid = source.mesh(hand['name'].removesuffix('_collision')).id
                va, vn = source.mesh_vertadr[mid], source.mesh_vertnum[mid]
                rotation = np.zeros(9)
                mujoco.mju_quat2Mat(rotation, source.mesh_quat[mid])
                source_points = source.mesh_vert[va:va+vn] @ rotation.reshape(3, 3).T + source.mesh_pos[mid]
                self.assertLessEqual(hand['max_envelope_error_m'], hand['tolerance_m'])
                faces = np.array(hand['faces'])
                edges = Counter(tuple(sorted(e)) for f in faces for e in [(f[0], f[1]), (f[1], f[2]), (f[2], f[0])])
                self.assertTrue(all(n == 2 for n in edges.values()))
                for face in faces:
                    a, b, c = points[face]
                    n = np.cross(b-a, c-a)
                    self.assertGreater(np.linalg.norm(n), 1e-12)
                    self.assertLess(np.max((points-a) @ n), 1e-10)
                    self.assertLessEqual(np.max((source_points-a) @ (n/np.linalg.norm(n))), 1e-7)

    def test_every_palm_and_finger_link_contacts_a_prop(self):
        root = ET.fromstring(self.xml)
        probe = ET.SubElement(root.findall('worldbody')[-1], 'body', name='probe', pos='3 0 1')
        ET.SubElement(probe, 'freejoint', name='probe_free')
        ET.SubElement(probe, 'geom', name='probe', type='sphere', size='.002', mass='.001',
                      contype=str(PROP), conaffinity='31')
        m = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
        d = mujoco.MjData(m)
        probe_id, qadr = m.geom('probe').id, m.joint('probe_free').qposadr[0]
        for i in np.flatnonzero(m.geom_contype == HAND):
            with self.subTest(geom=m.geom(int(i)).name):
                mujoco.mj_resetData(m, d)
                mujoco.mj_forward(m, d)
                mid = m.geom_dataid[i]
                va, fa = m.mesh_vertadr[mid], m.mesh_faceadr[mid]
                points = m.mesh_vert[va + m.mesh_face[fa]].astype(float)
                normal = np.cross(points[1]-points[0], points[2]-points[0])
                normal /= np.linalg.norm(normal)
                point = points.mean(axis=0) + normal*.001
                d.qpos[qadr:qadr+3] = d.geom_xpos[i] + d.geom_xmat[i].reshape(3, 3) @ point
                mujoco.mj_forward(m, d)
                pairs = {frozenset((c.geom1, c.geom2)) for c in d.contact}
                self.assertIn(frozenset((int(i), probe_id)), pairs)

    def test_four_source_sole_spheres_per_foot_contact_floor(self):
        m, d = self.model, mujoco.MjData(self.model)
        Controller(m).initialize(d)
        d.qpos[2] -= .07
        mujoco.mj_forward(m, d)
        floor = m.geom('floor').id
        touching = {c.geom2 if c.geom1 == floor else c.geom1 for c in d.contact if floor in (c.geom1, c.geom2)}
        for side in ('left', 'right'):
            ankle = m.body(f'{side}_ankle_roll_link').id
            source_ankle = self.source.body(f'{side}_ankle_roll_link').id
            source_spheres = [i for i in range(self.source.ngeom)
                              if self.source.geom_bodyid[i] == source_ankle
                              and self.source.geom_type[i] == mujoco.mjtGeom.mjGEOM_SPHERE]
            spheres = [i for i in range(m.ngeom) if m.geom_bodyid[i] == ankle
                       and m.geom_type[i] == mujoco.mjtGeom.mjGEOM_SPHERE]
            self.assertEqual(len(spheres), 4)
            np.testing.assert_array_equal(m.geom_pos[spheres], self.source.geom_pos[source_spheres])
            np.testing.assert_array_equal(m.geom_size[spheres], self.source.geom_size[source_spheres])
            self.assertTrue(set(spheres).issubset(touching), side)
            self.assertFalse(any(m.geom_bodyid[i] == ankle and
                                 m.geom_type[i] == mujoco.mjtGeom.mjGEOM_CAPSULE
                                 for i in range(m.ngeom)))
        self.assertFalse(any(m.geom_contype[i] == FOOT_BOX for i in touching))

    def test_normals_and_visual_budget_survive_binary_export(self):
        m = self.model
        visible = {int(m.geom_dataid[i]) for i in range(m.ngeom)
                   if m.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH and m.geom_contype[i] == 0}
        self.assertEqual(len(visible), 49)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'visual_meshes.bin'
            counts = export_visual_meshes(m, path, visible)
            self.assertLessEqual(sum(counts), 90000)
            with path.open('rb') as stream:
                self.assertEqual(struct.unpack('<I', stream.read(4))[0], m.nmesh)
                for i, count in enumerate(counts):
                    self.assertEqual(struct.unpack('<I', stream.read(4))[0], count*3)
                    packed = np.frombuffer(stream.read(count*3*6*4), dtype='<f4').reshape(count, 3, 6)
                    if i in visible:
                        np.testing.assert_array_equal(packed, mesh_stream(m, i))
                        np.testing.assert_allclose(np.linalg.norm(packed[:, :, 3:], axis=2), 1, atol=1e-5)
                    else:
                        self.assertEqual(count, 0)
                self.assertEqual(stream.read(), b'')
        torso = mesh_stream(m, m.mesh('torso_link').id)
        self.assertGreater(np.mean(np.linalg.norm(torso[:, 0, 3:]-torso[:, 1, 3:], axis=1) > .001), .25)

    def test_versioned_assets_match_manifest(self):
        manifest = json.loads((ASSETS/'visual/manifest.json').read_text())
        self.assertEqual(len(manifest['meshes']), 49)
        for name, record in manifest['meshes'].items():
            with self.subTest(mesh=name):
                path = ASSETS/'visual'/f'{name}.obj'
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), record['sha256'])

    def test_source_fixed_hands_and_collision_only_variants_compile(self):
        fixed, _ = make_model('stand', hands=False)
        self.assertEqual(fixed.nu, 29)
        collisions, _ = make_model('stand', geometry='collision_only')
        self.assertEqual(collisions.nu, 43)
        self.assertEqual(collisions.ngeom, self.model.ngeom)
        with self.assertRaises(ValueError):
            make_model(geometry='typo')


if __name__ == '__main__':
    unittest.main()
