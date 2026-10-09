import copy
import io
import pickle
import unittest
import numpy as np
from author_dataset import ArrayUnpickler, guard_quest_split, quest_weights, validate_motion, world_offsets_to_root
from select_author_motions import select


class AuthorDatasetTests(unittest.TestCase):
    def selection_fixture(self):
        paths = ['OMOMO_g1_GMR/a.pkl', 'OMOMO_g1_GMR/b.pkl', 'AMASS_g1_GMR8/c.pkl', 'twist1_to_twist2/d.pkl', 'v1_v2_v3_g1/pico.pkl']
        manifest = ''.join(f'- file: {p}\n  weight: {10 if "pico" in p else 1}\n' for p in paths)
        index = {'bytes': 100, 'entries': [{'name': 'TWIST2_full/' + p} for p in paths + ['AMASS_g1_GMR8/unknown.pkl']]}
        return manifest, index

    def test_train_membership_and_mass(self):
        m, i = self.selection_fixture()
        r = select(m, i, count=3)
        self.assertEqual(len(r['clips']), 3)
        self.assertEqual(sum(c['sampling_weight'] for c in r['clips']), 4)
        self.assertEqual(r['quest_replacement_weight'], 10)
        self.assertEqual(r['archive_pkls_not_in_author_manifest'], 1)
        for c in r['clips']:
            self.assertEqual((c['author_split'], c['local_split']), ('train', 'train'))
            self.assertNotIn('pico', c['author_path'])
            self.assertNotIn('unknown', c['author_path'])
        self.assertEqual(r, select(m, i, count=3))

    def test_missing_manifest_member_fails(self):
        m, i = self.selection_fixture()
        i['entries'] = i['entries'][1:]
        with self.assertRaises(ValueError): select(m, i, count=3)

    def test_oversized_subset_fails(self):
        m, i = self.selection_fixture()
        with self.assertRaises(ValueError): select(m, i, count=5)

    def fixture(self):
        return [{'id':'a','split':'train','stage':1,'sha256':'a'}, {'id':'b','split':'validation','stage':2,'sha256':'b'}, {'id':'a-mirror','split':'train','stage':1,'sha256':'c','augmentation':'sagittal_reflection','parent_id':'a'}]

    def test_mirrors_share_parent_mass(self):
        c = self.fixture() + [{'id':'d','split':'train','stage':3,'sha256':'d'}]
        self.assertEqual(quest_weights(c, 100), {'a':25,'a-mirror':25,'d':50})
        guard_quest_split(c, [2])

    def test_stage_leak_fails(self):
        c = self.fixture(); c[1]['split'] = 'train'
        with self.assertRaises(ValueError): guard_quest_split(c, [2])

    def test_duplicate_content_across_splits_fails(self):
        c = self.fixture(); c[1]['sha256'] = 'a'
        with self.assertRaises(ValueError): guard_quest_split(c, [2])

    def test_validation_parent_cannot_augment_train(self):
        c = self.fixture(); c[2]['parent_id'] = 'b'
        with self.assertRaises(ValueError): quest_weights(c, 100)

    def test_unpickler_rejects_code(self):
        with self.assertRaises(pickle.UnpicklingError): ArrayUnpickler(io.BytesIO(pickle.dumps(eval))).load()

    def test_rotated_pelvis_coordinates(self):
        from scipy.spatial.transform import Rotation
        r = Rotation.from_euler('xyz', [[.2,-.3,1.4],[-.1,.4,-2.]])
        local = np.array([[[.2,.3,-.5],[-.2,.1,.4]]]*2)
        world = np.einsum('nij,nkj->nki', r.as_matrix(), local)
        np.testing.assert_allclose(world_offsets_to_root(world, r.as_quat()), local, atol=1e-14)

    def test_quaternion_convention_and_invalid_arrays(self):
        x = {'fps':100, 'root_pos':np.zeros((2,3)), 'root_rot':np.array([[0,0,0,1.]]*2), 'dof_pos':np.zeros((2,29)), 'local_body_pos':np.zeros((2,1,3)), 'link_body_list':['pelvis']}
        self.assertEqual(validate_motion(x)[0], .01)
        x['root_rot'] *= .95
        with self.assertRaises(ValueError): validate_motion(x)
        validate_motion(x, strict_quaternions=False)
        x['dof_pos'][0,0] = np.nan
        with self.assertRaises(ValueError): validate_motion(x, strict_quaternions=False)


if __name__ == '__main__': unittest.main()
