import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from contact_clearance import ContactClearance, LiftTable, lift_penalty, sha
from build_lift import normalize


class Penalty(unittest.TestCase):
    def test_only_lifted_reference_feet_pay(self):
        reference = torch.tensor([[.10, .00], [.10, .02], [.10, .10], [.10, .10]])
        robot = torch.tensor([[.00, .00], [.10, .00], [.15, .05], [.00, .10]])
        force = torch.tensor([[300., 300.], [0., 300.], [0., 0.], [300., 0.]])
        valid = torch.tensor([True, True, True, False])
        loss = lift_penalty(robot, reference, force, valid)
        # 1: magnet (contact + 10 cm short -> (10/5)^2=4); 2: followed, other foot below 3 cm
        # 3: overshoot is free, right foot 5 cm short -> 1; 4: invalid clip.
        np.testing.assert_allclose(loss.numpy(), [5., 0., 1., 0.], atol=1e-6)

    def test_shortfall_is_capped(self):
        loss = lift_penalty(torch.zeros(1, 2), torch.tensor([[.5, 0.]]), torch.zeros(1, 2), torch.tensor([True]))
        self.assertAlmostEqual(float(loss), 4.)


class Table(unittest.TestCase):
    def test_floor_offset(self):
        h = np.zeros((100, 2)); h[:, 0] = .01; h[:, 1] = .01; h[40:60, 0] = .11
        rel, valid = normalize(h)
        self.assertTrue(valid)
        self.assertAlmostEqual(float(rel[50, 0]), .10, places=6)
        self.assertAlmostEqual(float(rel[:, 1].max()), 0., places=6)
        self.assertFalse(normalize(h + .05)[1])

    def test_sample_by_file_and_time(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d); (d/'manifest.json').write_text('{}')
            quest = np.zeros((10, 2), np.float32); quest[5:, 0] = .08
            np.savez_compressed(d/'lift.npz', motion_0000=quest, motion_0001=np.zeros((1, 2), np.float32))
            files = {'q.pkl': {'key': 'motion_0000', 'fps': 10., 'valid': True, 'quest': True},
                     'a.pkl': {'key': 'motion_0001', 'valid': False, 'quest': False}}
            (d/'lift.json').write_text(json.dumps({'dataset_manifest_sha256': sha(d/'manifest.json'),
                                                   'npz_sha256': sha(d/'lift.npz'), 'files': files}))
            table = LiftTable(d/'lift.npz', [str(d/'a.pkl'), str(d/'q.pkl')], d, 'cpu')
            ref, valid = table.sample(torch.tensor([1, 1, 1, 0]), torch.tensor([.2, .6, 1.5, .6]))
            np.testing.assert_allclose(ref[:, 0].numpy(), [0., .08, .08, 0.])
            self.assertEqual(valid.tolist(), [True, True, False, False])
            with self.assertRaises(ValueError):
                LiftTable(d/'lift.npz', [str(d/'q.pkl')], d, 'cpu')

    def test_weight_needs_table(self):
        with self.assertRaises(ValueError):
            ContactClearance(None, {'lift_penalty_weight': .5}, None, None, None)
        with self.assertRaises(ValueError):
            ContactClearance(None, {}, None, None, None, 'lift.npz')


if __name__ == '__main__':
    unittest.main()
