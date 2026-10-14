import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
import torch
from native_command_bank import NativeCommandBank, QuestActorHistory, install
from build_command_bank import mirror_command


class NativeCommandTests(unittest.TestCase):
    def fixture(self, root):
        clips = [dict(file='quest/train/q.pkl', split='train', source='quest', sha256='target-q', seconds=.4),
                 dict(file='author/a.pkl', split='train', source='twist2_author', sha256='target-a', seconds=.5),
                 dict(file='quest/validation/v.pkl', split='validation', source='quest', sha256='target-v', seconds=.5)]
        (root/'manifest.json').write_text(json.dumps({'clips': clips}))
        arr = np.zeros((3, 36)); arr[:, 0] = [0, .13, .4]; arr[:, 1:] = np.arange(35)+np.arange(3)[:, None]*100
        path = root/'commands.npz'; np.savez(path, q=arr)
        h = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
        meta = dict(schema='quest-native-command-bank-v1', npz_sha256=h(path),
                    dataset_manifest_sha256=h(root/'manifest.json'),
                    files={'quest/train/q.pkl': dict(key='q', target_sha256='target-q')})
        path.with_suffix('.json').write_text(json.dumps(meta))
        return path, [str(root/c['file']) for c in clips[:2]]

    def test_causal_boundaries_fallback_and_terminal_hold(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); path, files = self.fixture(root)
            bank = NativeCommandBank(path, files, root, 'cpu')
            ids = torch.tensor([0, 0, 0, 0, 0, 1])
            times = torch.tensor([-.1, .129999, .13, .4, 5., .2], dtype=torch.float64)
            fallback = torch.full((6, 35), -23.)
            got = bank.sample(ids, times, fallback)
            self.assertEqual(got[:, 6].tolist(), [6., 6., 106., 206., 206., -23.])
            self.assertTrue(torch.equal(got[4, [0, 1, 5]], torch.zeros(3)))
            self.assertTrue(torch.equal(got[5], fallback[5]))
            self.assertTrue(torch.equal(fallback, torch.full((6, 35), -23.)))

    def test_reject_validation_in_loader_or_bank(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); path, files = self.fixture(root)
            with self.assertRaises(ValueError):
                NativeCommandBank(path, files+[str(root/'quest/validation/v.pkl')], root, 'cpu')
            meta = json.loads(path.with_suffix('.json').read_text())
            meta['files']['quest/validation/v.pkl'] = meta['files']['quest/train/q.pkl']
            path.with_suffix('.json').write_text(json.dumps(meta))
            with self.assertRaises(ValueError):
                NativeCommandBank(path, files, root, 'cpu')

    def test_native_zero_reset_history_and_current_tail(self):
        state = QuestActorHistory(2, 'cpu')
        a = torch.arange(254).reshape(2, 127).float()
        first = state.observe(a)
        self.assertEqual(first.shape, (2, 1432))
        self.assertEqual(first[:, 127:1397].count_nonzero(), 0)
        self.assertTrue(torch.equal(first[:, 1397:], a[:, :35]))
        state.reset(torch.tensor([0]))
        second = state.observe(a+1000)
        self.assertEqual(second[0, 127:1397].count_nonzero(), 0)
        self.assertTrue(torch.equal(second[1, 127:1397].reshape(10, 127)[-1], a[1]))
        self.assertEqual(second[1, 127:127+9*127].count_nonzero(), 0)
        # Returned observation owns its history; a later tick cannot overwrite it.
        self.assertEqual(first[:, 127:1397].count_nonzero(), 0)

    def test_adapter_does_not_change_critic_or_reward_references(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); path, files = self.fixture(root)
            env = SimpleNamespace(dt=.02, num_obs=1432, obs_type='student_future', num_envs=2,
                                  device='cpu', _motion_lib=SimpleNamespace(_motion_files=files),
                                  _motion_ids=torch.tensor([0, 1]), obs_history_buf=torch.zeros(2, 10, 127))
            # Freshly constructed source env: no physics step yet, so no roll/pitch and a zero actor buffer.
            env.obs_buf = torch.zeros(2, 1432)
            env.compute_observations = lambda: self.fail('source observations need a physics step')
            env.reset_idx = lambda ids: None
            install(env, path, root)
            self.assertEqual(env.obs_buf.count_nonzero(), 0)
            env.roll = torch.zeros(2)
            clean = torch.arange(2864).reshape(2, 1432).float()
            critic = torch.randn(2, 500)
            env.ref_dof_pos = torch.randn(2, 29); ref_before = env.ref_dof_pos.clone()
            def obs():
                env.obs_buf = clean.clone(); env.privileged_obs_buf = critic.clone()
            env.compute_observations = obs
            env.reset_idx = lambda ids: None
            env._get_motion_times = lambda: torch.tensor([.15, .15])
            state = install(env, path, root)
            self.assertTrue(torch.equal(env.privileged_obs_buf, critic))
            self.assertTrue(torch.equal(env.ref_dof_pos, ref_before))
            self.assertTrue(torch.equal(env.obs_buf[:, 35:127], clean[:, 35:127]))
            self.assertTrue(torch.equal(env.obs_buf[1, :35], clean[1, :35]))
            self.assertTrue(torch.equal(env.obs_buf[:, 1397:], env.obs_buf[:, :35]))
            env.reset_idx(torch.tensor([0])); env.compute_observations()
            self.assertEqual(env.obs_buf[0, 127:1397].count_nonzero(), 0)
            self.assertGreater(env.obs_buf[1, 127:1397].count_nonzero(), 0)

    def test_reflection_is_involutive_and_preserves_pitch(self):
        names = ['left_hip_roll_joint', 'left_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_pitch_joint']
        cmd = np.arange(20.).reshape(2, 10)
        reflected = mirror_command(cmd, names)
        np.testing.assert_array_equal(mirror_command(reflected, names), cmd)
        np.testing.assert_array_equal(reflected[:, 6], -cmd[:, 8])
        np.testing.assert_array_equal(reflected[:, 7], cmd[:, 9])


if __name__ == '__main__':
    unittest.main()
