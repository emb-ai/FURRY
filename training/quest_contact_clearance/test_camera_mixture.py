import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from camera_mixture import bank, combine
from native_command_bank import NativeCommandBank


def h(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def timeline(value, seconds=.4):
    t = np.arange(0, seconds + 1e-9, .1)
    return np.column_stack((t, np.full((len(t), 35), value)))


class CameraMixtureTests(unittest.TestCase):
    def fixture(self, root):
        # Mixture as author_dataset.py writes it: Quest PKL hash in sha256,
        # original NPZ hash in npz_sha256.
        clips = [dict(file='quest/train/m.pkl', split='train', source='quest', sha256='pkl-m', npz_sha256='npz-m', seconds=.4, stage=1),
                 dict(file='quest/train/c.pkl', split='train', source='quest', sha256='pkl-c', npz_sha256='npz-c', seconds=.4, stage=101,
                      input_source='quest-camera'),
                 dict(file='author/a.pkl', split='train', source='twist2_author', sha256='pkl-a', seconds=.5),
                 dict(file='quest/validation/v.pkl', split='validation', source='quest', sha256='pkl-v', npz_sha256='npz-v', seconds=.4, stage=103)]
        (root/'mix').mkdir()
        (root/'mix/manifest.json').write_text(json.dumps({'schema': 'twist2-quest-mixture-v2', 'clips': clips}))
        np.savez(root/'native.npz', a=timeline(1.))
        (root/'native.json').write_text(json.dumps(dict(schema='quest-native-command-bank-v1', npz_sha256=h(root/'native.npz'),
                                                        files={'quest/train/m.pkl': dict(key='a', target_sha256='pkl-m')})))
        np.savez(root/'camera.npz', b=timeline(2.), v=timeline(3.))
        (root/'camera.json').write_text(json.dumps(dict(schema='quest-camera-command-bank-v1', npz_sha256=h(root/'camera.npz'),
                                                        files={'train/c.npz': dict(key='b', target_sha256='npz-c')})))
        return [str(root/'mix'/c['file']) for c in clips[:3]]

    def test_merged_bank_loads_and_keeps_each_timeline(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); files = self.fixture(root)
            meta = bank(root/'mix', [root/'native.npz', root/'camera.npz'], root/'out.npz')
            self.assertEqual(meta['seconds_by_input'], {'quest-camera': .4, 'quest-meta': .4})
            loaded = NativeCommandBank(root/'out.npz', files, root/'mix', 'cpu')
            got = loaded.sample(torch.tensor([0, 1, 2]), torch.full((3,), .2, dtype=torch.float64), torch.full((3, 35), -1.))
            self.assertEqual(got[:, 6].tolist(), [1., 2., -1.])

    def test_native_bank_pairs_through_its_own_mixture(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); files = self.fixture(root)
            # Old mixture: same NPZ, different PKL bytes.
            (root/'old').mkdir()
            (root/'old/manifest.json').write_text(json.dumps({'clips': [dict(file='quest/train/m.pkl', sha256='old-pkl-m', npz_sha256='npz-m')]}))
            meta = json.loads((root/'native.json').read_text())
            meta['files']['quest/train/m.pkl']['target_sha256'] = 'old-pkl-m'
            meta['dataset_manifest_sha256'] = h(root/'old/manifest.json')
            (root/'native.json').write_text(json.dumps(meta))
            with self.assertRaises(ValueError):
                bank(root/'mix', [root/'native.npz', root/'camera.npz'], root/'out.npz')
            bank(root/'mix', [f"{root/'native.npz'}:{root/'old'}", root/'camera.npz'], root/'out.npz')
            loaded = NativeCommandBank(root/'out.npz', files, root/'mix', 'cpu')
            got = loaded.sample(torch.tensor([0]), torch.tensor([.2], dtype=torch.float64), torch.zeros(1, 35))
            self.assertEqual(got[0, 6].item(), 1.)

    def test_missing_or_duplicate_coverage_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); self.fixture(root)
            with self.assertRaises(ValueError):
                bank(root/'mix', [root/'camera.npz'], root/'out.npz')
            meta = json.loads((root/'camera.json').read_text())
            meta['files']['train/m.npz'] = dict(key='v', target_sha256='npz-m')
            (root/'camera.json').write_text(json.dumps(meta))
            with self.assertRaises(ValueError):
                bank(root/'mix', [root/'native.npz', root/'camera.npz'], root/'out2.npz')

    def test_retarget_moves_native_timeline_to_rebuilt_clip(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); files = self.fixture(root)
            for name, value in (('old', 0.), ('new', .01)):
                (root/name/'train').mkdir(parents=True)
                np.savez(root/name/'train/m.npz', qpos=np.full((41, 36), value))
                clip = dict(id='m', file='train/m.npz', split='train', stage=1, seconds=.4, sha256=h(root/name/'train/m.npz'))
                (root/name/'manifest.json').write_text(json.dumps(dict(clips=[clip])))
            # The native bank pairs with the old target; the mixture holds the rebuilt one.
            clips = json.loads((root/'mix/manifest.json').read_text())
            clips['clips'][0]['npz_sha256'] = h(root/'new/train/m.npz')
            (root/'mix/manifest.json').write_text(json.dumps(clips))
            meta = json.loads((root/'native.json').read_text())
            meta['files']['quest/train/m.pkl']['target_sha256'] = h(root/'old/train/m.npz')
            (root/'native.json').write_text(json.dumps(meta))
            with self.assertRaises(ValueError):
                bank(root/'mix', [root/'native.npz', root/'camera.npz'], root/'out.npz')
            out = bank(root/'mix', [root/'native.npz', root/'camera.npz'], root/'out.npz', (root/'old', root/'new'))
            self.assertEqual(out['retarget']['clips'], 1)
            loaded = NativeCommandBank(root/'out.npz', files, root/'mix', 'cpu')
            got = loaded.sample(torch.tensor([0]), torch.tensor([.2], dtype=torch.float64), torch.zeros(1, 35))
            self.assertEqual(got[0, 6].item(), 1.)
            np.savez(root/'new/train/m.npz', qpos=np.zeros((40, 36)))
            clips = json.loads((root/'new/manifest.json').read_text())
            clips['clips'][0]['sha256'] = h(root/'new/train/m.npz')
            (root/'new/manifest.json').write_text(json.dumps(clips))
            with self.assertRaises(ValueError):
                bank(root/'mix', [root/'native.npz', root/'camera.npz'], root/'out2.npz', (root/'old', root/'new'))

    def test_combine_rejects_shared_stage(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for name in ('a', 'b'):
                (root/name/'train').mkdir(parents=True)
                np.savez(root/name/'train'/f'{name}.npz', x=np.zeros(2))
                clip = dict(id=name, file=f'train/{name}.npz', split='train', stage=1, seconds=1., sha256=h(root/name/'train'/f'{name}.npz'))
                (root/name/'manifest.json').write_text(json.dumps(dict(schema='quest-g1-motion-v1', validation_stages=[], clips=[clip])))
            with self.assertRaises(ValueError):
                combine(root/'out', [root/'a', root/'b'])


if __name__ == '__main__':
    unittest.main()
