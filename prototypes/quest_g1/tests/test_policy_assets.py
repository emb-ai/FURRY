"""Package real author actors and check candidate/replay provenance."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from g1_sim.controller import UPSTREAM
from scripts.policy_assets import export_policies


class PolicyAssetsTests(unittest.TestCase):
    def test_default_catalog_contains_published_candidates(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict('os.environ', {}, clear=True):
            root = Path(directory)
            for scene in ('lab', 'stand', 'cup', 'push_t'):
                name = 'recording_metadata.json' if scene == 'lab' else f'recording_metadata-{scene}.json'
                (root / name).write_text(json.dumps({'sha256': {}, 'scene': scene}))
            catalog = export_policies(root, UPSTREAM)
            self.assertEqual([row['id'] for row in catalog],
                             ['twist2-20k', 'twist2-25k', 'quest-111260-500', 'quest-111260-1000',
                              'quest-111586-200', 'quest-111586-300', 'quest-111586-500', 'homie-v2'])
            artifacts = {}
            for run in ('quest-111260', 'quest-111586'):
                provenance = json.loads((Path(__file__).resolve().parents[1] /
                                         f'policies/{run}/provenance.json').read_text())
                artifacts.update({item['id']: item for item in provenance['artifacts']})
            for row in catalog[2:7]:
                artifact = artifacts[row['id']]
                self.assertEqual(row['sha256'], artifact['sha256'])
                self.assertEqual(hashlib.sha256((root / row['file']).read_bytes()).hexdigest(), artifact['sha256'])
                expected_hz = 50 if row['id'].startswith('quest-111586-') else 100
                self.assertEqual(row['policy_hz'], 50)
                self.assertEqual(row['legacy_policy_hz'], expected_hz)
                metadata = json.loads((root / f'recording_metadata-stand-policy-{row["id"]}.json').read_text())
                self.assertEqual(metadata['policy_hz'], 50)
                self.assertEqual(metadata['legacy_policy_hz'], expected_hz)
            homie = catalog[-1]
            self.assertEqual(homie['backend'], 'groot_decoupled_v1')
            self.assertEqual(homie['policy_hz'], 50)
            self.assertEqual(homie['legacy_policy_hz'], 50)
            metadata = json.loads((root / 'recording_metadata-stand-policy-homie-v2.json').read_text())
            self.assertEqual(metadata['policy_backend'], 'groot_decoupled_v1')
            for prefix in ('', 'secondary_'):
                filename, digest = homie[prefix+'file'], homie[prefix+'sha256']
                self.assertEqual(hashlib.sha256((root / filename).read_bytes()).hexdigest(), digest)
                self.assertEqual(metadata['policy_'+prefix+'file'], filename)
                self.assertEqual(metadata['policy_'+prefix+'sha256'], digest)
                self.assertEqual(metadata['sha256'][filename], digest)
            # Removing this mode also removes both models from the APK assets.
            config = root/'empty.json'
            config.write_text('[]')
            export_policies(root, UPSTREAM, config)
            self.assertFalse((root/homie['file']).exists())
            self.assertFalse((root/homie['secondary_file']).exists())

    def test_decoupled_backend_requires_correct_pair_and_hashes(self):
        models = Path(__file__).resolve().parents[1]/'policies/homie-v2'
        balance = models/'GR00T-WholeBodyControl-Balance.onnx'
        walk = models/'GR00T-WholeBodyControl-Walk.onnx'
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);config=root/'config.json'
            row=dict(id='pair', title='Pair', note='Experimental', backend='groot_decoupled_v1', path=str(balance))
            config.write_text(json.dumps([row]))
            with self.assertRaisesRegex(ValueError, 'secondary model'):
                export_policies(root, UPSTREAM, config)
            row.update(secondary_path=str(walk),secondary_sha256='0'*64)
            config.write_text(json.dumps([row]))
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                export_policies(root, UPSTREAM, config)
            row.pop('secondary_path');row.pop('secondary_sha256');row['backend']='twist2'
            config.write_text(json.dumps([row]))
            with self.assertRaisesRegex(ValueError, 'Incompatible Quest policy'):
                export_policies(root, UPSTREAM, config)

    def test_unfetched_lfs_pointer_explains_how_to_download(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'candidate.onnx'
            source.write_text('version https://git-lfs.github.com/spec/v1\noid sha256:' + '0' * 64 + '\nsize 1\n')
            config = root / 'config.json'
            config.write_text(json.dumps([dict(id='candidate', path='candidate.onnx',
                                               title='Candidate', note='Experimental')]))
            with self.assertRaisesRegex(ValueError, 'git lfs pull'):
                export_policies(root, UPSTREAM, config)

    def test_candidate_metadata_and_stale_asset_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for scene in ('lab', 'stand', 'cup', 'push_t'):
                name = 'recording_metadata.json' if scene == 'lab' else f'recording_metadata-{scene}.json'
                (root / name).write_text(json.dumps({'sha256': {'policy.onnx': 'baseline', 'scene.xml': 'geometry'},
                                                    'scene': scene, 'nq': 50}))
            source = UPSTREAM / 'assets/ckpts/twist2_1017_25k.onnx'
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            config = root / 'config.json'
            config.write_text(json.dumps([dict(id='test-candidate', path=str(source), sha256=digest,
                                               title='Candidate', note='Experimental')]))
            catalog = export_policies(root, UPSTREAM, config)
            self.assertEqual([row['id'] for row in catalog], ['twist2-20k', 'twist2-25k', 'test-candidate'])
            for scene in ('lab', 'stand', 'cup', 'push_t'):
                metadata = json.loads((root / f'recording_metadata-{scene}-policy-test-candidate.json').read_text())
                self.assertEqual(metadata['policy_sha256'], digest)
                self.assertEqual(metadata['sha256']['policy-test-candidate.onnx'], digest)
                self.assertNotIn('policy.onnx', metadata['sha256'])
                self.assertEqual(metadata['sha256']['scene.xml'], 'geometry')
                self.assertEqual(metadata['scene'], scene)
            config.write_text('[]')
            export_policies(root, UPSTREAM, config)
            self.assertFalse((root / 'policy-test-candidate.onnx').exists())
            self.assertNotIn('test-candidate', (root / 'policy_catalog.txt').read_text())

    def test_wrong_candidate_hash_fails_build(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / 'config.json'
            config.write_text(json.dumps([dict(id='candidate', path=str(UPSTREAM / 'assets/ckpts/twist2_1017_20k.onnx'),
                                               sha256='0' * 64, title='Candidate', note='Experimental')]))
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                export_policies(root, UPSTREAM, config)
