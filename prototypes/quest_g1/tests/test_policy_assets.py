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
                             ['twist2-20k', 'twist2-25k', 'quest-111260-500', 'quest-111260-1000'])
            provenance = json.loads((Path(__file__).resolve().parents[1] /
                                     'policies/quest-111260/provenance.json').read_text())
            for row, artifact in zip(catalog[2:], provenance['artifacts']):
                self.assertEqual(row['sha256'], artifact['sha256'])
                self.assertEqual(hashlib.sha256((root / row['file']).read_bytes()).hexdigest(), artifact['sha256'])

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
