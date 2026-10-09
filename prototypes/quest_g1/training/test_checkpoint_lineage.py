import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from checkpoint_lineage import audit, audit_evaluation


class LineageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.previous = Path(self.temp.name)/'manifest.json'
        self.current = {'clips':[
            {'id':'author/a', 'source':'twist2_author', 'split':'train', 'sha256':'a', 'author_path':'a.pkl'},
            {'id':'quest/a', 'source':'quest', 'split':'train', 'stage':1, 'sha256':'b'},
            {'id':'quest/a-mirror', 'source':'quest', 'split':'train', 'stage':1, 'parent_id':'quest/a', 'sha256':'c'},
            {'id':'quest/v', 'source':'quest', 'split':'validation', 'stage':2, 'sha256':'d'}]}
        self.previous.write_text(json.dumps(self.current))
        self.checkpoint = {'dataset_manifest_sha256':hashlib.sha256(self.previous.read_bytes()).hexdigest()}

    def test_derivative_preserves_all_old_train(self):
        self.current['clips'][1]['sha256'] = 'toe-clamp-derivative'
        r = audit(self.checkpoint, self.previous, self.current)
        self.assertEqual(r['ancestor_train_preserved'], 3)
        self.assertEqual(r['ancestor_train_in_validation'], 0)

    def test_runtime_holdout_matches_and_rejects_replacement(self):
        folder = Path(self.temp.name)/'eval'; folder.mkdir()
        (folder/'v.npz').write_bytes(b'validation')
        digest = hashlib.sha256(b'validation').hexdigest()
        self.current['clips'][-1]['npz_sha256'] = digest
        row = dict(self.current['clips'][-1], file='v.npz', sha256=digest)
        (folder/'manifest.json').write_text(json.dumps({'clips':[row]}))
        self.assertTrue(audit_evaluation(self.current, folder)['runtime_validation_matches_training_holdout'])
        (folder/'v.npz').write_bytes(b'train substituted')
        with self.assertRaisesRegex(ValueError, 'content differs'):
            audit_evaluation(self.current, folder)
        row['id'] = 'quest/a'
        (folder/'manifest.json').write_text(json.dumps({'clips':[row]}))
        with self.assertRaisesRegex(ValueError, 'identities differ'):
            audit_evaluation(self.current, folder)

    def test_wrong_checkpoint_manifest_rejected(self):
        self.checkpoint['dataset_manifest_sha256'] = 'wrong'
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            audit(self.checkpoint, self.previous, self.current)

    def test_old_train_or_mirror_cannot_move_to_validation(self):
        for index in (0, 1, 2):
            m = copy.deepcopy(self.current); m['clips'][index]['split'] = 'validation'
            with self.assertRaises(ValueError):audit(self.checkpoint, self.previous, m)

    def test_renamed_copy_of_old_train_rejected(self):
        c = dict(self.current['clips'][1], id='renamed', split='validation', stage=2)
        self.current['clips'].append(c)
        with self.assertRaisesRegex(ValueError, 'training content'):
            audit(self.checkpoint, self.previous, self.current)

    def test_changed_content_from_old_train_stage_rejected(self):
        c = dict(self.current['clips'][1], id='new-cut', split='validation', sha256='new-content')
        self.current['clips'].append(c)
        with self.assertRaisesRegex(ValueError, 'train stage'):
            audit(self.checkpoint, self.previous, self.current)

    def test_validation_cannot_gain_author_data(self):
        c = dict(self.current['clips'][0], id='new-author', split='validation', sha256='new-author-content')
        self.current['clips'].append(c)
        with self.assertRaisesRegex(ValueError, 'author data'):
            audit(self.checkpoint, self.previous, self.current)


if __name__ == '__main__':
    unittest.main()
