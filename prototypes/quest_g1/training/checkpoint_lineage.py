"""Reject warm starts that would evaluate on an ancestor's training data."""
import hashlib
import json


def audit_evaluation(current_manifest, evaluation_dataset):
    """The separate MuJoCo NPZ dataset must be the exact protected holdout."""
    expected = {c['id']:c for c in current_manifest['clips'] if c['split'] == 'validation'}
    manifest = json.loads((evaluation_dataset/'manifest.json').read_text())
    rows = [c for c in manifest['clips'] if c['split'] == 'validation']
    actual = {c['id']:c for c in rows}
    if len(rows) != len(actual) or actual.keys() != expected.keys():
        raise ValueError('Runtime evaluation holdout identities differ')
    for name, c in actual.items():
        digest = hashlib.sha256((evaluation_dataset/c['file']).read_bytes()).hexdigest()
        if digest != c['sha256'] or digest != expected[name]['npz_sha256'] or c['stage'] != expected[name]['stage']:
            raise ValueError('Runtime evaluation holdout content differs: '+name)
    return {'runtime_validation_matches_training_holdout':True, 'clips':len(actual)}


def audit(checkpoint, previous_manifest, current_manifest):
    previous_bytes = previous_manifest.read_bytes()
    expected = checkpoint.get('dataset_manifest_sha256')
    if not expected or hashlib.sha256(previous_bytes).hexdigest() != expected:
        raise ValueError('Checkpoint training-manifest hash mismatch')
    previous = json.loads(previous_bytes)
    old = {c['id']: c for c in previous['clips']}
    new = {c['id']: c for c in current_manifest['clips']}
    if len(old) != len(previous['clips']) or len(new) != len(current_manifest['clips']):
        raise ValueError('Duplicate motion identity')
    for name, c in old.items():
        if name not in new or new[name]['split'] != c['split'] or new[name].get('source') != c.get('source'):
            raise ValueError('Ancestor motion removed or moved across split: '+name)
        if c.get('source') == 'twist2_author':
            if new[name]['sha256'] != c['sha256'] or new[name].get('author_path') != c.get('author_path'):
                raise ValueError('Ancestor author training data changed: '+name)
        elif new[name].get('stage') != c.get('stage') or new[name].get('parent_id') != c.get('parent_id'):
            raise ValueError('Quest source group changed: '+name)
    old_train = [c for c in old.values() if c['split'] == 'train']
    validation = [c for c in new.values() if c['split'] == 'validation']
    hashes = {c[k] for c in old_train for k in ('sha256', 'npz_sha256') if k in c}
    stages = {c['stage'] for c in old_train if c.get('source') == 'quest'}
    for c in validation:
        if any(c.get(k) in hashes for k in ('sha256', 'npz_sha256')):
            raise ValueError('Ancestor training content in validation')
        if c.get('source') != 'quest' or c['stage'] in stages:
            raise ValueError('Validation contains author data or ancestor Quest train stage')
    return {'ancestor_manifest_sha256':expected, 'ancestor_train_preserved':len(old_train),
            'validation_clips':len(validation), 'ancestor_train_in_validation':0,
            'quest_validation_stages':sorted({c['stage'] for c in validation})}
