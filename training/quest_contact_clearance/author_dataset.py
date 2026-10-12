"""Auditable TWIST2 non-PICO train subset + protected Quest stage holdout.

No upstream motion is assigned to validation. Selection never uses rollout scores.
The YAML manifests can be loaded by the unmodified upstream MotionLib.
"""
import argparse
import collections
import hashlib
import json
import pickle
import re
import shutil
from pathlib import Path

import numpy as np
from motion_coordinates import world_offsets_to_root

from select_author_motions import FAMILIES, REV


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ArrayUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module in ('numpy.core.multiarray', 'numpy._core.multiarray') and name in ('_reconstruct', 'scalar'):
            from numpy.core import multiarray
            return getattr(multiarray, name)
        if module == 'numpy' and name in ('ndarray', 'dtype'):
            return getattr(np, name)
        raise pickle.UnpicklingError(f'Unsupported pickle global {module}.{name}')


def load_motion(path):
    with Path(path).open('rb') as f:
        return ArrayUnpickler(f).load()


def validate_motion(x, expected_links=None, strict_quaternions=True):
    fps = float(x['fps'])
    n = len(x['root_pos'])
    names = tuple(x['link_body_list'])
    if not np.isfinite(fps) or fps <= 0 or n < 2:
        raise ValueError('Invalid duration/fps')
    if len(names) != len(set(names)) or (expected_links is not None and names != tuple(expected_links)):
        raise ValueError('Body ordering mismatch')
    for k, shape in [('root_pos', (n, 3)), ('root_rot', (n, 4)), ('dof_pos', (n, 29)), ('local_body_pos', (n, len(names), 3))]:
        a = np.asarray(x[k])
        if a.shape != shape or not np.isfinite(a).all():
            raise ValueError(f'Invalid {k}: {a.shape}, expected {shape}')
    norm_error = np.max(abs(np.linalg.norm(x['root_rot'], axis=1) - 1))
    if norm_error > (1e-3 if strict_quaternions else .1):
        raise ValueError('Invalid root quaternion norm')
    if np.max(abs(np.asarray(x['local_body_pos'])[:, names.index('pelvis')])) > 1e-4:
        raise ValueError('local_body_pos pelvis must be zero')
    return (n - 1) / fps, names


def quest_weights(clips, total):
    train = {c['id']: c for c in clips if c['split'] == 'train'}
    families = collections.defaultdict(list)
    for c in train.values():
        parent = c.get('parent_id', c['id'])
        if parent not in train or train[parent].get('parent_id'):
            raise ValueError('Augmentation parent missing from train')
        if c['stage'] != train[parent]['stage']:
            raise ValueError('Augmentation changed stage')
        families[parent].append(c['id'])
    if not families:
        raise ValueError('No Quest train clips')
    return {name: total / len(families) / len(group) for group in families.values() for name in group}


def guard_quest_split(clips, validation_stages):
    ids = set()
    hashes = {}
    for c in clips:
        if c['id'] in ids:
            raise ValueError('Duplicate Quest id')
        ids.add(c['id'])
        if c['split'] not in ('train', 'validation'):
            raise ValueError('Only explicit Quest train/validation accepted')
        expected = 'validation' if c['stage'] in validation_stages else 'train'
        if c['split'] != expected:
            raise ValueError('Held-out stage leakage')
        if c.get('augmentation') and c['split'] != 'train':
            raise ValueError('Validation augmentation forbidden')
        if c['sha256'] in hashes and hashes[c['sha256']] != c['split']:
            raise ValueError('Cross-split duplicate content')
        hashes[c['sha256']] = c['split']


def build(selection_path, raw, quest, output, author_manifest):
    selection = json.loads(selection_path.read_text())
    author_text = author_manifest.read_text()
    members = set(re.findall(r'^- file: (.+)$', author_text, re.M))
    if sha(author_manifest) != selection['source_manifest_sha256']:
        raise ValueError('Author manifest hash mismatch')
    if selection['source_revision'] != REV:
        raise ValueError('Unexpected upstream revision')
    if output.exists():
        raise ValueError('Output must be new; never modify an existing dataset')
    qm = json.loads((quest / 'manifest.json').read_text())
    # Existing public demo fixtures include PICO; they are deliberately not imported.
    qc = [c for c in qm['clips'] if c['split'] in ('train', 'validation')]
    guard_quest_split(qc, qm['validation_stages'])
    weights = quest_weights(qc, selection['quest_replacement_weight'])
    download = json.loads((raw / 'download-manifest.json').read_text())
    if download['selection_sha256'] != sha(selection_path):
        raise ValueError('Download/selection mismatch')
    downloaded = {c['author_path']: c['sha256'] for c in download['files']}
    output.mkdir(parents=True)
    rows = []
    links = None
    for c in selection['clips']:
        rel = Path(c['author_path'])
        if rel.is_absolute() or '..' in rel.parts or rel.parts[0] not in FAMILIES:
            raise ValueError('Non-whitelisted author family/path')
        if c['author_split'] != 'train' or c['local_split'] != 'train':
            raise ValueError('Upstream split changed')
        if str(rel) not in members:
            raise ValueError('File outside author train manifest')
        src = raw / rel
        if sha(src) != downloaded[str(rel)]:
            raise ValueError('Downloaded motion changed')
        x = load_motion(src)
        seconds, links = validate_motion(x, links, strict_quaternions=False)
        dest = output / 'author_train' / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
        rows.append({'id': 'author/' + str(rel.with_suffix('')), 'file': str(dest.relative_to(output)), 'split': 'train', 'source': 'twist2_author', 'author_split': 'train', 'author_path': str(rel), 'source_revision': REV, 'family': c['family'], 'weight': c['sampling_weight'], 'seconds': seconds, 'fps': float(x['fps']), 'max_quaternion_norm_error': float(np.max(abs(np.linalg.norm(x['root_rot'], axis=1) - 1))), 'sha256': sha(dest)})
    for c in qc:
        src = quest / c['file']
        if sha(src) != c['sha256']:
            raise ValueError(f'Quest source changed: {src}')
        with np.load(src, allow_pickle=False) as z:
            q = z['qpos']
            x = {'fps': float(z['fps']), 'root_pos': q[:, :3], 'root_rot': q[:, [4, 5, 6, 3]], 'dof_pos': q[:, 7:], 'local_body_pos': world_offsets_to_root(z['local_body_pos'], q[:, [4, 5, 6, 3]]), 'link_body_list': z['link_body_list'].tolist()}
            seconds, _ = validate_motion(x, links)
        dest = output / 'quest' / c['file']
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)  # Byte-for-byte original holdout remains available for MuJoCo eval.
        with dest.with_suffix('.pkl').open('wb') as f:
            pickle.dump(x, f, protocol=4)
        rows.append(dict(c, source='quest', file=str(dest.with_suffix('.pkl').relative_to(output)), npz_file=str(dest.relative_to(output)), npz_sha256=sha(dest), sha256=sha(dest.with_suffix('.pkl')), weight=weights.get(c['id'], 1.0), seconds=seconds))
    train_hashes = {c['sha256'] for c in rows if c['split'] == 'train'}
    if any(c['sha256'] in train_hashes for c in rows if c['split'] == 'validation'):
        raise ValueError('Cross-source train/validation duplicate content')
    summary = {
        'schema': 'twist2-quest-mixture-v2', 'source_revision': REV,
        'selection_sha256': sha(selection_path), 'author_manifest_sha256': selection['source_manifest_sha256'],
        'download_manifest_sha256': sha(raw / 'download-manifest.json'),
        'quest_manifest_sha256': sha(quest / 'manifest.json'),
        'validation_stages': qm['validation_stages'],
        'scaling': 'Already retargeted to G1: upstream motions unchanged; Quest fixed anatomy scaling retained, no second scaling.',
        'body_coordinates': 'PKL local_body_pos is root-oriented (R_root inverse times world offset). Legacy Quest NPZ stays byte-identical with world-axis offsets for runtime evaluation.',
        'sampling': 'Per-motion upstream weights; non-PICO family masses retained after subsampling. Quest replaces total PICO mass; mirrors share parent weight. No duration weighting or survival filtering.',
        'split_policy': 'Every upstream selected file is in pinned author TRAIN. Quest whole-stage holdout unchanged. No upstream/unknown/PICO files in validation.',
        'upstream_data_caveat': 'TWIST1 converted files have non-unit quaternions (up to ~7.3% norm error). Preserved byte-for-byte to reproduce upstream; recorded per file, not silently normalized or filtered.',
        'validation_limitations': 'Existing 12 Quest clips from one session; already used diagnostically. No independent test set; adding upstream train does not create new validation.',
        'quest_probability_before_curriculum': selection['quest_probability_before_curriculum'],
        'counts': dict(collections.Counter(c['source'] + '/' + c['split'] for c in rows)),
        'seconds': {k: sum(c['seconds'] for c in rows if c['source'] + '/' + c['split'] == k) for k in {c['source'] + '/' + c['split'] for c in rows}},
        'clips': rows}
    for split in ('train', 'validation'):
        # Upstream resolves root_path against CWD; the launch entrypoint must chdir to this dataset.
        lines = ['root_path: .', 'motions:']
        for c in rows:
            if c['split'] == split:
                lines.extend(['- file: ' + json.dumps(c['file']), '  weight: ' + str(c['weight'])])
        (output / (split + '.yaml')).write_text('\n'.join(lines) + '\n')
    (output / 'manifest.json').write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for arg in ('selection', 'raw', 'quest', 'output', 'author-manifest'):
        p.add_argument('--' + arg, type=Path, required=True)
    a = p.parse_args()
    result = build(a.selection, a.raw, a.quest, a.output, a.author_manifest)
    print(json.dumps({k: v for k, v in result.items() if k != 'clips'}, indent=2))
