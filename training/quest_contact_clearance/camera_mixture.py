"""Camera-leg clips in the TWIST2/Quest mixture, with one actor command bank.

combine: merge quest-g1-motion-v1 datasets (e.g. night-v3 Meta clips and a
  camera_dataset.py export) into one Quest dataset for author_dataset.py.
  Files are copied byte for byte; stages must not collide, so each source
  keeps its own whole-stage holdout.
bank: after author_dataset.py, pair every Quest TRAIN clip of the mixture with
  exactly one command timeline from the given banks and write a
  quest-native-command-bank-v1 file for native_command_bank.py. Native banks
  match on the mixture PKL hash, camera banks on the original NPZ hash
  (author_dataset.py copies NPZ bytes and rebuilds the PKL). A native bank
  given as BANK:OLD_MIXTURE matches on the NPZ hash recorded in its own
  mixture instead, so a rebuilt PKL with different bytes still pairs.
Usage:
  camera_mixture.py combine OUT QUEST_DATASET ...
  camera_mixture.py bank --dataset MIXTURE --banks BANK.npz[:OLD_MIXTURE] ... --output OUT.npz
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np

BANK_SCHEMAS = ('quest-native-command-bank-v1', 'quest-camera-command-bank-v1')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def combine(output, sources):
    output = Path(output)
    if output.exists():
        raise ValueError('Use a fresh output path')
    clips, stage_owner, validation, rows = [], {}, set(), []
    for src in map(Path, sources):
        m = json.loads((src/'manifest.json').read_text())
        if m.get('schema') != 'quest-g1-motion-v1':
            raise ValueError(f'Not a Quest motion dataset: {src}')
        tag = m.get('source', 'quest-meta')
        for c in m['clips']:
            if c['split'] not in ('train', 'validation'):
                continue
            if stage_owner.setdefault(c['stage'], src) != src:
                raise ValueError(f"Stage {c['stage']} appears in two sources")
            if sha(src/c['file']) != c['sha256']:
                raise ValueError(f"Source changed: {src/c['file']}")
            dest = output/c['file']
            if dest.exists():
                raise ValueError(f'File collision: {c["file"]}')
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src/c['file'], dest)
            clips.append(dict(c, input_source=tag))
        validation |= set(m['validation_stages'])
        rows.append({'path': str(src.resolve()), 'manifest_sha256': sha(src/'manifest.json'), 'source': tag})
    if len({c['id'] for c in clips}) != len(clips):
        raise ValueError('Duplicate clip id across sources')
    manifest = {'schema': 'quest-g1-motion-v1', 'fps': 100, 'combined_from': rows,
                'validation_stages': sorted(validation),
                'duration_seconds': {s: sum(c['seconds'] for c in clips if c['split'] == s) for s in ('train', 'validation')},
                'seconds_by_source': {f"{r['source']}/{s}": sum(c['seconds'] for c in clips if c['input_source'] == r['source'] and c['split'] == s)
                                      for r in rows for s in ('train', 'validation')},
                'clips': clips}
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2))
    return manifest


def bank(dataset, banks, output):
    dataset, output = Path(dataset), Path(output)
    if output.exists() or output.with_suffix('.json').exists():
        raise ValueError('Use a fresh output path')
    manifest = json.loads((dataset/'manifest.json').read_text())
    if manifest.get('schema') != 'twist2-quest-mixture-v2':
        raise ValueError('Expected an author_dataset.py mixture')
    by_hash, sources = {}, []
    paths = []
    for i, spec in enumerate(map(str, banks)):
        path, _, own = spec.partition(':'); path = Path(path); paths.append(path)
        meta = json.loads(path.with_suffix('.json').read_text())
        if meta.get('schema') not in BANK_SCHEMAS or meta['npz_sha256'] != sha(path):
            raise ValueError(f'Unknown or changed command bank: {path}')
        npz_of = {}
        if own:
            if meta['dataset_manifest_sha256'] != sha(Path(own)/'manifest.json'):
                raise ValueError(f'{path} was not built for {own}')
            npz_of = {c['file']: c['npz_sha256'] for c in json.loads((Path(own)/'manifest.json').read_text())['clips'] if 'npz_sha256' in c}
        for name, row in meta['files'].items():
            target = npz_of[name] if own else row['target_sha256']
            if target in by_hash:
                raise ValueError('Two banks provide the same reward target')
            by_hash[target] = (i, row['key'], name)
        sources.append({k: v for k, v in meta.items() if k != 'files'} | {'path': str(path.resolve())})
    arrays, files, used = {}, {}, set()
    loaded = [np.load(p, allow_pickle=False) for p in paths]
    for c in manifest['clips']:
        if c['source'] != 'quest' or c['split'] != 'train':
            continue
        hits = [h for h in (c['sha256'], c.get('npz_sha256')) if h in by_hash]
        if len(hits) != 1:
            raise ValueError(f"Need exactly one command timeline for {c['file']}, found {len(hits)}")
        i, key, name = by_hash[hits[0]]; used.add(hits[0])
        arr = loaded[i][key]
        if arr.ndim != 2 or arr.shape[1] != 36 or not np.isfinite(arr).all() or arr[0, 0] != 0 or not (np.diff(arr[:, 0]) > 0).all():
            raise ValueError(f'Invalid command timeline for {c["file"]}')
        out_key = 'motion_%04d' % len(arrays); arrays[out_key] = arr
        files[c['file']] = {'key': out_key, 'target_sha256': c['sha256'], 'split': 'train', 'source': 'quest',
                            'input_source': c.get('input_source', 'quest-meta'), 'bank': i, 'bank_entry': name,
                            'frames': len(arr), 'seconds': c['seconds'], 'parent_id': c.get('parent_id'), 'stage': c['stage']}
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    inputs = sorted({f['input_source'] for f in files.values()})
    metadata = {'schema': 'quest-native-command-bank-v1', 'npz_sha256': sha(output),
                'dataset_manifest_sha256': sha(dataset/'manifest.json'), 'files': files,
                'merged_banks': sources, 'unused_bank_entries': len(by_hash) - len(used),
                'seconds_by_input': {s: sum(f['seconds'] for f in files.values() if f['input_source'] == s) for s in inputs},
                'command_sampling': 'causal zero-order hold at original command times; no future interpolation',
                'long_run_ready': False,
                'limitations': ['Merged inputs keep their own limitations; see merged_banks.']}
    output.with_suffix('.json').write_text(json.dumps(metadata, indent=2)+'\n')
    return metadata


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='command', required=True)
    c = sub.add_parser('combine'); c.add_argument('output', type=Path); c.add_argument('sources', type=Path, nargs='+')
    b = sub.add_parser('bank'); b.add_argument('--dataset', type=Path, required=True)
    b.add_argument('--banks', nargs='+', required=True); b.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.command == 'combine':
        m = combine(a.output, a.sources)
        print(json.dumps({k: m[k] for k in ('validation_stages', 'duration_seconds', 'seconds_by_source')}))
    else:
        m = bank(a.dataset, a.banks, a.output)
        print(json.dumps({'clips': len(m['files']), 'seconds_by_input': m['seconds_by_input'],
                          'unused_bank_entries': m['unused_bank_entries'], 'sha256': m['npz_sha256']}))


if __name__ == '__main__':
    main()
