"""TRAIN-only differences, not a measurement of true human leg error."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def summarize(x):
    x = np.concatenate(x).ravel()
    return {'mean_absolute': float(np.mean(np.abs(x))), 'p95_absolute': float(np.quantile(np.abs(x), .95)),
            'max_absolute': float(np.max(np.abs(x))), 'rms': float(np.sqrt(np.mean(x*x)))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--bank', type=Path, required=True)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    meta = json.loads(a.bank.with_suffix('.json').read_text())
    manifest = json.loads((a.dataset/'manifest.json').read_text())
    if hashlib.sha256((a.dataset/'manifest.json').read_bytes()).hexdigest() != meta['dataset_manifest_sha256']:
        raise ValueError('Dataset changed')
    aggregate = {'legs_rad': [], 'upper_body_rad': [], 'root_height_m': []}
    rows = []
    with np.load(a.bank, allow_pickle=False) as bank:
        for c in manifest['clips']:
            if c['split'] != 'train' or c['source'] != 'quest' or c.get('augmentation'):
                continue
            with np.load(a.dataset/c['npz_file'], allow_pickle=False) as z:
                q = z['qpos']; fps = float(z['fps'])
            timeline = bank[meta['files'][c['file']]['key']]
            times = np.arange(len(q))/fps
            idx = np.searchsorted(timeline[:, 0], times, side='right')-1
            idx = np.maximum(idx, 0)
            cmd = timeline[idx, 1:]
            parts = {'legs_rad': cmd[:, 6:18]-q[:, 7:19],
                     'upper_body_rad': cmd[:, 18:35]-q[:, 19:36],
                     'root_height_m': cmd[:, 2]-q[:, 2]}
            rows.append({'id': c['id'], 'stage': c['stage'], 'frames': len(q),
                         **{k: summarize([v]) for k, v in parts.items()}})
            for k, v in parts.items():
                aggregate[k].append(v)
    result = {'schema': 1, 'bank_sha256': meta['npz_sha256'], 'original_train_clips': len(rows),
              'validation_clips_used': 0, 'mirrors_counted_as_new_data': False,
              'units': 'radians or metres as named',
              'interpretation': 'Difference between reconstructed native commands and existing filtered GMR targets. Neither is measured human leg ground truth; no gait improvement is established.',
              'metrics': {k: summarize(v) for k, v in aggregate.items()}, 'per_clip': rows}
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result['metrics'], indent=2))


if __name__ == '__main__':
    main()
