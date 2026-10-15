"""Build the cup π_high training HDF5 from the published Quest cup exports.

Sources (furry-visual-il-v1, body35_grip2_gaze2):
- standing: datasets/quest-cup/2026-10-15 (HOMIE Standing, 40 train / 4 valid);
- v2: datasets/quest-cup/2026-10-15-homie-v2 (HOMIE v2 + catch-up, 59 demos, all train).

HOMIE Standing zeroes navigation before the Balance actor, but its recorded references keep the
operator's vx, vy and yaw rate. Under HOMIE v2 the same values would walk the robot, so in standing
demos they are set to the executed value, zero, in both actions and previous_command. Nothing else
is changed; images are copied as stored (LZF).

The v2 export has no validation split: the ``--v2-valid`` episodes with the lowest
sha256(episode name) go to valid (whole episodes, deterministic).
"""
import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

NAVIGATION = [0, 1, 5]  # vx, vy, yaw rate in body35


def split_v2(names, episodes, count):
    order = sorted(names, key=lambda n: hashlib.sha256(episodes[n].encode()).hexdigest())
    return set(order[:count])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('output', type=Path)
    p.add_argument('--standing', type=Path, help='cup44-no-shadow-causal.hdf5 (omit for v2 only)')
    p.add_argument('--v2', type=Path, required=True, help='cup59-no-shadow-causal.hdf5')
    p.add_argument('--v2-valid', type=int, default=6)
    a = p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    sources = ([('standing', a.standing)] if a.standing else []) + [('v2', a.v2)]
    rows = []
    with h5py.File(a.output, 'w') as out:
        out.attrs.update(schema='furry-visual-il-v1', action_schema='body35_grip2_gaze2', fps=30)
        train, valid = [], []
        for kind, path in sources:
            with h5py.File(path, 'r') as h:
                if h.attrs['schema'] != 'furry-visual-il-v1' or h.attrs['action_schema'] != 'body35_grip2_gaze2' or int(h.attrs['fps']) != 30:
                    raise ValueError(f'{path}: unexpected schema')
                masks = {s: [x.decode() for x in h['mask/'+s][:]] for s in ('train', 'valid')}
                episodes = {n: h['data/'+n].attrs['episode'] for n in masks['train']+masks['valid']}
                held = split_v2(masks['train'], episodes, a.v2_valid) if kind == 'v2' and not masks['valid'] else set()
                for split in ('train', 'valid'):
                    for name in masks[split]:
                        new = f'demo_{len(rows)}'
                        h.copy(h['data/'+name], out.require_group('data'), name=new)
                        d = out['data/'+new]
                        if kind == 'standing':
                            for key in ('actions', 'obs/previous_command'):
                                x = d[key][:]; x[:, NAVIGATION] = 0; d[key][...] = x
                        d.attrs['source'] = kind; d.attrs['source_demo'] = name
                        d.attrs['navigation_zeroed'] = kind == 'standing'
                        target = 'valid' if split == 'valid' or name in held else 'train'
                        (valid if target == 'valid' else train).append(new)
                        rows.append(dict(demo=new, source=kind, source_demo=name, episode=episodes[name], split=target,
                                         samples=int(d.attrs['num_samples'])))
        out.create_dataset('mask/train', data=np.array(train, dtype='S'))
        out.create_dataset('mask/valid', data=np.array(valid, dtype='S'))
        out.attrs['report'] = json.dumps(dict(sources={k: str(v) for k, v in sources}, navigation_zeroed='standing',
                                              v2_valid_rule=f'{a.v2_valid} lowest sha256(episode)', demos=rows))
    summary = {f'{k}/{s}': [sum(1 for r in rows if r['source'] == k and r['split'] == s),
                            sum(r['samples'] for r in rows if r['source'] == k and r['split'] == s)]
               for k, _ in sources for s in ('train', 'valid')}
    print(json.dumps(dict(output=str(a.output), demos_frames=summary), indent=1))


if __name__ == '__main__':
    main()
