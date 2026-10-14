"""Calibration-epoch sources and run origins of a Quest capture, as prepare.py makes them.

build_command_bank.py needs the epoch{N}.txt files that prepare.py feeds to
native GMR and the run origins of its continuous (pre-curation) manifest.
Neither is kept in the published dataset. This rebuilds both from the raw
episode with prepare.py's selection logic (used frames, stages, tracking-jump
exclusions, calibration epochs) and no retargeting, so it runs anywhere.
Usage: meta_epochs.py EPISODE_DIR OUT_DIR
Writes OUT/epoch{N}.txt (export_native_commands input) and OUT/continuous.json.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np


def read(path):
    with path.open() as f:
        names = f.readline().strip().split(',')
    a = np.loadtxt(path, delimiter=',', skiprows=1, ndmin=2)
    if not np.isfinite(a).all():
        raise ValueError(f'Nonfinite {path}')
    return a, {k: i for i, k in enumerate(names)}


def regions(mask):
    ids = np.flatnonzero(mask)
    return [x for x in np.split(ids, np.flatnonzero(np.diff(ids) > 1)+1) if len(x)]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('episode', type=Path); p.add_argument('output', type=Path)
    a = p.parse_args()
    out = a.output; out.mkdir(parents=True, exist_ok=False)
    i, ic = read(a.episode/'input.csv'); b, bc = read(a.episode/'body.csv'); t, tc = read(a.episode/'capture_timeline.csv')
    assert np.array_equal(i[:, ic['sequence']], b[:, bc['sequence']])
    # From here to the epoch loop: prepare.py's selection, unchanged.
    seq = i[:, ic['sequence']].astype(np.int64); times = i[:, ic['receive_ns']]/1e9; wall = times-times[0]
    idx = np.searchsorted(seq, t[:, tc['sequence']]); acc = np.diff(t[:, tc['accepted_seconds']], prepend=0) > 0
    used = np.zeros(len(i), bool); stage = np.full(len(i), -1); used[idx[acc]] = True; stage[idx[acc]] = t[acc, tc['stage']].astype(int)+1
    for j in np.flatnonzero(acc & (np.diff(t[:, tc['stage']], prepend=t[0, tc['stage']]) > 0)):
        stage[idx[j]] = int(t[j-1, tc['stage']])+1
    pos = b[:, [bc[f'pose_{j}_{k}'] for j in range(14) for k in range(3)]].reshape(-1, 14, 3)
    quat = b[:, [bc[f'pose_{j}_{k}'] for j in range(14) for k in range(3, 7)]].reshape(-1, 14, 4)
    quat /= np.linalg.norm(quat, axis=2, keepdims=True)
    dp = np.linalg.norm(np.diff(pos, axis=0), axis=2).max(axis=1); angle = 2*np.arccos(np.clip(abs((quat[1:]*quat[:-1]).sum(2)), 0, 1)).max(axis=1)
    consecutive = used[1:] & used[:-1] & (stage[1:] == stage[:-1]) & (np.diff(times) < .05)
    exclusions = []
    for j in np.flatnonzero(consecutive & ((dp > .25) | (angle > np.pi/4)))+1:
        lo, hi = wall[j]-.5, wall[j]+.5; used &= ~((wall >= lo) & (wall <= hi)); exclusions.append([float(lo), float(hi)])
    for side in ['left', 'right']:
        used &= (i[:, ic[side+'_flags']].astype(int) & 3) == 3; used &= i[:, ic[side+'_active']] == 1
    used &= b[:, bc['valid']] == 1
    cal = [int(np.searchsorted(seq, int(e['sequence']))) for e in csv.DictReader((a.episode/'events.csv').open()) if e['event'] == 'calibrate']
    if not cal:
        raise ValueError('Missing calibration')
    bodycols = [bc[f'{kind}_{j}_{k}'] for j in range(14) for kind in ['pose', 'rest'] for k in range(7)]
    headcols = [ic[f'head_{k}'] for k in ['x', 'y', 'z', 'qw', 'qx', 'qy', 'qz']]
    restcols = [bc[f'rest_{j}_{k}'] for j in range(14) for k in range(7)]

    def rawrow(j):
        return np.r_[seq[j], i[j, ic['xr_time_ns']], b[j, bc['time_ns']], b[j, bc['skeleton_version']], i[j, headcols], b[j, bodycols]]
    epochs = []
    for cn, cj in enumerate(cal):
        end = cal[cn+1] if cn+1 < len(cal) else len(b)
        compatible = np.max(abs(b[cj:end, restcols]-b[cj, restcols]), axis=1) < 1e-5
        ej = np.arange(cj, end)[compatible & (b[cj:end, bc['valid']] == 1)]
        for lo, hi in exclusions:
            ej = ej[~((wall[ej] >= lo) & (wall[ej] <= hi))]
        with (out/f'epoch{cn}.txt').open('w') as f:
            np.savetxt(f, np.vstack([rawrow(cj)]+[rawrow(j) for j in ej]), fmt='%.17g')
        epochs.append({'file': f'epoch{cn}.txt', 'calibration_sequence': int(seq[cj]), 'frames': int(len(ej))+1})
    # Run origins: prepare.py names runs stage{s}-run{n} and starts its first
    # clip ({name}-000000) at the run's first used frame.
    clips = []
    for s in range(1, 16):
        for runn, ids in enumerate(regions(used & (stage == s))):
            if times[ids[-1]]-times[ids[0]] < 2:
                continue
            clips.append({'id': f'stage{s:02d}-run{runn:02d}-000000', 'stage': s, 'source_start_wall_s': float(wall[ids[0]])})
    manifest = {'schema': 'quest-capture-run-origins-v1', 'source_episode': a.episode.name,
                'source_sha256': {n: sha(a.episode/n) for n in ['input.csv', 'body.csv', 'capture_timeline.csv', 'events.csv']},
                'exclusions_wall_seconds': exclusions, 'epochs': epochs, 'clips': clips,
                'note': 'Rebuilt with prepare.py selection logic; origins only, no retargeted clips.'}
    (out/'continuous.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({'epochs': epochs, 'runs': len(clips), 'exclusions': len(exclusions)}))


if __name__ == '__main__':
    main()
