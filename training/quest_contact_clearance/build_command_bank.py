"""Pair native commands with existing TRAIN clips without changing their NPZ/PKL.

Uses each continuous run's original body clock. Receive time is used only to
locate the recorded run origin, never to differentiate commands or interpolate
across tracking events. Mirrors are deterministic augmentation, not new data.
"""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def mirror_command(cmd, joints):
    out = cmd.copy()
    out[:, [1, 3, 5]] *= -1
    for i, name in enumerate(joints):
        other = 'right_'+name[5:] if name.startswith('left_') else ('left_'+name[6:] if name.startswith('right_') else name)
        out[:, 6+i] = cmd[:, 6+joints.index(other)] * (-1 if 'roll' in name or 'yaw' in name else 1)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--continuous-manifest', type=Path, required=True)
    p.add_argument('--episode', type=Path, required=True)
    p.add_argument('--exports', type=Path, nargs='+', required=True)
    p.add_argument('--export-receipt', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--camera-tracking', type=int, choices=[0, 1], required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists() or a.output.with_suffix('.json').exists():
        raise ValueError('Use a fresh output path')
    manifest = json.loads((a.dataset/'manifest.json').read_text())
    continuous = json.loads(a.continuous_manifest.read_text())
    export_modes = [json.loads(Path(str(f)+'.metadata.json').read_text()) for f in a.exports]
    if any(bool(r['camera_tracking']) != bool(a.camera_tracking) or r['full_feet'] or r['catch_up'] for r in export_modes):
        raise ValueError('Export modes do not match the requested bank')
    inputs = np.genfromtxt(a.episode/'input.csv', delimiter=',', names=True)
    wall = (inputs['receive_ns']-inputs['receive_ns'][0])/1e9
    epoch_rows = [np.genfromtxt(f, delimiter=',', names=True) for f in a.exports]
    by_seq = {int(v): (ei, i) for ei, rows in enumerate(epoch_rows) for i, v in enumerate(rows['sequence'])}
    joints = [j.attrib['name'] for j in ET.parse(a.model).getroot().iter('joint') if j.get('name')]
    if len(joints) != 29:
        raise ValueError('Expected the 29-joint GMR model')
    origins = {c['id'].rsplit('-', 1)[0]: c for c in continuous['clips'] if c['id'].endswith('-000000')}
    originals, arrays, index = {}, {}, {}
    quest = [c for c in manifest['clips'] if c['source'] == 'quest' and c['split'] == 'train']
    for c in quest:
        if c.get('augmentation'):
            continue
        run = c['id'].split('-part')[0].rsplit('-', 1)[0]
        base_wall = origins[run]['source_start_wall_s']
        ii = int(np.abs(wall-base_wall).argmin())
        if abs(wall[ii]-base_wall) > 1e-5:
            raise ValueError('Cannot match the original run clock: '+run)
        ei, ri = by_seq[int(inputs['sequence'][ii])]
        rows = epoch_rows[ei]
        start = rows['time_ns'][ri]/1e9 + c['source_start_wall_s']-base_wall
        t = rows['time_ns']/1e9
        # The final native solve for a repeated body timestamp is the held value.
        keep = np.r_[np.diff(t) > 0, True]
        t, rows = t[keep], rows[keep]
        if not (np.diff(t) > 0).all():
            raise ValueError('Body clock must be monotonic inside a calibration epoch')
        first = np.searchsorted(t, start, side='right')-1
        last = np.searchsorted(t, start+c['seconds'], side='right')
        if first < 0 or last <= first+1:
            raise ValueError('Missing command coverage: '+c['id'])
        cmd = np.column_stack([rows[f'cmd_{k}'][first:last] for k in range(35)])
        relative = t[first:last]-start
        relative[0] = 0.
        originals[c['id']] = np.column_stack((relative, cmd))
    for c in quest:
        if c.get('augmentation'):
            if c['augmentation'] != 'sagittal_reflection':
                raise ValueError('Unsupported augmentation')
            arr = originals[c['parent_id']].copy()
            arr[:, 1:] = mirror_command(arr[:, 1:], joints)
        else:
            arr = originals[c['id']]
        key = 'motion_%04d' % len(arrays)
        arrays[key] = arr
        index[c['file']] = {'key': key, 'target_sha256': c['sha256'], 'split': 'train',
                           'source': 'quest', 'frames': len(arr), 'seconds': c['seconds'],
                           'parent_id': c.get('parent_id'), 'stage': c['stage']}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.output, **arrays)
    receipt = json.loads(a.export_receipt.read_text())
    metadata = {'schema': 'quest-native-command-bank-v1', 'npz_sha256': sha(a.output),
                'dataset_manifest_sha256': sha(a.dataset/'manifest.json'), 'files': index,
                'native_export': receipt, 'camera_tracking': bool(a.camera_tracking),
                'export_modes': export_modes,
                'source_sha256': {str(f): sha(f) for f in a.exports + [a.continuous_manifest, a.episode/'input.csv']},
                'command_sampling': 'causal zero-order hold at original body-event times; no future interpolation',
                'reward_target': 'existing filtered GMR reference, not measured human legs',
                'runtime_match': {'full_feet': False, 'catch_up': False, 'startup_blend': False,
                                  'exact_recorded_consumption_schedule': False},
                'long_run_ready': False,
                'limitations': ['Human-only capture has no consumed policy-command log.',
                               'Native retargeting events reconstructed at body updates, not the original policy scheduler.',
                               '14-marker capture cannot reconstruct full-foot mode.',
                               'No paired measured human feet; this does not identify a Virtual Legs error distribution.',
                               'Catch-up must use the candidate robot state and consistent reward goals.']}
    a.output.with_suffix('.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(json.dumps({'clips': len(index), 'original_clips': len(originals), 'sha256': sha(a.output)}))


if __name__ == '__main__':
    main()
