"""Build and verify the bimanual (two objects) causal RGB training HDF5 from raw Quest episodes.

Same conventions as datasets/quest-cup/2026-10-15: no shadows, current physical
head camera with the previous 30 Hz gaze sample, 224x224 RGB, actions
body35_grip2_gaze2, whole-episode train/valid split.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[2]
PUBLISHED = ROOT/'rendered/bimanual69-no-shadow-causal.hdf5'
sys.path.insert(0, str(REPO / 'prototypes/quest_g1'))
sys.path.insert(0, str(REPO / 'datasets/quest-cup/2026-10-15/tools'))
from g1_sim.visual_dataset import inspect_episode, digest, table, SCHEMA
from training_renderer import TrainingRenderer


def survey(raw):
    """Classify every raw episode; accept successful ones of a single family."""
    rows, accepted, family, seen = [], [], None, set()
    for path in sorted(raw.glob('episode-*')):
        row = {'episode': path.name, 'outcome': json.loads((path/'learning-result.json').read_text())['outcome']}
        scene = table(path/'scene_state.csv')
        row['seed'] = int(scene['userdata_0'][0])
        try:
            e = inspect_episode(path)
            if e['identity'] in seen:
                raise ValueError('Duplicate recording content')
            if family is not None and e['family'] != family:
                raise ValueError('Different scene/model/policy configuration')
            family = e['family']; seen.add(e['identity']); accepted.append(e)
            row.update(accepted=True, identity=e['identity'], samples=len(e['indices']))
        except (ValueError, KeyError, OSError) as exc:
            row.update(accepted=False, reason=str(exc))
        rows.append(row)
    return rows, accepted, family


def build(output, validation_fraction, report_path=None):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    rows, episodes, family = survey(ROOT/'raw')
    n = len(episodes)
    valid_count = min(n-1, max(1, round(n*validation_fraction)))
    # Exporter rule: the lowest content identities go to validation.
    valid = {e['identity'] for e in sorted(episodes, key=lambda e: e['identity'])[:valid_count]}
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix('.partial.hdf5')
    masks, total, started = {'train': [], 'valid': []}, 0, time.time()
    renderer_info = {'shadows': False, 'resolution': [224, 224], 'source_resolution': [640, 360],
                     'mujoco': episodes[0]['capture']['mujoco_version'],
                     'camera': 'current physical head + previous sampled gaze; archived first gaze for initial state'}
    try:
        with h5py.File(partial, 'x') as h:
            h.attrs.update(schema=SCHEMA, fps=30, action_schema='body35_grip2_gaze2')
            h.attrs['renderer'] = json.dumps(renderer_info)
            h.attrs['family'] = json.dumps(family)
            data = h.create_group('data')
            for j, e in enumerate(episodes):
                ids = e['indices']; count = len(ids); name = f'demo_{j}'
                split = 'valid' if e['identity'] in valid else 'train'
                d = data.create_group(name)
                d.attrs.update(num_samples=count, episode=e['path'].name, family=json.dumps(e['family']),
                               capture=json.dumps(e['capture']), manifest=json.dumps(e['meta']))
                d.create_dataset('actions', data=e['actions'][ids].astype('float32'))
                previous = np.zeros((count, 39), np.float32)
                previous[1:] = e['actions'][ids[:-1]]
                obs = d.create_group('obs')
                obs.create_dataset('previous_command', data=previous)
                images = obs.create_dataset('agentview_image', (count, 224, 224, 3), dtype='uint8',
                                            chunks=(1, 224, 224, 3), compression='lzf')
                renderer = TrainingRenderer(e)
                renderer.model.light_castshadow[:] = 0
                try:
                    for k, idx in enumerate(ids):
                        images[k] = renderer.render_observation(idx, ids[max(0, k-1)])
                finally:
                    renderer.close()
                d.create_dataset('source_frame_index', data=ids)
                d.create_dataset('timestamps', data=e['grid'])
                d.create_dataset('source_sim_time', data=e['frames']['sim_time'][ids])
                masks[split].append(name.encode()); total += count
                for row in rows:
                    if row['episode'] == e['path'].name:
                        row.update(demo=name, split=split)
                print(e['path'].name, name, split, count, flush=True)
            for split, names in masks.items():
                h.create_dataset('mask/' + split, data=np.asarray(names, dtype='S32'))
        partial.rename(output)
    finally:
        partial.unlink(missing_ok=True)
    report = {'schema': 'furry-bimanual-export-v1', 'family': family, 'renderer': renderer_info,
              'episodes_raw': len(rows), 'episodes_accepted': n,
              'train': len(masks['train']), 'valid': len(masks['valid']),
              'independent_test_session': False, 'total_frames': total,
              'seconds': round(time.time()-started, 1), 'output': output.name, 'sha256': digest(output),
              'episodes': rows}
    # The published report describes the published HDF5; rebuilds write next to their output.
    report_path = Path(report_path) if report_path else output.with_suffix('.report.json')
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps({k: report[k] for k in ('episodes_raw', 'episodes_accepted', 'train', 'valid', 'total_frames', 'seconds', 'sha256')}))


def write_manifest(device_hashes):
    """Record every raw file's SHA256 (as read on the headset) plus the HDF5 hash."""
    device = {e['episode']: e['sha256'] for e in json.loads(Path(device_hashes).read_text())['episodes']}
    report = json.loads((ROOT/'dataset-report.json').read_text())
    episodes = []
    for row in report['episodes']:
        path = ROOT/'raw'/row['episode']
        files = {f.name: digest(f) for f in sorted(path.iterdir())}
        if files != device[row['episode']]:
            raise ValueError(f'{row["episode"]}: local files differ from headset hashes')
        episodes.append({**row, 'files': files})
    manifest = {'schema': 'furry-quest-bimanual-release-v1', 'date': '2026-10-15',
                'raw_episodes': len(episodes), 'rgb_episodes': report['episodes_accepted'],
                'rgb_frames': report['total_frames'], 'train': report['train'], 'validation': report['valid'],
                'independent_test_session': False, 'family': report['family'],
                'raw_byte_preservation': 'Headset file hashes; only local .download-exported markers omitted',
                'rendered': {PUBLISHED.name: report['sha256']}, 'episodes': episodes}
    (ROOT/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    print(len(episodes), 'episodes,', sum(len(e['files']) for e in episodes), 'files')


def verify(output, render_check=False):
    """Check hashes, re-derive every array from raw, optionally re-render first/middle/last frames."""
    manifest = json.loads((ROOT/'manifest.json').read_text())
    for e in manifest['episodes']:
        path = ROOT/'raw'/e['episode']
        assert {f.name for f in path.iterdir()} == set(e['files']), e['episode']
        for name, sha in e['files'].items():
            assert digest(path/name) == sha, (e['episode'], name, 'run git lfs pull?')
    output = Path(output)
    if output.resolve() == PUBLISHED.resolve():
        assert digest(output) == manifest['rendered'][output.name]
    rendered, max_diff, max_fraction = 0, 0, 0.
    with h5py.File(output, 'r') as h:
        train = {x.decode() for x in h['mask/train'][:]}; valid = {x.decode() for x in h['mask/valid'][:]}
        assert not train & valid and train | valid == set(h['data'])
        for name, d in h['data'].items():
            e = inspect_episode(ROOT/'raw'/d.attrs['episode']); ids = e['indices']; n = len(ids)
            np.testing.assert_array_equal(d['actions'][:], e['actions'][ids].astype('float32'))
            np.testing.assert_array_equal(d['source_frame_index'][:], ids)
            np.testing.assert_array_equal(d['timestamps'][:], e['grid'])
            prev = d['obs/previous_command'][:]
            np.testing.assert_array_equal(prev[1:], d['actions'][:-1]); assert not prev[0].any()
            images = d['obs/agentview_image']
            assert images.shape == (n, 224, 224, 3) and images.dtype == np.uint8
            for first in range(0, n, 32):
                assert images[first:first+32].std() > 1
            if render_check:
                r = TrainingRenderer(e); r.model.light_castshadow[:] = 0
                try:
                    # EGL rasterization may differ by one level on isolated pixels.
                    for k in (0, n//2, n-1):
                        diff = np.abs(r.render_observation(ids[k], ids[max(0, k-1)]).astype(int) - images[k])
                        max_diff = max(max_diff, int(diff.max())); max_fraction = max(max_fraction, float((diff > 0).mean()))
                        assert diff.max() <= 2 and (diff > 0).mean() < 1e-3, (name, k, diff.max())
                        rendered += 1
                finally:
                    r.close()
    result = {'raw_files_hash_checked': sum(len(e['files']) for e in manifest['episodes']),
              'episodes': len(train | valid), 'train': len(train), 'valid': len(valid), 'rendered_frames_checked': rendered,
              'render_max_abs_diff': max_diff, 'render_max_mismatch_fraction': max_fraction}
    print(json.dumps(result))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    b = sub.add_parser('build', help='render a new HDF5 from raw/ (never overwrites)')
    b.add_argument('output', type=Path); b.add_argument('--validation-fraction', type=float, default=.1)
    b.add_argument('--report', type=Path, help='report path (default: <output>.report.json)')
    v = sub.add_parser('verify', help='check hashes and arrays of an HDF5 against raw/')
    v.add_argument('output', type=Path, nargs='?', default=PUBLISHED); v.add_argument('--render-check', action='store_true')
    m = sub.add_parser('manifest', help='write manifest.json from headset SHA256 receipts')
    m.add_argument('device_hashes', type=Path)
    a = p.parse_args()
    if a.command == 'build':
        build(a.output, a.validation_fraction, a.report)
    elif a.command == 'verify':
        verify(a.output, a.render_check)
    else:
        write_manifest(a.device_hashes)
