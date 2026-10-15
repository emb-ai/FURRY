"""Verify the published cup data, rebuild causal RGB, or export an episode MP4."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np

DATA = Path(__file__).resolve().parents[1]
REPO = DATA.parents[2]
sys.path.insert(0, str(REPO / 'prototypes/quest_g1'))
from g1_sim.visual_dataset import inspect_episode, digest, SCHEMA
from training_renderer import TrainingRenderer


def selected(episode=None):
    plan = json.loads((DATA / 'training-manifest.json').read_text())
    for split, rows in plan['splits'].items():
        for row in rows:
            if episode and row['episode'] != episode:
                continue
            source = inspect_episode(DATA / 'raw' / row['episode'])
            if source['identity'] != row['identity']:
                raise ValueError('Episode identity changed: ' + row['episode'])
            if source['family'] != plan['model_families'][row['family_id']]['family']:
                raise ValueError('Episode model/policy family changed')
            yield split, row, source


def rebuild(output, episode=None):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix('.partial.hdf5')
    if partial.exists():
        raise FileExistsError(partial)
    masks = {'train': [], 'valid': []}
    total = 0
    try:
        with h5py.File(partial, 'x') as h:
            h.attrs.update(schema=SCHEMA, fps=30, action_schema='body35_grip2_gaze2')
            h.attrs['renderer'] = json.dumps(json.loads((DATA / 'dataset-report.json').read_text())['renderer'])
            data = h.create_group('data')
            for j, (split, row, e) in enumerate(selected(episode)):
                ids = e['indices']; n = len(ids); name = f'demo_{j}'
                d = data.create_group(name)
                d.attrs.update(num_samples=n, episode=row['episode'], family=json.dumps(e['family']),
                               capture=json.dumps(e['capture']), manifest=json.dumps(e['meta']))
                d.create_dataset('actions', data=e['actions'][ids].astype('float32'))
                previous = np.zeros((n, 39), np.float32)
                previous[1:] = e['actions'][ids[:-1]]
                obs = d.create_group('obs')
                obs.create_dataset('previous_command', data=previous)
                images = obs.create_dataset('agentview_image', (n,224,224,3), dtype='uint8',
                                            chunks=(1,224,224,3), compression='lzf')
                renderer = TrainingRenderer(e)
                renderer.model.light_castshadow[:] = 0
                try:
                    for k, idx in enumerate(ids):
                        images[k] = renderer.render_observation(idx, ids[max(0,k-1)])
                finally:
                    renderer.close()
                d.create_dataset('source_frame_index', data=ids)
                d.create_dataset('timestamps', data=e['grid'])
                d.create_dataset('source_sim_time', data=e['frames']['sim_time'][ids])
                masks[split].append(name.encode()); total += n
                print(row['episode'], split, n, flush=True)
            if not total:
                raise ValueError('No selected episode in the published training manifest')
            for split, names in masks.items():
                h.create_dataset('mask/' + split, data=np.asarray(names, dtype='S32'))
        partial.rename(output)
    finally:
        partial.unlink(missing_ok=True)
    print(json.dumps({'output':str(output), 'frames':total, 'sha256':digest(output)}))


def video(episode, output, recorded_gaze=False):
    """Stored training RGB, or full-resolution archived camera for any raw episode."""
    from g1_sim.visual_dataset import EpisodeRenderer, table
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    renderer = None; h = None; process = None
    try:
        if recorded_gaze:
            root = DATA / 'raw' / episode
            capture = json.loads((root / 'learning.json').read_text())
            assert digest(root / 'learning-model.mjb') == capture['model_sha256']
            frames = table(root / 'frames.csv'); camera = table(root / 'learning.csv')
            scene = table(root / 'scene_state.csv')
            for stream in (camera, scene):
                for key in ('sequence', 'step', 'sim_time'):
                    np.testing.assert_array_equal(stream[key], frames[key])
            e = dict(path=root, capture=capture, frames=frames, camera=camera, scene=scene)
            good = np.flatnonzero((camera['camera_valid'] == 1) & (frames['valid'] == 1))
            if not len(good): raise ValueError('No valid archived camera frames')
            times = frames['sim_time'][good]
            breaks = np.flatnonzero((np.diff(good) != 1) | (np.diff(times) <= 0) |
                                    (np.diff(times) > .021) | (np.diff(camera['segment'][good]) != 0)) + 1
            indices = []
            for run in np.split(good, breaks):
                t = frames['sim_time'][run]
                grid = t[0] + np.arange(int(np.floor((t[-1]-t[0])*30))+1)/30
                indices.extend(run[np.searchsorted(t,grid+1e-9,side='right')-1])
            renderer = EpisodeRenderer(e); renderer.model.light_castshadow[:] = 0
            def images():
                for index in indices:
                    renderer.render(index)
                    yield renderer.renderer.render()
            stream = images(); size = '640x360'
        else:
            h = h5py.File(DATA / 'rendered/cup44-no-shadow-causal.hdf5', 'r')
            demos = [d for d in h['data'].values() if d.attrs['episode'] == episode]
            if not demos:
                raise ValueError('Not in the 44-episode RGB dataset; use --recorded-gaze for raw episodes')
            stream = demos[0]['obs/agentview_image']; size = '224x224'
        process = subprocess.Popen(['ffmpeg','-nostdin','-n','-v','error','-f','rawvideo',
            '-pix_fmt','rgb24','-s',size,'-r','30','-i','-','-an','-c:v','libx264',
            '-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(output)], stdin=subprocess.PIPE)
        for rgb in stream:
            process.stdin.write(np.ascontiguousarray(rgb).tobytes())
        process.stdin.close()
        if process.wait(): raise RuntimeError('ffmpeg failed')
    finally:
        if renderer: renderer.close()
        if h: h.close()
        if process and process.poll() is None:
            process.terminate(); process.wait()
    print(output)


def verify(render_check=False):
    manifest = json.loads((DATA / 'manifest.json').read_text())
    for item in manifest['files']:
        p = DATA / item['path']
        if p.stat().st_size != item['bytes'] or digest(p) != item['sha256']:
            raise ValueError('Checksum mismatch / missing Git LFS object: ' + item['path'])
    rendered = 0; families = set(); total = 0
    with h5py.File(DATA / 'rendered/cup44-no-shadow-causal.hdf5', 'r') as h:
        train = {x.decode() for x in h['mask/train'][:]}
        valid = {x.decode() for x in h['mask/valid'][:]}
        assert len(train) == 40 and len(valid) == 4 and not train & valid
        assert train | valid == set(h['data'])
        for j, (split, row, e) in enumerate(selected()):
            name = f'demo_{j}'; d = h['data'][name]; ids = e['indices']; n = len(ids)
            assert name in (train if split == 'train' else valid)
            assert d.attrs['episode'] == row['episode']
            np.testing.assert_array_equal(d['actions'][:], e['actions'][ids].astype('float32'))
            np.testing.assert_array_equal(d['source_frame_index'][:], ids)
            np.testing.assert_array_equal(d['timestamps'][:], e['grid'])
            np.testing.assert_array_equal(d['source_sim_time'][:], e['frames']['sim_time'][ids])
            previous = d['obs/previous_command'][:]
            np.testing.assert_array_equal(previous[1:], d['actions'][:-1])
            assert not previous[0].any()
            images = d['obs/agentview_image']
            assert images.shape == (n,224,224,3) and images.dtype == np.uint8
            for first in range(0,n,32):
                assert images[first:first+32].std() > 1
            key = (row['family_id'], split)
            if render_check and key not in families:
                families.add(key); renderer = TrainingRenderer(e)
                renderer.model.light_castshadow[:] = 0
                try:
                    for k in (0,n//2,n-1):
                        rgb = renderer.render_observation(ids[k],ids[max(0,k-1)])
                        np.testing.assert_array_equal(rgb, images[k])
                        rendered += 1
                finally:
                    renderer.close()
            total += n
        assert total == 12982
    print(json.dumps({'verified_files':len(manifest['files']), 'episodes':44,
                      'frames':total, 'rendered_frames_exact':rendered}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    subs = p.add_subparsers(dest='command', required=True)
    v = subs.add_parser('verify'); v.add_argument('--render-check', action='store_true')
    r = subs.add_parser('rebuild'); r.add_argument('output', type=Path); r.add_argument('--episode')
    m = subs.add_parser('video'); m.add_argument('episode'); m.add_argument('output', type=Path)
    m.add_argument('--recorded-gaze', action='store_true', help='Archived same-frame camera pose; not training observations')
    args = p.parse_args()
    if args.command == 'verify': verify(args.render_check)
    elif args.command == 'rebuild': rebuild(args.output, args.episode)
    else: video(args.episode, args.output, args.recorded_gaze)


if __name__ == '__main__':
    main()
