"""Export finalized Quest task episodes to an image imitation-learning HDF5.

Images are reconstructed from the archived MuJoCo model, state and camera;
these are simulation observations, not Quest screenshots. Actions are the
recorded 35-value tracker reference, two grips, and virtual camera yaw/pitch.
No policy inference or synthetic demonstrations are substituted for recordings.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile
import zipfile

import numpy as np

SCHEMA = 'furry-visual-il-v1'
SCENES = {'cup', 'wb_dex', 'bimanual', 'basket', 'push_t'}
ACTION_DIM = 39


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def table(path):
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        names = reader.fieldnames
        rows = list(reader)
    if not rows:
        raise ValueError(f'Empty stream: {Path(path).name}')
    values = np.asarray([[float(r[k]) for k in names] for r in rows], np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f'Nonfinite stream: {Path(path).name}')
    return {k: values[:, i] for i, k in enumerate(names)}


def columns(t, prefix, count):
    return np.stack([t[f'{prefix}_{i}'] for i in range(count)], axis=1)


def inspect_episode(path, allow_fixtures=False):
    """Validate raw streams before allocating a renderer or writing a dataset."""
    path = Path(path)
    done = read_json(path/'complete.json')
    meta = read_json(path/'manifest.json')
    session = read_json(path/'session_config.json')
    capture = read_json(path/'learning.json')
    result = read_json(path/'learning-result.json')
    if session.get('fixture') and not allow_fixtures:
        raise ValueError('Synthetic test fixture is not a human demonstration')
    if capture['schema'] != 'furry-visual-capture-v1' or capture['action_schema'] != 'body35_grip2_gaze2':
        raise ValueError('Unsupported capture/action schema')
    if session['capture'] != 'human_and_robot' or session['scene'] not in SCENES:
        raise ValueError('This scene/capture has no supported task-success criterion')
    if meta.get('policy_backend') == 'dexweave_body_v1':
        raise ValueError('DexWeave absolute-reference actions need a separate dataset adapter')
    if result['outcome'] != 1:
        raise ValueError(f"Task did not succeed (outcome={result['outcome']})")
    model_path = path/'learning-model.mjb'
    if digest(model_path) != capture['model_sha256']:
        raise ValueError('Archived model checksum mismatch')
    f, a, c = [table(path/name) for name in ('frames.csv', 'mimic.csv', 'learning.csv')]
    n = len(f['step'])
    if n != done['frame_rows'] or n != result['frames'] or not np.array_equal(c['frame_index'], np.arange(n)):
        raise ValueError('Frame count mismatch')
    for stream in (a, c):
        for key in ('sequence', 'step', 'sim_time'):
            if not np.array_equal(f[key], stream[key]):
                raise ValueError(f'Unsynchronized stream: {key}')
    scene = table(path/'scene_state.csv') if (path/'scene_state.csv').exists() else None
    if scene:
        for key in ('sequence', 'step', 'sim_time'):
            if not np.array_equal(scene[key], f[key]):
                raise ValueError('Unsynchronized scene state')
    good = (f['apply_reference'] == 1) & (f['valid'] == 1) & (c['camera_valid'] == 1)
    candidates = np.flatnonzero(good)
    if not len(candidates):
        raise ValueError('No valid first-person controlled frames')
    start = candidates[0]
    terminal = np.flatnonzero(c['outcome'] != 0)
    end = int(terminal[0]) if len(terminal) else n - 1
    if end <= start or c['outcome'][start] != 0:
        raise ValueError('Recording starts after the task ended')
    sl = slice(start, end+1)
    if not good[sl].all():
        raise ValueError('Tracking/control gap inside the demonstration')
    if len(np.unique(c['segment'][sl])) != 1 or np.any(f['reset'][sl]):
        raise ValueError('Reset, recalibration or pause inside the demonstration')
    times = f['sim_time'][sl]
    if np.any(np.diff(times) <= 0) or np.max(np.diff(times)) > .021:
        raise ValueError('Nonmonotonic or missing simulation frames')
    if times[-1] - times[0] < 1:
        raise ValueError('Less than one second of task demonstration')
    if scene and 'userdata_4' in scene and (scene['userdata_4'][start] != 0 or
                                            (session['scene'] == 'bimanual' and scene['userdata_5'][start] != 0)):
        raise ValueError('Recording began after the object was lifted')
    if np.max(np.abs(np.linalg.norm(np.stack([c['camera_q'+k] for k in ('w','x','y','z')], 1), axis=1)-1)) > 1e-5:
        raise ValueError('Invalid camera quaternion')
    actions = np.column_stack((columns(a, 'mimic', 35), f['grip_left'], f['grip_right'], c['gaze_yaw'], c['gaze_pitch']))
    if np.any(actions[:, 35:37] < 0) or np.any(actions[:, 35:37] > 1):
        raise ValueError('Grip outside [0,1]')
    # Fixed-rate causal sampling: never pick a future state/action.
    grid = times[0] + np.arange(int(np.floor((times[-1]-times[0])*30))+1)/30
    indices = start + np.searchsorted(times, grid + 1e-9, side='right') - 1
    if np.max(grid-f['sim_time'][indices]) > .021:
        raise ValueError('Resampling exceeded the capture interval')
    # All windows/splits are whole-episode based. Content duplicates cannot leak.
    identity = hashlib.sha256(bytes.fromhex(digest(path/'frames.csv')) + bytes.fromhex(digest(path/'mimic.csv'))).hexdigest()
    family = {k: meta.get(k) for k in ('policy_id','policy_backend','policy_sha256','policy_secondary_sha256','input_patch','catch_up')}
    family.update(scene=session['scene'], model_sha256=capture['model_sha256'], camera_mode=capture['camera_mode'])
    return dict(path=path, meta=meta, capture=capture, frames=f, camera=c, scene=scene,
                actions=actions, indices=indices, grid=grid, identity=identity, family=family,
                start=start, end=end)


class EpisodeRenderer:
    def __init__(self, episode):
        import mujoco
        self.mj = mujoco
        self.episode = episode
        cap = episode['capture']
        if mujoco.__version__ != cap['mujoco_version']:
            raise ValueError(f"Install mujoco=={cap['mujoco_version']} for this model")
        self.model = mujoco.MjModel.from_binary_path(str(episode['path']/'learning-model.mjb'))
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=360, width=640)
        self.camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, 'ego')
        self.qpos = columns(episode['frames'], 'qpos', self.model.nq)
        self.qvel = columns(episode['frames'], 'qvel', self.model.nv)

    def render(self, index):
        from PIL import Image
        e, m, d, mj = self.episode, self.model, self.data, self.mj
        d.qpos[:] = self.qpos[index]; d.qvel[:] = self.qvel[index]
        if e['scene']:
            for name, count in [('userdata', m.nuserdata), ('mocap_pos', m.nmocap*3), ('mocap_quat', m.nmocap*4)]:
                if count:
                    getattr(d, name).flat[:] = [e['scene'][f'{name}_{k}'][index] for k in range(count)]
        mj.mj_forward(m, d)
        c = e['camera']
        # Set a fixed world camera: archived pose already includes virtual pan/tilt.
        pos = np.array([c['camera_'+k][index] for k in 'xyz'])
        quat = np.array([c['camera_q'+k][index] for k in ('w','x','y','z')])
        matrix = np.zeros(9); mj.mju_quat2Mat(matrix, quat); matrix = matrix.reshape(3,3)
        self.renderer.update_scene(d, camera=self.camera_id)
        for cam in self.renderer.scene.camera:
            cam.pos[:] = pos
            cam.forward[:] = -matrix[:,2]
            cam.up[:] = matrix[:,1]
        rgb = self.renderer.render()
        return np.asarray(Image.fromarray(rgb).resize((224,224), Image.Resampling.BILINEAR))

    def close(self):
        self.renderer.close()


def export(paths, output, validation_fraction=.1, allow_fixtures=False):
    import h5py
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    report = {'schema':SCHEMA, 'accepted':[], 'rejected':[], 'fps':30, 'independent_test_session':False, 'allow_test_fixtures':allow_fixtures}
    episodes, seen, family = [], set(), None
    for path in sorted(map(Path, paths)):
        try:
            e = inspect_episode(path, allow_fixtures)
            if e['identity'] in seen:
                raise ValueError('Duplicate recording content')
            if family is not None and e['family'] != family:
                raise ValueError('Different scene/model/policy configuration; export to a separate dataset')
            seen.add(e['identity']); family = e['family']; episodes.append(e)
        except (ValueError, KeyError, OSError) as exc:
            report['rejected'].append({'path':str(path), 'reason':str(exc)})
    output.parent.mkdir(parents=True, exist_ok=True)
    if not episodes:
        output.with_suffix('.report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
        raise ValueError('No accepted successful episodes; see rejection report')
    valid_count = min(len(episodes)-1, max(1, round(len(episodes)*validation_fraction))) if len(episodes)>1 and validation_fraction else 0
    valid_ids = set(sorted(range(len(episodes)), key=lambda i: episodes[i]['identity'])[:valid_count])
    temp = output.with_suffix('.partial.hdf5')
    try:
        with h5py.File(temp, 'x') as h:
            h.attrs['schema'] = SCHEMA; h.attrs['fps'] = 30
            h.attrs['action_schema'] = 'body35_grip2_gaze2'
            h.attrs['family'] = json.dumps(family)
            group = h.create_group('data'); masks = {'train':[], 'valid':[]}; total=0
            group.attrs['env_args'] = json.dumps({'env_name':'FURRYQuest', 'type':2, 'env_kwargs':family})
            for j, e in enumerate(episodes):
                ids = e['indices']; n = len(ids); name=f'demo_{j}'
                demo=group.create_group(name); demo.attrs['num_samples']=n
                demo.attrs['source_episode']=str(e['path']); demo.attrs['source_sha256']=e['identity']
                demo.attrs['manifest']=json.dumps(e['meta']); demo.attrs['capture']=json.dumps(e['capture'])
                demo.create_dataset('actions',data=e['actions'][ids].astype('float32'))
                obs=demo.create_group('obs')
                # Strictly previous sampled command; zero sentinel at episode start.
                previous=np.zeros((n,ACTION_DIM),np.float32); previous[1:]=e['actions'][ids[:-1]]
                obs.create_dataset('previous_command',data=previous)
                images=obs.create_dataset('agentview_image',(n,224,224,3),dtype='uint8',chunks=(1,224,224,3),compression='lzf')
                renderer=EpisodeRenderer(e)
                try:
                    for k, idx in enumerate(ids): images[k]=renderer.render(idx)
                finally: renderer.close()
                demo.create_dataset('source_frame_index',data=ids)
                demo.create_dataset('source_sim_time',data=e['frames']['sim_time'][ids])
                demo.create_dataset('timestamps',data=e['grid'])
                demo.create_dataset('rewards',data=np.r_[np.zeros(n-1),1].astype('float32'))
                demo.create_dataset('dones',data=np.r_[np.zeros(n-1),1].astype('uint8'))
                split='valid' if j in valid_ids else 'train'; masks[split].append(name.encode())
                total+=n; report['accepted'].append({'path':str(e['path']),'demo':name,'samples':n,'split':split,'sha256':e['identity']})
            group.attrs['total']=total
            for split,names in masks.items(): h.create_dataset('mask/'+split,data=np.asarray(names,dtype='S32'))
            h.attrs['report']=json.dumps(report)
        temp.rename(output)
    finally:
        if temp.exists(): temp.unlink()
    report['dataset_sha256']=digest(output); report['family']=family
    output.with_suffix('.report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    return report


def unpack_archives(source, destination):
    """Extract only regular files under an episode root, never trust ZIP paths."""
    for i, archive in enumerate(sorted(Path(source).glob('*.zip'))):
        folder=Path(destination)/str(i); folder.mkdir()
        with zipfile.ZipFile(archive) as z:
            for info in z.infolist():
                target=(folder/info.filename).resolve()
                if not target.is_relative_to(folder.resolve()) or (info.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError('Unsafe ZIP member')
            z.extractall(folder)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',type=Path,help='Episode folder, recordings folder, or folder of exported ZIPs')
    p.add_argument('output',type=Path,help='New .hdf5 file, one scene/policy per dataset')
    p.add_argument('--validation-fraction',type=float,default=.1)
    p.add_argument('--allow-test-fixtures', action='store_true', help=argparse.SUPPRESS)
    args=p.parse_args()
    if not 0<=args.validation_fraction<1: p.error('validation fraction must be in [0,1)')
    with tempfile.TemporaryDirectory(prefix='furry-dataset-') as temporary:
        paths=[args.source] if (args.source/'manifest.json').is_file() else [x.parent for x in args.source.rglob('manifest.json')]
        unpack_archives(args.source,temporary)
        paths += [x.parent for x in Path(temporary).rglob('manifest.json')]
        report=export(paths,args.output,args.validation_fraction,args.allow_test_fixtures)
        print(json.dumps({'dataset':str(args.output),'accepted':len(report['accepted']),'rejected':report['rejected']},ensure_ascii=False,indent=2))


if __name__=='__main__': main()
