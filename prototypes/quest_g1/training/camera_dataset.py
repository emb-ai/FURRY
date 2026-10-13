"""Camera-leg episodes -> quest-g1-motion-v1 clips + paired actor command bank.

Reward target: cleaned camera legs (camera_targets.py) re-solved by native GMR
(camera_commands with a body override). Actor input: the recorded live
mimic.csv, i.e. what the policy actually saw, on the same 100 Hz sim clock.
Validation holds whole episodes. Clip processing follows prepare.py: 100 Hz,
6 Hz zero-phase Butterworth, joint limits, sole floor guard, <=30 s pieces.

Per episode PREFIX (from camera_commands) this reads:
  PREFIX.pose.csv / PREFIX.mimic.csv     raw camera replay (camera flag, check)
  PREFIX.meta.mimic.csv                  Meta-leg replay (live camera state)
  PREFIX.TARGET.pose.csv                 cleaned target replay
Usage: camera_dataset.py out --assets android/assets --pair EPISODE_DIR:PREFIX ...
       [--validation EPISODE_ID ...] [--target clean3]
"""
import argparse, hashlib, json, pickle
from pathlib import Path
import numpy as np
import mujoco
from scipy.signal import butter, sosfiltfilt
from scipy.spatial.transform import Rotation, Slerp
from motion_coordinates import world_offsets_to_root

LEGS = slice(9, 21)  # mimic leg joints
START_S = .6  # live startup blend
# Live vs replayed camera legs. With the camera on in both, differences come
# from the registration (1-7 deg median) and act as input noise. When only
# one side has camera legs, the other side holds Meta legs (20-70 deg apart),
# so the target or the actor lacks the real leg motion: exclude.
MISMATCH_DEG, CAMERA_ON_DEG = 15., 1.
SPEED_CEILING = {'leg_torso': 12., 'arm': 16.}  # rad/s, as dataset-v2 curation
MARGIN_S, MIN_CLIP_S, MAX_CLIP = .25, 2., 3000


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def table(path):
    with open(path) as f:
        names = f.readline().strip().split(',')
    a = np.loadtxt(path, delimiter=',', skiprows=1, ndmin=2)
    return a, {n: i for i, n in enumerate(names)}


def regions(mask):
    ids = np.flatnonzero(mask)
    return [r for r in np.split(ids, np.flatnonzero(np.diff(ids) > 1) + 1) if len(r)]


def ticks(episode, prefix, target):
    """Per-tick arrays on the shared sim clock, with the usable-tick mask."""
    live, lc = table(episode / 'mimic.csv')
    raw, rc = table(f'{prefix}.pose.csv')
    rmim, _ = table(f'{prefix}.mimic.csv')
    mmim, _ = table(f'{prefix}.meta.mimic.csv')
    tgt, tc = table(f'{prefix}.{target}.pose.csv')
    frames, fc = table(episode / 'frames.csv')
    # camera_commands writes no pose for the first (pre-reference) tick.
    live, rmim, mmim, frames = live[1:], rmim[1:], mmim[1:], frames[1:]
    for a, cols in ((raw, [0, 1]), (rmim, [0, 1]), (mmim, [0, 1]), (tgt, [0, 1]), (frames, [fc['sequence'], fc['step']])):
        if len(a) != len(live) or not np.array_equal(a[:, cols], live[:, :2]):
            raise ValueError(f'Tick mismatch in {prefix}')
    t = live[:, 2]
    if not (np.diff(t) > 0).all():
        raise ValueError('Sim clock must increase (resets are not supported)')
    cmd = live[:, 3:]
    legs = lambda a: np.degrees(np.abs(cmd[:, LEGS] - a[:, 3:][:, LEGS])).max(1)
    mismatch = legs(rmim)
    live_on, replay_on = legs(mmim) > CAMERA_ON_DEG, raw[:, rc['camera_weight']] > .01
    bad = (live_on != replay_on) | (mismatch > MISMATCH_DEG)
    k = int(MARGIN_S / np.median(np.diff(t)))
    bad = np.convolve(bad, np.ones(2 * k + 1), 'same') > 0
    ok = (frames[:, fc['apply_reference']] == 1) & (frames[:, fc['valid']] == 1) & (t - t[0] >= START_S) & ~bad
    q = tgt[:, [tc[f'q{k}'] for k in range(36)]]
    return dict(t=t, seq=tgt[:, tc['sequence']].astype(np.int64), q=q, cmd=cmd, ok=ok,
                camera=raw[:, rc['camera_weight']], mismatch=mismatch)


def resample(t, seq, q):
    """First tick of each body sequence -> 100 Hz grid (prepare.py style)."""
    first = np.r_[True, np.diff(seq) != 0]
    bt, q = t[first], q[first]
    grid = np.arange(bt[0], bt[-1], .01)
    new = np.empty((len(grid), 36))
    for k in [0, 1, 2] + list(range(7, 36)):
        new[:, k] = np.interp(grid, bt, q[:, k])
    new[:, 3:7] = Slerp(bt, Rotation.from_quat(q[:, [4, 5, 6, 3]]))(grid).as_quat()[:, [3, 0, 1, 2]]
    cols = np.r_[0:3, 7:36]
    new[:, cols] = sosfiltfilt(butter(2, 6, fs=100, output='sos'), new[:, cols], axis=0)
    return grid, new


class Robot:
    def __init__(self, assets):
        self.model = mujoco.MjModel.from_xml_path(str(assets / 'gmr_model.xml')); self.fk = mujoco.MjData(self.model)
        self.phys = mujoco.MjModel.from_xml_path(str(assets / 'scene.xml')); self.pd = mujoco.MjData(self.phys)
        self.qadr = [self.phys.joint(self.model.joint(j).name).qposadr[0] for j in range(1, self.model.njnt)]
        # Sole shapes that collide with the floor (spheres since the pad change).
        ph, floor = self.phys, self.phys.geom('floor').id
        feet = [ph.body(f'{s}_ankle_roll_link').id for s in ['left', 'right']]
        self.pads = [g for g in range(ph.ngeom) if ph.geom_bodyid[g] in feet and
                     (ph.geom_contype[g] & ph.geom_conaffinity[floor] or ph.geom_contype[floor] & ph.geom_conaffinity[g])]
        if not self.pads or any(ph.geom_type[g] not in (mujoco.mjtGeom.mjGEOM_SPHERE, mujoco.mjtGeom.mjGEOM_CAPSULE) for g in self.pads):
            raise ValueError('Unsupported sole geometry')
        self.links = [self.model.body(j).name for j in range(1, self.model.nbody)]
        names = [self.model.joint(j).name for j in range(1, self.model.njnt)]
        self.arm = np.array(['shoulder' in n or 'elbow' in n or 'wrist' in n for n in names])

    def finish(self, new):
        """Joint limits, sole floor guard, world-axis local body positions."""
        for j in range(1, self.model.njnt):
            k = self.model.jnt_qposadr[j]; new[:, k] = np.clip(new[:, k], *self.model.jnt_range[j])
        lifts, local = [], []
        for row in new:
            self.pd.qpos[:7] = row[:7]; self.pd.qpos[self.qadr] = row[7:]; mujoco.mj_kinematics(self.phys, self.pd)
            low = min(self.pd.geom_xpos[g, 2] - self.phys.geom_size[g, 0] - (self.phys.geom_size[g, 1] * abs(self.pd.geom_xmat[g, 8])
                      if self.phys.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE else 0.) for g in self.pads)
            lift = max(0., -low); row[2] += lift; lifts.append(lift)
            self.fk.qpos[:] = row; mujoco.mj_kinematics(self.model, self.fk); local.append(self.fk.xpos[1:].copy() - row[:3])
        return np.asarray(lifts), np.asarray(local)

    def too_fast(self, new):
        speed = np.abs(np.gradient(new[:, 7:], axis=0)) * 100
        bad = (speed[:, ~self.arm] > SPEED_CEILING['leg_torso']).any(1) | (speed[:, self.arm] > SPEED_CEILING['arm']).any(1)
        k = int(MARGIN_S * 100)
        return np.convolve(bad, np.ones(2 * k + 1), 'same') > 0


def commands(d, t0, t1):
    """Live commands as a causal zero-order hold, times relative to t0."""
    first = np.searchsorted(d['t'], t0, side='right') - 1
    last = np.searchsorted(d['t'], t1, side='right')
    rel = d['t'][first:last] - t0; rel[0] = 0.
    return np.column_stack((rel, d['cmd'][first:last]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('output', type=Path); p.add_argument('--assets', type=Path, required=True)
    p.add_argument('--pair', nargs='+', required=True); p.add_argument('--validation', nargs='*', default=[])
    p.add_argument('--target', default='clean3')
    a = p.parse_args()
    out = a.output.resolve()
    if out.exists():
        raise ValueError('Use a fresh output path')
    robot = Robot(a.assets)
    manifest = {'schema': 'quest-g1-motion-v1', 'fps': 100, 'source': 'quest-camera',
                'split_unit': 'whole episode', 'validation_episodes': a.validation,
                'reward_target': f'camera legs cleaned by camera_targets.py ({a.target}), native GMR via camera_commands body override',
                'actor_input': 'recorded live mimic.csv (camera fusion as run on the headset)',
                'filter': '6 Hz zero-phase Butterworth, order 2 on positions and joint angles; quaternion SLERP; floor guard reapplied after smoothing',
                'curation': dict(speed_ceiling_rad_s=SPEED_CEILING, exclude_margin_s=MARGIN_S, min_clip_s=MIN_CLIP_S,
                                 startup_s=START_S, live_replay_leg_mismatch_deg=MISMATCH_DEG),
                'episodes': {}, 'clips': []}
    arrays, files = {}, {}
    for pair in a.pair:
        episode, prefix = pair.split(':', 1); episode = Path(episode)
        eid = episode.name.replace('episode-', '')
        split = 'validation' if eid in a.validation else 'train'
        d = ticks(episode, prefix, a.target)
        manifest['episodes'][eid] = {'split': split, 'ticks': int(len(d['t'])), 'usable_share': float(d['ok'].mean()),
                                     'camera_share': float((d['camera'] > .5).mean()),
                                     'source_sha256': {n: sha(episode / n) for n in ['mimic.csv', 'frames.csv', 'camera.jsonl', 'body.csv']},
                                     'target_sha256': sha(f'{prefix}.{a.target}.pose.csv')}
        for run, ids in enumerate(regions(d['ok'])):
            if d['t'][ids[-1]] - d['t'][ids[0]] < MIN_CLIP_S:
                continue
            grid, new = resample(d['t'][ids], d['seq'][ids], d['q'][ids])
            cam = np.interp(grid, d['t'][ids], d['camera'][ids])
            for part, keep in enumerate(regions(~robot.too_fast(new))):
                if len(keep) < MIN_CLIP_S * 100:
                    continue
                q = new[keep].copy(); lifts, local = robot.finish(q)
                for start in range(0, len(q) - 199, MAX_CLIP):
                    stop = min(start + MAX_CLIP, len(q))
                    clipq = q[start:stop].copy(); clipq[:, :2] -= clipq[0, :2]
                    cid = f'{eid}-run{run:02d}-part{part:02d}-{start:06d}'
                    path = out / split / f'{cid}.npz'; path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(path, qpos=clipq, fps=100, local_body_pos=local[start:stop], link_body_list=robot.links)
                    payload = {'fps': 100, 'root_pos': clipq[:, :3], 'root_rot': clipq[:, [4, 5, 6, 3]], 'dof_pos': clipq[:, 7:],
                               'local_body_pos': world_offsets_to_root(local[start:stop], clipq[:, [4, 5, 6, 3]]), 'link_body_list': robot.links}
                    with path.with_suffix('.pkl').open('wb') as f:
                        pickle.dump(payload, f, protocol=4)
                    t0 = grid[keep[start]]; seconds = (stop - start - 1) / 100
                    vel = np.diff(clipq[:, 7:], axis=0) * 100
                    clip = {'id': cid, 'file': str(path.relative_to(out)), 'split': split, 'source': 'quest-camera', 'episode': eid,
                            'stage': 0, 'source_start_sim_s': float(t0), 'seconds': float(seconds),
                            'camera_share': float((cam[keep[start]:keep[start] + stop - start] > .5).mean()),
                            'max_joint_speed_rad_s': float(abs(vel).max()), 'p99_joint_speed_rad_s': float(np.quantile(abs(vel), .99)),
                            'max_floor_lift_m': float(lifts[start:stop].max()), 'sha256': sha(path)}
                    manifest['clips'].append(clip)
                    if split == 'train':
                        key = 'motion_%04d' % len(arrays)
                        arrays[key] = commands(d, t0, t0 + seconds)
                        files[clip['file']] = {'key': key, 'target_sha256': clip['sha256'], 'split': 'train', 'source': 'quest-camera',
                                               'frames': len(arrays[key]), 'seconds': clip['seconds'], 'parent_id': None, 'stage': 0}
                    print(cid, split, f"{seconds:.1f}s camera {clip['camera_share']:.2f}", flush=True)
    manifest['duration_seconds'] = {s: sum(c['seconds'] for c in manifest['clips'] if c['split'] == s) for s in ['train', 'validation']}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    for split in ['train', 'validation']:
        lines = ['root_path: ' + str(out), 'motions:']
        for c in manifest['clips']:
            if c['split'] == split:
                lines += ['- file: ' + str(Path(c['file']).with_suffix('.pkl')), '  weight: ' + str(c['seconds'])]
        (out / f'{split}.yaml').write_text('\n'.join(lines) + '\n')
    bank = out / 'command_bank.npz'
    np.savez_compressed(bank, **arrays)
    bank.with_suffix('.json').write_text(json.dumps({
        'schema': 'quest-camera-command-bank-v1', 'npz_sha256': sha(bank), 'dataset_manifest_sha256': sha(out / 'manifest.json'),
        'files': files, 'layout': 'per clip [N, 36]: time from clip start (s), then the 35-dim mimic command',
        'command_sampling': 'causal zero-order hold at recorded 100 Hz sim ticks; no future interpolation',
        'reward_target': manifest['reward_target'], 'actor_input': manifest['actor_input'],
        'runtime_match': {'camera_fusion': True, 'catch_up': 'as recorded', 'startup_blend': False},
        'limitations': ['Single operator and a few sessions.',
                        'Cleaned targets are not validated against markers yet (AprilTag session pending).',
                        'No mirrored augmentation.']}, indent=2) + '\n')
    print(json.dumps(manifest['duration_seconds']), 'bank clips', len(files), flush=True)


if __name__ == '__main__':
    main()
