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
Each episode is its own stage (--stage-base + pair index), so validation is
a whole-stage holdout as author_dataset.py expects. --mirror adds sagittal
reflections of TRAIN clips with mirrored commands (not new data).
Usage: camera_dataset.py out --assets android/assets --pair EPISODE_DIR:PREFIX ...
       [--validation EPISODE_ID ...] [--target clean3] [--mirror]
"""
import argparse, hashlib, json, pickle
from pathlib import Path
import numpy as np
import mujoco
from scipy.signal import butter, sosfiltfilt
from scipy.spatial.transform import Rotation, Slerp
from motion_coordinates import world_offsets_to_root

LEGS = slice(6, 18)  # leg joints in the 35-dim command (6 root terms, then 29 joints)
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
        self.joints = names
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

    def limit_excess(self, q):
        """Largest joint-limit violation (rad) before clipping."""
        lo, hi = self.model.jnt_range[1:].T
        return float(np.max(np.maximum(lo - q[:, 7:], q[:, 7:] - hi)).clip(0))

    def too_fast(self, new):
        speed = np.abs(np.gradient(new[:, 7:], axis=0)) * 100
        bad = (speed[:, ~self.arm] > SPEED_CEILING['leg_torso']).any(1) | (speed[:, self.arm] > SPEED_CEILING['arm']).any(1)
        k = int(MARGIN_S * 100)
        return np.convolve(bad, np.ones(2 * k + 1), 'same') > 0


def opposite(name):
    return 'right_' + name[5:] if name.startswith('left_') else 'left_' + name[6:] if name.startswith('right_') else name


def reflect(q, joints):
    """G1 sagittal reflection of qpos36 (as mirror_dataset.py)."""
    out = q.copy(); out[:, 1] *= -1; out[:, [4, 6]] *= -1  # S R S, S=diag(1,-1,1), quaternion wxyz
    for i, name in enumerate(joints):
        out[:, 7 + i] = q[:, 7 + joints.index(opposite(name))] * (-1 if 'roll' in name or 'yaw' in name else 1)
    return out


def mirror_command(cmd, joints):
    """[t, 35 cmd]: lateral velocity, roll and yaw rate flip; joints swap sides
    (as build_command_bank.py)."""
    out = cmd.copy(); out[:, [2, 4, 6]] *= -1
    for i, name in enumerate(joints):
        out[:, 7 + i] = cmd[:, 7 + joints.index(opposite(name))] * (-1 if 'roll' in name or 'yaw' in name else 1)
    return out


def commands(d, t0, t1):
    """Live commands as a causal zero-order hold, times relative to t0."""
    first = np.searchsorted(d['t'], t0, side='right') - 1
    last = np.searchsorted(d['t'], t1, side='right')
    rel = d['t'][first:last] - t0; rel[0] = 0.
    return np.column_stack((rel, d['cmd'][first:last]))


def write(out, split, cid, q, local, links):
    path = out / split / f'{cid}.npz'; path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, qpos=q, fps=100, local_body_pos=local, link_body_list=links)
    payload = {'fps': 100, 'root_pos': q[:, :3], 'root_rot': q[:, [4, 5, 6, 3]], 'dof_pos': q[:, 7:],
               'local_body_pos': world_offsets_to_root(local, q[:, [4, 5, 6, 3]]), 'link_body_list': links}
    with path.with_suffix('.pkl').open('wb') as f:
        pickle.dump(payload, f, protocol=4)
    vel = np.diff(q[:, 7:], axis=0) * 100
    return {'file': str(path.relative_to(out)), 'max_joint_speed_rad_s': float(abs(vel).max()),
            'p99_joint_speed_rad_s': float(np.quantile(abs(vel), .99)), 'sha256': sha(path)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('output', type=Path); p.add_argument('--assets', type=Path, required=True)
    p.add_argument('--pair', nargs='+', required=True); p.add_argument('--validation', nargs='*', default=[])
    p.add_argument('--target', default='clean3')
    p.add_argument('--stage-base', type=int, default=101, help='first episode stage; keep clear of other Quest stages')
    p.add_argument('--mirror', action='store_true', help='add sagittal reflections of TRAIN clips')
    p.add_argument('--source', default='quest-camera', help='input_source tag; a separate tag gets its own --quest-shares mass')
    a = p.parse_args()
    out = a.output.resolve()
    if out.exists():
        raise ValueError('Use a fresh output path')
    robot = Robot(a.assets)
    stages = {Path(pair.split(':', 1)[0]).name.replace('episode-', ''): a.stage_base + i for i, pair in enumerate(a.pair)}
    if set(a.validation) - set(stages):
        raise ValueError('Validation episode not among --pair')
    manifest = {'schema': 'quest-g1-motion-v1', 'fps': 100, 'source': a.source,
                'split_unit': 'whole episode (one stage per episode)', 'validation_episodes': a.validation,
                'validation_stages': sorted(stages[e] for e in a.validation), 'episode_stages': stages,
                'reward_target': f'camera legs cleaned by camera_targets.py ({a.target}), native GMR via camera_commands body override',
                'actor_input': 'recorded live mimic.csv (camera fusion as run on the headset)',
                'filter': '6 Hz zero-phase Butterworth, order 2 on positions and joint angles; quaternion SLERP; floor guard reapplied after smoothing',
                'curation': dict(speed_ceiling_rad_s=SPEED_CEILING, exclude_margin_s=MARGIN_S, min_clip_s=MIN_CLIP_S,
                                 startup_s=START_S, live_replay_leg_mismatch_deg=MISMATCH_DEG),
                'episodes': {}, 'clips': []}
    arrays, files, mirror_checks = {}, {}, []

    def bank(clip, cmd):
        key = 'motion_%04d' % len(arrays); arrays[key] = cmd
        files[clip['file']] = {'key': key, 'target_sha256': clip['sha256'], 'split': 'train', 'source': a.source,
                               'frames': len(cmd), 'seconds': clip['seconds'], 'parent_id': clip.get('parent_id'),
                               'augmentation': clip.get('augmentation'), 'stage': clip['stage']}

    for pair in a.pair:
        episode, prefix = pair.split(':', 1); episode = Path(episode)
        eid = episode.name.replace('episode-', '')
        split = 'validation' if eid in a.validation else 'train'
        d = ticks(episode, prefix, a.target)
        manifest['episodes'][eid] = {'split': split, 'stage': stages[eid], 'ticks': int(len(d['t'])), 'usable_share': float(d['ok'].mean()),
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
                    t0 = grid[keep[start]]; seconds = (stop - start - 1) / 100
                    clip = {'id': cid, 'split': split, 'source': a.source, 'episode': eid,
                            'stage': stages[eid], 'source_start_sim_s': float(t0), 'seconds': float(seconds),
                            'camera_share': float((cam[keep[start]:keep[start] + stop - start] > .5).mean()),
                            'max_floor_lift_m': float(lifts[start:stop].max()),
                            **write(out, split, cid, clipq, local[start:stop], robot.links)}
                    manifest['clips'].append(clip)
                    print(cid, split, f"{seconds:.1f}s camera {clip['camera_share']:.2f}", flush=True)
                    if split != 'train':
                        continue
                    cmd = commands(d, t0, t0 + seconds); bank(clip, cmd)
                    if not a.mirror:
                        continue
                    mq = reflect(clipq, robot.joints)
                    assert np.allclose(reflect(mq, robot.joints), clipq, atol=1e-12)
                    excess = robot.limit_excess(mq)
                    mlifts, mlocal = robot.finish(mq)
                    mirror = dict(clip, id=cid + '-mirror', augmentation='sagittal_reflection', parent_id=cid,
                                  max_floor_lift_m=float(mlifts.max()), **write(out, split, cid + '-mirror', mq, mlocal, robot.links))
                    manifest['clips'].append(mirror); bank(mirror, mirror_command(cmd, robot.joints))
                    mirror_checks.append({'parent_id': cid, 'max_joint_limit_excess_rad': excess,
                                          'max_floor_lift_m': float(mlifts.max())})
    if a.mirror:
        manifest['augmentation'] = {'type': 'G1 sagittal reflection, train only', 'independent_demonstrations_added': 0,
                                    'augmented_clips': len(mirror_checks), 'checks': mirror_checks,
                                    'note': 'Mirrors are clipped to joint limits and floor-guarded like originals.'}
    manifest['duration_seconds'] = {s: sum(c['seconds'] for c in manifest['clips'] if c['split'] == s) for s in ['train', 'validation']}
    manifest['camera_seconds'] = {s: sum(c['seconds'] * c['camera_share'] for c in manifest['clips'] if c['split'] == s and not c.get('augmentation'))
                                  for s in ['train', 'validation']}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    for split in ['train', 'validation']:
        lines = ['root_path: ' + str(out), 'motions:']
        for c in manifest['clips']:
            if c['split'] == split:
                lines += ['- file: ' + str(Path(c['file']).with_suffix('.pkl')), '  weight: ' + str(c['seconds'])]
        (out / f'{split}.yaml').write_text('\n'.join(lines) + '\n')
    bank_path = out / 'command_bank.npz'
    np.savez_compressed(bank_path, **arrays)
    bank_path.with_suffix('.json').write_text(json.dumps({
        'schema': 'quest-camera-command-bank-v1', 'npz_sha256': sha(bank_path), 'dataset_manifest_sha256': sha(out / 'manifest.json'),
        'files': files, 'layout': 'per clip [N, 36]: time from clip start (s), then the 35-dim mimic command',
        'command_sampling': 'causal zero-order hold at recorded 100 Hz sim ticks; no future interpolation',
        'reward_target': manifest['reward_target'], 'actor_input': manifest['actor_input'],
        'runtime_match': {'camera_fusion': True, 'catch_up': 'as recorded', 'startup_blend': False},
        'limitations': ['Single operator and a few sessions.',
                        'Cleaned targets are not validated against markers yet (AprilTag session pending).',
                        'Mirrored clips are augmentation, not independent data.' if a.mirror else 'No mirrored augmentation.']}, indent=2) + '\n')
    print(json.dumps(manifest['duration_seconds']), 'camera', json.dumps(manifest['camera_seconds']), 'bank clips', len(files), flush=True)


if __name__ == '__main__':
    main()
