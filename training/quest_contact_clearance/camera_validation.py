"""Camera-leg validation in MuJoCo with the live actor commands.

commands: read the raw validation episodes (local only, never committed) and
  write each validation clip's live mimic.csv commands as a causal timeline
  [t, 35 commands], cut exactly as camera_dataset.py cuts train clips.
evaluate: roll a 50 Hz ONNX policy over the validation clips from frame 0.
  --mode live feeds the live commands to the actor (zero-order hold, no future
  interpolation); --mode target feeds commands derived from the cleaned target
  (the usual evaluate.py input). Reward target and metrics are the same in both.
  The lift metric compares sole clearance (floor-colliding spheres) of the robot
  with the reference while the reference foot is up: contact, shortfall, peak
  ratio and retention 1 s into held lifts, split by training's moving mask.
  --catch-up adds the headset catch-up loop (catch_up.servo: reference root as
  goal, robot root as actual) to the actor command, as on the headset.
Usage:
  camera_validation.py commands --dataset DATASET --episodes EPISODE_DIR ... --output CMD.npz
  camera_validation.py evaluate --assets ASSETS --dataset DATASET --policy P.onnx
      [--commands CMD.npz --mode live] [--catch-up] --output REPORT.json
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

LEGS = slice(0, 12)
# Lifted-foot metric: reference sole clearance (floor-colliding spheres) above
# LIFT_ON counts as lifted; an event is a lifted run whose peak reaches
# LIFT_PEAK. Held events last at least HELD_S; retention is read 1 s after
# the lift starts. Contact uses the reward's 5 N threshold.
LIFT_ON, LIFT_PEAK, HELD_S, RETAIN_S = .03, .05, .5, 1.


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def live_commands(episode):
    """camera_dataset.ticks(): drop the pre-reference first tick; [sim_time, 35 commands]."""
    a = np.loadtxt(Path(episode)/'mimic.csv', delimiter=',', skiprows=1, ndmin=2)[1:]
    t, cmd = a[:, 2], a[:, 3:]
    if cmd.shape[1] != 35 or not (np.diff(t) > 0).all():
        raise ValueError(f'Unexpected mimic.csv in {episode}')
    return t, cmd


def build_commands(dataset, episodes, output):
    manifest = json.loads((Path(dataset)/'manifest.json').read_text())
    clips = [c for c in manifest['clips'] if c['split'] == 'validation']
    by_id = {Path(e).name.replace('episode-', ''): Path(e) for e in episodes}
    if {c['episode'] for c in clips} - set(by_id):
        raise ValueError('Missing raw validation episode')
    arrays, files, cache = {}, {}, {}
    for c in clips:
        eid = c['episode']
        episode = by_id[eid]
        if sha(episode/'mimic.csv') != manifest['episodes'][eid]['source_sha256']['mimic.csv']:
            raise ValueError(f'mimic.csv differs from the dataset source: {eid}')
        t, cmd = cache.setdefault(eid, live_commands(episode))
        t0, t1 = c['source_start_sim_s'], c['source_start_sim_s'] + c['seconds']
        first, last = np.searchsorted(t, t0, side='right') - 1, np.searchsorted(t, t1, side='right')
        rel = t[first:last] - t0; rel[0] = 0.
        key = 'clip_%04d' % len(arrays); arrays[key] = np.column_stack((rel, cmd[first:last]))
        files[c['file']] = {'key': key, 'id': c['id'], 'stage': c['stage'], 'episode': eid, 'target_sha256': c['sha256']}
    np.savez_compressed(output, **arrays)
    Path(output).with_suffix('.json').write_text(json.dumps({
        'schema': 'quest-camera-validation-commands-v1', 'npz_sha256': sha(output),
        'dataset_manifest_sha256': sha(Path(dataset)/'manifest.json'), 'files': files,
        'command_sampling': 'causal zero-order hold at original command times'}, indent=2) + '\n')
    print(json.dumps({'clips': len(files), 'stages': sorted({f['stage'] for f in files.values()})}))


def lift_events(ref, robot, contact, dt):
    """ref/robot sole clearance [T, 2] (m), contact [T, 2] bool -> per-event rows."""
    rows = []
    for side in range(ref.shape[1]):
        up = ref[:, side] > LIFT_ON
        edges = np.diff(np.r_[0, up.astype(int), 0])
        for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            r, h = ref[a:b, side], robot[a:b, side]
            if r.max() < LIFT_PEAK:
                continue
            row = {'side': side, 'seconds': (b - a) * dt, 'ref_peak_m': float(r.max()),
                   'peak_ratio': float(h.max() / r.max()), 'contact_fraction': float(contact[a:b, side].mean())}
            k = a + round(RETAIN_S / dt)
            if k < b:
                w = slice(max(a, k - round(.1 / dt)), min(b, k + round(.1 / dt) + 1))
                row['retention_1s'] = float(robot[w, side].mean() / ref[w, side].mean())
            rows.append(row)
    return rows


def lift_summary(events, ref, robot, contact, dt, moving=None):
    """Pooled over trials: events from lift_events, frame arrays concatenated [T, 2];
    moving [T] is training's relaxed-leg mask, used only to split the contact fraction."""
    lifted = ref > LIFT_ON
    held = [e for e in events if e['seconds'] >= HELD_S]
    retained = [e['retention_1s'] for e in held if 'retention_1s' in e]
    median = lambda xs: float(np.median(xs)) if len(xs) else None
    return {'lifted_foot_seconds': float(lifted.sum() * dt),
            'lifted_contact_fraction': float(contact[lifted].mean()) if lifted.any() else None,
            'lifted_shortfall_cm_mean': float(100 * np.maximum(ref - robot, 0)[lifted].mean()) if lifted.any() else None,
            'events': len(events), 'held_events': len(held),
            'peak_ratio_median': median([e['peak_ratio'] for e in events]),
            'held_peak_ratio_median': median([e['peak_ratio'] for e in held]),
            'held_contact_fraction_median': median([e['contact_fraction'] for e in held]),
            'retention_1s_events': len(retained), 'retention_1s_median': median(retained),
            'retention_1s_p25': float(np.percentile(retained, 25)) if retained else None,
            **({} if moving is None else {
                'lifted_seconds_' + name: float((lifted & mask[:, None]).sum() * dt) for name, mask in
                (('moving', moving), ('still', ~moving))}),
            **({} if moving is None else {
                'lifted_contact_fraction_' + name: float(contact[lifted & mask[:, None]].mean())
                if (lifted & mask[:, None]).any() else None for name, mask in (('moving', moving), ('still', ~moving))})}


def evaluate(assets, dataset, policy, output, commands=None, mode='target', seeds=(0,), catch_up=False):
    import onnxruntime as ort
    import env as mj
    from gait_evaluation import runtime_env_class
    from sole_geometry import sole_clearance
    if catch_up:
        import torch
        from catch_up import servo
    motions = mj.Motions(dataset, 'validation')
    live = None
    if mode == 'live':
        meta = json.loads(Path(commands).with_suffix('.json').read_text())
        if meta['npz_sha256'] != sha(commands) or meta['dataset_manifest_sha256'] != sha(Path(dataset)/'manifest.json'):
            raise ValueError('Validation commands do not match the dataset')
        z = np.load(commands)
        live = {}
        for item in motions.items:
            row = meta['files'][item['meta']['file']]
            if row['target_sha256'] != item['meta']['sha256']:
                raise ValueError('Validation commands paired with another target')
            live[item['meta']['id']] = z[row['key']]
    elif mode != 'target':
        raise ValueError('mode must be live or target')

    class Env(runtime_env_class(mj.Env, 50)):
        def observation(self):
            if (live is None and not catch_up) or self.steps % self.policy_stride:
                return super().observation()
            # Same assembly as env.Env.observation with only the actor command replaced.
            if live is None:
                cmd = self.motion['cmd'][self.k].astype(np.float32)
            else:
                table = live[self.motion['meta']['id']]
                t = (self.k - self.first) * mj.DT
                cmd = table[max(0, np.searchsorted(table[:, 0], t + 1e-9, side='right') - 1), 1:].astype(np.float32)
                if t > self.motion['meta']['seconds']:
                    cmd[[0, 1, 5]] = 0
            if catch_up:
                pose = lambda q: (torch.from_numpy(q[None, :3]), torch.from_numpy(q[None, [4, 5, 6, 3]]))  # wxyz -> xyzw
                cmd = servo(torch.from_numpy(cmd[None]), *pose(self.ref.qpos[:7]), *pose(self.data.qpos[:7]))[0].numpy()
            saved = self.motion['cmd'][self.k].copy()
            self.motion['cmd'][self.k] = cmd
            try:
                return super().observation()
            finally:
                self.motion['cmd'][self.k] = saved

    opts = ort.SessionOptions(); opts.intra_op_num_threads = 1; opts.inter_op_num_threads = 1
    session = ort.InferenceSession(str(policy), sess_options=opts, providers=['CPUExecutionProvider'])
    name = session.get_inputs()[0].name
    env = Env(assets, motions); env.max_steps = 10**7
    trials, frames = [], []
    for clip, item in enumerate(motions.items):
        for seed in seeds:
            env.rng = np.random.default_rng(seed)
            obs = env.reset(clip=clip, start=0, randomize=seed != 0)
            leg, slip, root, moving = [], [], [], []
            ref_h, robot_h, loaded, pelvis = [], [], [], []
            while True:
                if env.steps % env.policy_stride == 0:
                    action = session.run(None, {name: obs[None]})[0][0]
                obs, _, done, info = env.step(action)
                q = env.motion['q'][env.k]
                leg.append(np.degrees(np.abs(env.data.qpos[env.qa[LEGS]] - q[7:19])))
                # Training's relaxed-leg mask (locomotion_ablation): reference root moving or turning.
                v = env.motion['v'][env.k]
                moving.append(np.hypot(*env.motion['cmd'][env.k, :2]) > .05 or abs(v[5]) > .15)
                contact = env.contact_force > 5  # loaded feet only, as the sliding reward
                slip.extend(env.contact_speed[contact])
                root.append(info['root_xy_error_m'])
                ref_h.append([sole_clearance(env.model, env.ref, f, env.floor) for f in env.feet])
                robot_h.append([sole_clearance(env.model, env.data, f, env.floor) for f in env.feet])
                loaded.append(contact)
                pelvis.append((env.data.qpos[2], env.data.qpos[2] - env.ref.qpos[2]))
                if done:
                    break
            leg, moving = np.array(leg), np.array(moving)
            ref_h, robot_h, loaded, pelvis = np.array(ref_h), np.array(robot_h), np.array(loaded), np.array(pelvis)
            events = lift_events(ref_h, robot_h, loaded, mj.DT)
            frames.append((events, ref_h, robot_h, loaded, moving))
            trials.append({'clip': item['meta']['id'], 'stage': item['meta']['stage'], 'seed': seed,
                           'camera_share': item['meta'].get('camera_share'),
                           'reference_seconds': (len(item['q']) - 1) * mj.DT, 'survived_seconds': info['seconds'],
                           'fell': info['fall'], 'leg_joint_error_deg_mean': float(leg.mean()),
                           'leg_joint_error_deg_p95': float(np.percentile(leg, 95)),
                           'moving_fraction': float(moving.mean()),
                           'leg_joint_error_deg_still': float(leg[~moving].mean()) if (~moving).any() else None,
                           'leg_joint_error_deg_moving': float(leg[moving].mean()) if moving.any() else None,
                           'support_slip_m_s_mean': float(np.mean(slip)) if slip else 0.,
                           'support_slip_m_s_p90': float(np.percentile(slip, 90)) if slip else 0.,
                           'slide_path_m': info['slide_path_m'],
                           'root_xy_error_m_final': root[-1], 'root_xy_error_m_max': float(np.max(root)),
                           'pelvis_z_min_m': float(pelvis[:, 0].min()), 'pelvis_below_reference_max_m': float(-pelvis[:, 1].min()),
                           'reference_stance_clearance_p10_m': float(np.percentile(ref_h, 10)),
                           'lift': lift_summary(events, ref_h, robot_h, loaded, mj.DT, moving)})
            print(json.dumps(trials[-1]), flush=True)
    summary = {}
    for stage in sorted({t['stage'] for t in trials}):
        rows = [t for t in trials if t['stage'] == stage]
        w = np.array([t['survived_seconds'] for t in rows])
        summary[str(stage)] = {
            'clips': len(rows), 'falls': sum(t['fell'] for t in rows),
            'survived_s': float(w.sum()), 'reference_s': sum(t['reference_seconds'] for t in rows),
            **{k: float(np.average([t[k] for t in rows], weights=w)) for k in
               ('leg_joint_error_deg_mean', 'leg_joint_error_deg_p95', 'support_slip_m_s_mean', 'support_slip_m_s_p90')},
            'slide_path_m_per_s': float(sum(t['slide_path_m'] for t in rows) / w.sum()),
            'root_xy_error_m_final_mean': float(np.mean([t['root_xy_error_m_final'] for t in rows])),
            **{k: float(np.average([t[k] for t in rows], weights=[t['survived_seconds'] * (f if k.endswith('moving') else 1 - f) for t, f in ((t, t['moving_fraction']) for t in rows)]))
               for k in ('leg_joint_error_deg_still', 'leg_joint_error_deg_moving')
               if all(t[k] is not None for t in rows)},
            'moving_fraction': float(np.average([t['moving_fraction'] for t in rows], weights=w)),
            'root_xy_error_m_max': float(np.max([t['root_xy_error_m_max'] for t in rows]))}
    ref_h, robot_h, loaded, moving = [np.concatenate([f[i] for f in frames]) for i in (1, 2, 3, 4)]
    summary['lift'] = lift_summary([e for f in frames for e in f[0]], ref_h, robot_h, loaded, mj.DT, moving)
    summary['falls'] = sum(t['fell'] for t in trials)
    summary['pelvis_z_min_m'] = min(t['pelvis_z_min_m'] for t in trials)
    report = {'policy': str(policy), 'policy_sha256': sha(policy), 'mode': mode, 'catch_up': catch_up,
              'commands_sha256': sha(commands) if live is not None else None,
              'dataset_manifest_sha256': sha(Path(dataset)/'manifest.json'),
              'scene_sha256': sha(Path(assets)/'scene.xml'), 'control_hz': 50, 'seeds': list(seeds),
              'summary': summary, 'trials': trials}
    Path(output).write_text(json.dumps(report, indent=2))
    print(json.dumps(summary, indent=2))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='command', required=True)
    c = sub.add_parser('commands')
    c.add_argument('--dataset', type=Path, required=True); c.add_argument('--episodes', nargs='+', required=True)
    c.add_argument('--output', type=Path, required=True)
    e = sub.add_parser('evaluate')
    for k in ('assets', 'dataset', 'policy', 'output'):
        e.add_argument('--' + k, type=Path, required=True)
    e.add_argument('--commands', type=Path); e.add_argument('--mode', default='target')
    e.add_argument('--seeds', nargs='+', type=int, default=[0])
    e.add_argument('--catch-up', action='store_true', help='headset catch-up loop on the actor command')
    a = p.parse_args()
    if a.command == 'commands':
        build_commands(a.dataset, a.episodes, a.output)
    else:
        evaluate(a.assets, a.dataset, a.policy, a.output, a.commands, a.mode, a.seeds, a.catch_up)


if __name__ == '__main__':
    main()
