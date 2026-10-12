"""Sphere-sole kinematics and load-normalized sliding; all frames are world.

Isaac net body forces cannot resolve individual contact loads: near-plane
points share the foot load. The approximation is explicit and audited in smoke.
"""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import torch


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rotate(q, v):
    q = q / torch.linalg.vector_norm(q, dim=-1, keepdim=True).clamp(min=1e-8)
    xyz, w = q[..., :3], q[..., 3:]
    t = 2 * torch.cross(xyz.expand_as(v), v, dim=-1)
    return v + w * t + torch.cross(xyz.expand_as(v), t, dim=-1)


def read_soles(path):
    tree = ET.parse(path)
    centers, radii, com = [], [], []
    for side in ('left', 'right'):
        link = tree.find(".//link[@name='%s_ankle_roll_link']" % side)
        cs, rs = [], []
        for c in link.findall('collision'):
            sphere = c.find('geometry/sphere')
            if sphere is None:
                raise ValueError('Only audited sphere soles are supported')
            cs.append([float(x) for x in c.find('origin').get('xyz', '0 0 0').split()])
            rs.append(float(sphere.get('radius')))
        if len(cs) != 4:
            raise ValueError('Expected four collision spheres per sole')
        centers.append(cs); radii.append(rs)
        com.append([float(x) for x in link.find('inertial/origin').get('xyz').split()])
    return np.asarray(centers), np.asarray(radii), np.asarray(com)


def kinematics(states, centers, radii, origin_local, floor_height=0.):
    """states [...,2,13], pos at link origin; origin_local from validated receipt."""
    q = states[..., 3:7].unsqueeze(-2)
    offset = rotate(q, centers.expand(*states.shape[:-1], 4, 3))
    points = states[..., :3].unsqueeze(-2) + offset
    normal = torch.zeros_like(points); normal[..., 2] = 1.
    points = points - normal * radii[..., None]
    origin = states[..., :3] + rotate(states[..., 3:7], origin_local.expand_as(states[..., :3]))
    omega = states[..., 10:13].unsqueeze(-2).expand_as(points)
    velocity = states[..., 7:10].unsqueeze(-2) + torch.cross(omega, points-origin.unsqueeze(-2), dim=-1)
    gap = points[..., 2] - floor_height
    return gap.min(dim=-1).values, gap, velocity, points


def contact_penalty(gap, point_velocity, force_z, expected_support, weight_newtons,
                    contact_band=.008, speed_scale=.2):
    near = (gap <= contact_band).to(gap.dtype)
    shares = near / near.sum(-1, keepdim=True).clamp(min=1)
    # Each foot load occurs ONCE, then is split across geometric candidates.
    load = torch.minimum(force_z.clamp(min=0.), weight_newtons[:, None]) * (near.sum(-1) > 0)
    point_load = shares * load.unsqueeze(-1)
    v2 = point_velocity[..., :2].square().sum(-1) / speed_scale**2
    sliding = (point_load * v2.clamp(max=25.)).sum((-1, -2)) / load.sum(-1).clamp(min=1.)
    # Teacher-supported intervals penalize unloading/flight instead of rewarding it.
    missing_support = (1. - load.sum(-1)/(weight_newtons*.5)).clamp(0., 1.)
    missing_support = missing_support * expected_support.to(gap.dtype)
    return sliding + missing_support, sliding, missing_support, point_load


def clearance_penalty(actual, target, valid, scale=.025):
    # Two-sided bounded error: extra height and never landing cannot earn reward.
    loss = ((actual-target)/scale).square().clamp(max=16.) * valid
    return loss.sum(-1) / valid.sum(-1).clamp(min=1)


class PhaseTable:
    """Offline TRAIN-only kinematic teacher, indexed by immutable motion filename."""
    def __init__(self, path, files, dataset, device):
        path = Path(path)
        self.metadata = json.loads(path.with_suffix('.json').read_text())
        if self.metadata['dataset_manifest_sha256'] != sha(Path(dataset)/'manifest.json'):
            raise ValueError('Phase dataset differs')
        if self.metadata['npz_sha256'] != sha(path):
            raise ValueError('Phase cache changed')
        index = self.metadata['files']
        names = [str(Path(f).resolve().relative_to(Path(dataset).resolve())) for f in files]
        if set(names) != set(index):
            raise ValueError('Phase motion set differs')
        with np.load(path, allow_pickle=False) as z:
            arrays = [z[index[name]['key']].copy() for name in names]
        max_n = max(len(x) for x in arrays)
        padded = np.zeros((len(arrays), max_n, 2, 3), np.float32)
        for i, arr in enumerate(arrays):
            padded[i, :len(arr)] = arr
        self.table = torch.as_tensor(padded, device=device)
        self.fps = torch.tensor([index[n]['fps'] for n in names], device=device)
        self.lengths = torch.tensor([len(x) for x in arrays], device=device)

    def sample(self, ids, times):
        idx = torch.floor(times*self.fps[ids]).long().clamp(min=0)
        inside = idx < self.lengths[ids]-1
        idx = torch.minimum(idx, self.lengths[ids]-1)
        result = self.table[ids, idx].clone()
        result[..., 1] *= inside[:, None]  # validity; no targets beyond natural end
        return result


class ContactClearance:
    feature_dim = 6  # target/valid/stance per foot, same in A/B/C

    def __init__(self, env, config, sole_urdf, phase_path, dataset):
        self.env, self.config = env, config
        centers, radii, com = read_soles(sole_urdf)
        self.centers = torch.tensor(centers, dtype=torch.float32, device=env.device)
        self.radii = torch.tensor(radii, dtype=torch.float32, device=env.device)
        origin = config['velocity_origin']
        if origin not in ('link', 'com'):
            raise ValueError('Velocity origin must come from the simulator audit')
        self.origin_local = torch.tensor(com if origin == 'com' else np.zeros_like(com), dtype=torch.float32, device=env.device)
        # Read loaded rigid body properties: body-mass randomization is per env.
        properties = [env.gym.get_actor_rigid_body_properties(e, h) for e,h in zip(env.envs,env.actor_handles)]
        masses = [sum(b.mass for b in ps) for ps in properties]
        foot_ids = env.feet_indices.cpu().tolist()
        names = env.gym.get_actor_rigid_body_names(env.envs[0], env.actor_handles[0])
        if [names[i] for i in foot_ids] != ['left_ankle_roll_link','right_ankle_roll_link']:
            raise ValueError('Unexpected feet order')
        if origin == 'com':
            self.origin_local = torch.tensor([[[ps[i].com.x,ps[i].com.y,ps[i].com.z] for i in foot_ids]
                                             for ps in properties],device=env.device)
        self.weight = torch.tensor(masses, device=env.device) * 9.81
        self.phase = PhaseTable(phase_path, env._motion_lib._motion_files, dataset, env.device) if phase_path else None
        if config['condition'] == 'C' and self.phase is None:
            raise ValueError('C requires a validated teacher')
        self.features = torch.zeros(env.num_envs, self.feature_dim, device=env.device)
        self.cooldown_until = -1
        self.raw = {}
        self.samples = []
        self.delay_counts = torch.zeros(2, dtype=torch.long, device=env.device)
        self.source_counts = {}
        self._install()

    def advance(self):
        e = self.env
        phase = (self.phase.sample(e._motion_ids, e._get_motion_times()) if self.phase else
                 torch.zeros(e.num_envs, 2, 3, device=e.device))
        phase[..., 1] *= (e.episode_length_buf > 2)[:, None]
        if e.common_step_counter <= self.cooldown_until:
            phase[..., 1] = 0  # corrective touchdown after source push is allowed
        self.features = phase.flatten(1)
        actual, gaps, velocity, points = kinematics(e.rigid_body_states[:, e.feet_indices],
            self.centers, self.radii, self.origin_local)
        expected_support = ((phase[..., 1] > .5) & (phase[..., 2] > .5)).any(-1)
        total, sliding, missing, loads = contact_penalty(gaps, velocity,
            e.contact_forces[:, e.feet_indices, 2], expected_support, self.weight)
        self.raw = {'contact_points': total, 'sole_clearance': clearance_penalty(actual, phase[..., 0], phase[..., 1])}
        self.diagnostics = dict(clearance=actual, gap=gaps, velocity=velocity, points=points,
                                phase=phase, loads=loads, sliding=sliding, missing_support=missing)

    def _install(self):
        e = self.env
        original_reward, original_reset, original_push = e.compute_reward, e.reset_idx, e._push_robots
        weights = self.config['weights']
        for name in ('contact_points', 'sole_clearance'):
            e.reward_names.append(name)
            e.reward_functions.append(lambda name=name: -self.raw[name])
            e.reward_scales[name] = float(weights[name])*e.dt
            e.episode_sums[name] = torch.zeros(e.num_envs, device=e.device)
        self.term_names = list(e.reward_names)
        self.current_terms = {}
        for i, (name, fn) in enumerate(zip(e.reward_names, e.reward_functions)):
            def measured(fn=fn, name=name):
                value = fn()
                # Record the actual result, never call stateful rewards twice.
                self.current_terms[name] = torch.as_tensor(value, device=e.device).detach().expand(e.num_envs).clone()
                return value
            e.reward_functions[i] = measured
        def reward():
            self.advance()
            original_reward()
            terms = torch.stack([self.current_terms[n] for n in self.term_names], -1)
            weights = torch.tensor([e.reward_scales[n] / e.dt *
                (e.cfg.rewards.regularization_scale if n in e.cfg.rewards.regularization_names else 1.)
                for n in self.term_names], device=e.device)
            # Bounded deterministic sample for p95; all environments for means.
            self.samples.append((terms.mean(0), terms[::max(1, e.num_envs//128)].clone(), weights,
                                 self.features.reshape(-1, 2, 3)[..., 1].mean(),
                                 ((self.features.reshape(-1, 2, 3)[..., 0] > .005) &
                                  (self.features.reshape(-1, 2, 3)[..., 1] > .5)).float().mean()))
        def reset(ids, *args, **kwargs):
            original_reset(ids, *args, **kwargs)
            self.features[ids] = 0
        def push():
            original_push()
            self.cooldown_until = e.common_step_counter + round(1./e.dt)
        e.compute_reward, e.reset_idx, e._push_robots = reward, reset, push
        self.advance()

    def report(self):
        if not self.samples:
            return {}
        means = torch.stack([x[0] for x in self.samples]).mean(0)
        sample = torch.cat([x[1] for x in self.samples])
        weights = self.samples[-1][2]
        weighted = sample*weights
        denominator = weighted.abs().sum(-1).mean().clamp(min=1e-9)
        result = {'terms': {name: {'raw_mean': float(means[i]),
                    'raw_abs_p95_sample': float(torch.quantile(sample[:, i].abs(), .95)),
                    'weighted_mean_rate': float(means[i]*weights[i]),
                    'weighted_abs_p95_sample': float(torch.quantile(weighted[:, i].abs(), .95)),
                    'absolute_share': float(weighted[:, i].abs().mean()/denominator)}
                  for i, name in enumerate(self.term_names)},
                  'phase_valid_fraction': float(torch.stack([x[3] for x in self.samples]).mean()),
                  'phase_swing_fraction': float(torch.stack([x[4] for x in self.samples]).mean())}
        self.samples.clear()
        return result
