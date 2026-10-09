"""Dense displacement tracking on consecutive one-second windows.

Positions are measured in EACH trajectory's heading at window start. This
tracks progress/turning without asking the actor to recover an unobserved world
offset. Nine extra critic features expose the window state; actor is unchanged.
No future simulator state, delayed reward, or reference look-ahead is used.
"""
import torch


def yaw_xyzw(q):
    x, y, z, w = q.unbind(-1)
    # Norm-invariant extraction: author TWIST1 files include non-unit quaternions.
    return torch.atan2(2*(w*z+x*y), w*w+x*x-y*y-z*z)


def wrap(x):
    return torch.atan2(torch.sin(x), torch.cos(x))


def local_delta(p, anchor, yaw):
    d = p-anchor
    c, s = torch.cos(yaw), torch.sin(yaw)
    return torch.stack((c*d[:, 0]+s*d[:, 1], -s*d[:, 0]+c*d[:, 1]), -1)


def huber(x):
    a = x.abs()
    return torch.where(a <= 1, .5*x.square(), a-.5)


class WindowReward:
    feature_dim = 9

    def __init__(self, count, device, dt, seconds=1., position_scale=.25, yaw_scale=.5):
        self.dt, self.seconds = float(dt), float(seconds)
        self.steps = round(seconds/dt)
        if self.steps < 1 or abs(self.steps*dt-seconds) > 1e-6:
            raise ValueError('Window must be a positive integer number of physics-policy steps')
        self.position_scale, self.yaw_scale = position_scale, yaw_scale
        if min(position_scale, yaw_scale) <= 0:
            raise ValueError('Reward scales must be positive')
        self.age = torch.zeros(count, device=device, dtype=torch.long)
        self.anchor = torch.zeros(count, 6, device=device)
        self.features = torch.zeros(count, self.feature_dim, device=device)
        self.features[:, 5] = self.features[:, 7] = 1.
        self.position_penalty = torch.zeros(count, device=device)
        self.yaw_penalty = torch.zeros(count, device=device)
        self.distance_error = torch.zeros(count, device=device)

    def reset(self, ids, pos, yaw, ref_pos, ref_yaw):
        self.anchor[ids] = torch.cat((pos[ids], ref_pos[ids], yaw[ids, None], ref_yaw[ids, None]), -1)
        self.age[ids] = 0
        self.features[ids] = 0
        self.features[ids, 5] = self.features[ids, 7] = 1.

    def advance(self, pos, yaw, ref_pos, ref_yaw):
        self.age += 1
        actual = local_delta(pos, self.anchor[:, :2], self.anchor[:, 4])
        target = local_delta(ref_pos, self.anchor[:, 2:4], self.anchor[:, 5])
        actual_yaw = wrap(yaw-self.anchor[:, 4])
        target_yaw = wrap(ref_yaw-self.anchor[:, 5])
        error = actual-target
        self.position_penalty = huber(error/self.position_scale).sum(-1)
        self.yaw_penalty = huber(wrap(actual_yaw-target_yaw)/self.yaw_scale)
        self.distance_error = torch.linalg.vector_norm(error, dim=-1)
        self.features = torch.cat((actual, target, torch.sin(actual_yaw)[:, None],
                                   torch.cos(actual_yaw)[:, None], torch.sin(target_yaw)[:, None],
                                   torch.cos(target_yaw)[:, None], (self.age*self.dt/self.seconds)[:, None]), -1)
        # The boundary sample is scored BEFORE rebasing. New observations expose
        # the rebased state used for the next action, not a stale terminal window.
        ids = (self.age >= self.steps).nonzero(as_tuple=False).flatten()
        self.reset(ids, pos, yaw, ref_pos, ref_yaw)


def install(env, position_weight=.25, yaw_weight=.25):
    """Install after construction; reset hooks also handle upstream RSI resets."""
    state = WindowReward(env.num_envs, env.device, env.dt)
    def current():
        return env.root_states[:, :2], yaw_xyzw(env.root_states[:, 3:7]), env._ref_root_pos[:, :2], yaw_xyzw(env._ref_root_rot)
    state.reset(torch.arange(env.num_envs, device=env.device), *current())
    original_reward, original_reset = env.compute_reward, env.reset_idx
    for name, weight, field in [('window_displacement', position_weight, 'position_penalty'),
                                ('window_turn', yaw_weight, 'yaw_penalty')]:
        env.reward_names.append(name)
        env.reward_functions.append(lambda field=field: -getattr(state, field))
        env.reward_scales[name] = weight*env.dt  # Exactly once; invariant to policy Hz.
        env.episode_sums[name] = torch.zeros(env.num_envs, device=env.device)
    def compute_reward():
        state.advance(*current())
        original_reward()
    def reset_idx(ids, *args, **kwargs):
        original_reset(ids, *args, **kwargs)
        state.reset(ids, *current())
    env.compute_reward, env.reset_idx = compute_reward, reset_idx
    return state
