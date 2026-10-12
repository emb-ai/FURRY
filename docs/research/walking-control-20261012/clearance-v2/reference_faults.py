"""Faults in actor leg commands only; clean rewards/critic references survive.

The first twelve command joints are the two legs in the pinned G1 contract.
Independent RNG prevents the fault experiment from consuming PPO's RNG stream.
This is an explicit robustness experiment, not unmodified author training.
"""
import torch


class LegFaults:
    def __init__(self, count, device, seed=701, fraction=.5, max_delay=6):
        self.count, self.device = count, device
        self.fraction, self.max_delay = fraction, max_delay
        self.generator = torch.Generator(device=device).manual_seed(seed)
        self.buffer = torch.zeros(max_delay+1, count, 2, 12, device=device)
        self.age = torch.zeros(count, dtype=torch.long, device=device)
        self.delay = torch.zeros_like(self.age)
        self.enabled = torch.zeros(count, dtype=torch.bool, device=device)
        self.next_freeze = torch.zeros_like(self.age)
        self.freeze_left = torch.zeros_like(self.age)
        self.frozen = torch.zeros(count, 2, 12, device=device)
        self.pointer = 0
        self.total = torch.zeros(4, dtype=torch.long, device=device)
        self.reset(torch.arange(count, device=device))

    def randint(self, low, high, count):
        return torch.randint(low, high, (count,), device=self.device, generator=self.generator)

    def reset(self, ids):
        n = len(ids)
        self.age[ids] = 0
        self.enabled[ids] = torch.rand(n, device=self.device, generator=self.generator) < self.fraction
        self.delay[ids] = self.randint(1, self.max_delay+1, n)
        self.next_freeze[ids] = self.randint(75, 225, n)
        self.freeze_left[ids] = 0

    def apply(self, command, future):
        assert command.shape == (self.count, 35) and future.shape == (self.count, 1, 35)
        clean = torch.stack((command[:, 6:18], future[:, 0, 6:18]), dim=1)
        new = self.age == 0
        self.buffer[:, new] = clean[new]
        self.buffer[self.pointer] = clean
        delayed = self.buffer[(self.pointer-self.delay) % len(self.buffer), torch.arange(self.count, device=self.device)]
        moving = (torch.linalg.vector_norm(command[:, :2], dim=-1) > .05) | (command[:, 5].abs() > .15)
        active = self.enabled & moving
        start = active & (self.age >= self.next_freeze) & (self.freeze_left == 0)
        ids = start.nonzero(as_tuple=False).flatten()
        self.frozen[ids] = delayed[ids]
        self.freeze_left[ids] = self.randint(8, 17, len(ids))  # 160--320 ms at 50 Hz
        self.next_freeze[ids] = self.age[ids]+self.randint(200, 351, len(ids))
        freeze = active & (self.freeze_left > 0)
        selected = torch.where(freeze[:, None, None], self.frozen, delayed)
        selected = torch.where(active[:, None, None], selected, clean)
        out, next_out = command.clone(), future.clone()
        out[:, 6:18], next_out[:, 0, 6:18] = selected[:, 0], selected[:, 1]
        self.total += torch.stack((moving.sum(), active.sum(), freeze.sum(), (selected != clean).any(-1).any(-1).sum()))
        self.age += 1
        self.freeze_left = (self.freeze_left-1).clamp(min=0)
        self.pointer = (self.pointer+1) % len(self.buffer)
        return out, next_out


def install_faults(env, seed=701, fraction=.5):
    assert abs(env.dt-.02) < 1e-8 and env.obs_type == 'student_future'
    state = LegFaults(env.num_envs, env.device, seed, fraction)
    original_mimic, original_reset = env._get_mimic_obs, env.reset_idx
    def mimic():
        privileged, command, future = original_mimic()
        command, future = state.apply(command, future)
        return privileged, command, future
    def reset(ids, *args, **kwargs):
        original_reset(ids, *args, **kwargs)
        state.reset(ids)
    env._get_mimic_obs, env.reset_idx = mimic, reset
    return state


def install_nonlooping(env):
    """RSI terminates at remaining duration; future observations pad final pose.

Applied equally to control and treatment. PPO retains upstream timeout
bootstrapping. This changes upstream clip handling explicitly.
"""
    library = env._motion_lib
    original_frame, original_termination = library.calc_motion_frame, env.check_termination
    counters = {'end_resets': torch.zeros((), device=env.device, dtype=torch.long),
                'padded_samples': torch.zeros((), device=env.device, dtype=torch.long)}
    def frame(ids, times):
        # Upstream mutates times in place; protect the caller as well.
        lengths = library.get_motion_length(ids)
        past = times >= lengths
        bounded = torch.minimum(times, lengths-1e-5).clamp(min=0)
        result = list(original_frame(ids, bounded.clone()))
        for index in (2, 3, 5, 7, 8):  # velocities and per-frame root deltas
            result[index] = torch.where(past.reshape((-1,)+(1,)*(result[index].ndim-1)), torch.zeros_like(result[index]), result[index])
        counters['padded_samples'] += past.sum()
        return tuple(result)
    def termination():
        original_termination()
        end = env._get_motion_times() >= library.get_motion_length(env._motion_ids)-1e-5
        failure = env.reset_buf & ~env.time_out_buf
        env.time_out_buf |= end & ~failure
        env.reset_buf |= end
        counters['end_resets'] += end.sum()
    library.calc_motion_frame, env.check_termination = frame, termination
    return counters
