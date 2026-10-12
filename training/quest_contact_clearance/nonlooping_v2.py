"""Nonlooping RSI with explicit terminal causes; original PPO stays intact."""
from types import MethodType
import inspect
import torch
from source_patch import edited


def completion(age, lengths, offsets, natural, failure, dt):
    remaining = (lengths - offsets).clamp(min=dt)
    fraction = (age * dt / remaining).clamp(0., 1.)
    # Only natural completion earns source's success bands (>= .95).
    fraction = torch.minimum(fraction, torch.full_like(fraction, .949))
    fraction = torch.where(natural & ~failure, torch.ones_like(fraction), fraction)
    # A safety failure at the last frame must not make the clip easier.
    return torch.where(failure, fraction.clamp(max=.5), fraction)


def install(env):
    if getattr(env, '_nonlooping_v2_installed', False):
        raise ValueError('Nonlooping already installed')
    env._nonlooping_v2_installed = True
    lib = env._motion_lib
    original_frame = lib.calc_motion_frame
    counters = {k: torch.zeros((), dtype=torch.long, device=env.device) for k in
                ('end_resets', 'natural_successes', 'safety_failures', 'time_limits', 'padded_samples')}
    env.terminal_reasons = {k: torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
                            for k in ('natural_end', 'failure', 'time_limit')}
    # Keep every source safety predicate, including relaxed pose tracking.
    # Remove only the merges that destroy information about simultaneous causes.
    safety = edited(env.check_termination.__func__, [
        ('self.reset_buf |= motion_end', 'pass  # natural end handled independently'),
        ('self.reset_buf |= self.time_out_buf', 'pass  # preserve safety mask'),
    ], 'safety')
    difficulty_fn = env._update_motion_difficulty.__func__
    if difficulty_fn.__qualname__ == 'G1MimicDistill._update_motion_difficulty':
        # Pinned distillation subclass delegates identically in all three branches.
        body = inspect.getsource(difficulty_fn)
        if body.count('super()._update_motion_difficulty(env_ids)') != 3:
            raise ValueError('Distillation curriculum delegate changed')
        difficulty_fn = next(c.__dict__['_update_motion_difficulty'] for c in type(env).__mro__
                             if c.__name__ == 'HumanoidMimic')
    source_difficulty = edited(difficulty_fn, [
        ('self.episode_length_buf[env_ids] * self.dt / self._motion_lib.get_motion_length(reset_motion_ids)',
         'self._nonlooping_completion(env_ids)'),
    ], 'difficulty')
    original_reset = env.reset_idx
    def reset(ids, *args, **kwargs):
        # Preserve subclass force-curriculum hooks and all reset side effects.
        rates = {}
        if len(ids):
            duration = (env.episode_length_buf[ids]*env.dt).clamp(min=env.dt)
            for key, values in env.episode_sums.items():
                rates['metric_'+key] = (values[ids]/duration).mean()
                rates['rew_'+key] = (values[ids]*env.reward_scales[key]/duration).mean()
        result = original_reset(ids, *args, **kwargs)
        if len(ids): env.extras['episode'].update(rates)
        return result

    def get_completion(ids):
        reasons = env.terminal_reasons
        return completion(env.episode_length_buf[ids], lib.get_motion_length(env._motion_ids[ids]),
                          env._motion_time_offsets[ids], reasons['natural_end'][ids],
                          reasons['failure'][ids], env.dt)

    def frame(ids, times):
        lengths = lib.get_motion_length(ids)
        past = times >= lengths
        bounded = torch.minimum(times, (lengths - 1e-5).clamp(min=0)).clamp(min=0)
        result = list(original_frame(ids, bounded.clone()))
        for i in (2, 3, 5, 7, 8):
            result[i] = torch.where(past.reshape((-1,) + (1,) * (result[i].ndim - 1)),
                                    torch.zeros_like(result[i]), result[i])
        counters['padded_samples'] += past.sum()
        return tuple(result)

    def termination():
        safety(env)
        failure = env.reset_buf.bool().clone()
        end = env._get_motion_times() >= lib.get_motion_length(env._motion_ids) - 1e-5
        limit = env.episode_length_buf > env.max_episode_length
        env.terminal_reasons.update(natural_end=end.clone(), failure=failure, time_limit=limit.clone())
        env.time_out_buf = (end | limit) & ~failure
        env.reset_buf = failure | end | limit
        counters['end_resets'] += end.sum()
        counters['natural_successes'] += (end & ~failure).sum()
        counters['safety_failures'] += failure.sum()
        counters['time_limits'] += (limit & ~failure & ~end).sum()

    env._nonlooping_completion = get_completion
    env._update_motion_difficulty = MethodType(source_difficulty, env)
    env.reset_idx = reset
    env.check_termination = termination
    lib.calc_motion_frame = frame
    return counters
