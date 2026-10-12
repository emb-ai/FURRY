"""Explicit reward scaling and optimizer continuation; simulator state resets."""
import math
from dataclasses import replace


def set_clearance_multiplier(env, state, multiplier):
    if not math.isfinite(multiplier) or multiplier <= 0:
        raise ValueError('Clearance multiplier must be finite and positive')
    # Called after the frozen source's unit-weight geometry/scaling checks.
    if abs(env.reward_scales['low_moving_foot'] + env.dt) > 1e-8:
        raise ValueError('Expected original unit penalty before scaling')
    state.config = replace(state.config, reward_coefficient=-multiplier)
    env.reward_scales['low_moving_foot'] = -multiplier * env.dt
    return dict(clearance_multiplier=multiplier,
                clearance_reward_scale=env.reward_scales['low_moving_foot'],
                clearance_dt_applied_exactly_once=True)


def restore_optimizer(algorithm, checkpoint):
    algorithm.optimizer.load_state_dict(checkpoint['optimizer'])
    rates = {group['lr'] for group in algorithm.optimizer.param_groups}
    if len(rates) != 1:
        raise ValueError('Expected a single adaptive learning rate')
    algorithm.learning_rate = rates.pop()
    if 'adaptive_learning_rate' in checkpoint:
        if algorithm.learning_rate != checkpoint['adaptive_learning_rate']:
            raise ValueError('Optimizer/adaptive LR mismatch')
    return int(checkpoint['iteration'])
