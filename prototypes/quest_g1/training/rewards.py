"""Reward rates in physical time; Env integrates them once per physics step."""
from dataclasses import dataclass, asdict
import numpy as np

@dataclass(frozen=True)
class RewardConfig:
    slide_weight: float = 0.5
    slide_force_scale: float = 5.0  # N; includes lightly loaded scuffs
    slide_epsilon: float = 0.01    # m/s; suppress numerical jitter
    global_scale: float = 0.5      # m, half-reward error for the Cauchy kernel
    legacy: bool = False
    legacy_slip: bool = False

    def __post_init__(self):
        if not np.isfinite(list(asdict(self).values())).all():
            raise ValueError('Non-finite reward configuration')
        if self.slide_weight < 0 or min(self.slide_force_scale, self.slide_epsilon, self.global_scale) <= 0:
            raise ValueError('Invalid reward configuration')


def sliding_rate(normal_force, tangent_rms, config):
    """Contact-weighted path rate, summed across feet; never divide by contact time."""
    f = np.maximum(np.asarray(normal_force), 0.)
    v = np.asarray(tangent_rms)
    # Rational form avoids cancellation at zero speed.
    distance_rate = v * v / (np.sqrt(v * v + config.slide_epsilon**2) + config.slide_epsilon)
    return float(np.sum(f / (f + config.slide_force_scale) * distance_rate))


def global_tracking_reward(errors, config):
    if config.legacy:
        return float(2 * np.exp(-10 * np.sum(np.asarray(errors)**2)))
    mse = np.mean(np.sum(np.asarray(errors)**2, axis=-1))
    return float(2 / (1 + mse / config.global_scale**2))


def discrete_rate(value, dt, config):
    """Preserve upstream 50 Hz difference/event penalties after outer DT scaling.

    Action differences scale with dt; touchdown events occur once per contact.
    Unlike continuous state costs, multiplying either by a new dt alone changes
    its physical-time weight. The legacy experiment is preserved for audits.
    """
    if dt <= 0:
        raise ValueError('dt must be positive')
    return float(value) * (1. if config.legacy else .02 / dt)
