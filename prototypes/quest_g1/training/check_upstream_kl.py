"""Integration regression: upstream PPO must continue after high minibatch KL.

Run in the pinned upstream Python/Isaac environment. Uses a tiny CPU policy,
not a simulator rollout; verifies control flow rather than policy quality.
"""
import isaacgym  # noqa: F401 -- required before source package imports torch
import torch
from torch import nn
from torch.distributions import Normal
from rsl_rl.algorithms import DaggerPPO


class TinyPolicy(nn.Module):
    is_recurrent = False

    def __init__(self):
        super().__init__()
        self.actor = nn.Linear(4, 2)
        self.critic = nn.Linear(4, 1)
        self.std = nn.Parameter(torch.ones(2))

    def act(self, x, **kwargs):
        self.distribution = Normal(self.actor(x), self.std)
        return self.distribution.sample()

    def evaluate(self, x, **kwargs): return self.critic(x)
    def if_fix_std(self): return False
    def reset(self, dones): pass
    def get_actions_log_prob(self, actions): return self.distribution.log_prob(actions).sum(-1)
    @property
    def action_mean(self): return self.distribution.mean
    @property
    def action_std(self): return self.distribution.stddev
    @property
    def entropy(self): return self.distribution.entropy().sum(-1)


def main():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    model = TinyPolicy()
    alg = DaggerPPO(None, model, model, teacher_loaded=False, learning_rate=2e-4,
                    schedule='adaptive', desired_kl=.008, num_learning_epochs=5,
                    num_mini_batches=4, device='cpu')
    alg.init_storage(8, 4, [4], [4], [2])
    steps = []
    original_step = alg.optimizer.step
    def record_step(*args, **kwargs):
        steps.append(alg.optimizer.param_groups[0]['lr'])
        return original_step(*args, **kwargs)
    alg.optimizer.step = record_step
    for iteration in range(2):
        with torch.no_grad():
            for _ in range(4):
                obs = torch.randn(8, 4)
                alg.act(obs, obs, {})
                alg.process_env_step(torch.rand(8), torch.zeros(8, dtype=torch.bool), {})
            alg.compute_returns(torch.randn(8, 4))
        if iteration == 0:
            # Deliberately large divergence, far above the former run-stop threshold.
            alg.storage.mu.add_(5.)
        result = alg.update()
        assert all(torch.isfinite(torch.tensor(x)) for x in result)
        assert len(steps) == (iteration+1)*20
    assert steps[0] < 2e-4, steps
    assert alg.counter == 2
    print({'passed': True, 'high_kl_update_steps': 20, 'next_rollout_update_steps': 20,
           'first_adapted_lr': steps[0], 'completed_iterations': alg.counter})


if __name__ == '__main__': main()
