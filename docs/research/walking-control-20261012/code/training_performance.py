"""Optimizations which leave actor PPO, rewards and simulation frequency intact."""
import math
import torch


def critic_only_update(alg):
    """Same clipped value objective/minibatches/Adam as fixed-LR DaggerPPO.

    Actor/entropy/surrogate computations have zero trainable gradient in this
    phase. Omitting them changes the RNG stream (unused dropout/action samples),
    but not the critic update for a fixed rollout and minibatch permutation.
    Actor-phase updates must continue through upstream DaggerPPO.update().
    """
    ac = alg.actor_critic
    if (alg.schedule != 'fixed' or alg.teacher_loaded or alg.fix_std or ac.is_recurrent
            or ac.std.requires_grad or any(p.requires_grad for p in ac.actor.parameters())):
        raise ValueError('Fast critic path requires frozen actor/std, fixed LR and no teacher')
    storage = alg.storage
    batch_size = storage.num_envs*storage.num_transitions_per_env
    size = batch_size//alg.num_mini_batches
    indices = torch.randperm(alg.num_mini_batches*size, device=alg.device)
    observations = (storage.privileged_observations if storage.privileged_observations is not None
                    else storage.observations).flatten(0, 1)
    values, returns = storage.values.flatten(0, 1), storage.returns.flatten(0, 1)
    total = torch.zeros((), device=alg.device, dtype=torch.float64)
    for _ in range(alg.num_learning_epochs):
        for i in range(alg.num_mini_batches):
            ids = indices[i*size:(i+1)*size]
            predicted = ac.evaluate(observations[ids])
            target, old = returns[ids], values[ids]
            if alg.use_clipped_value_loss:
                clipped = old+(predicted-old).clamp(-alg.clip_param, alg.clip_param)
                value_loss = torch.maximum((predicted-target).square(), (clipped-target).square()).mean()
            else:
                value_loss = (predicted-target).square().mean()
            alg.optimizer.zero_grad()
            (alg.value_loss_coef*value_loss).backward()
            torch.nn.utils.clip_grad_norm_(ac.parameters(), alg.max_grad_norm)
            alg.optimizer.step()
            total += value_loss.detach().double()
    alg.counter += 1
    storage.clear()
    if alg.counter < alg.dagger_coef_anneal_steps:
        alg.dagger_coef *= .5*(1+math.cos(math.pi*alg.counter/alg.dagger_coef_anneal_steps))
    else:
        alg.dagger_coef = alg.dagger_coef_min
    # Surrogate was deliberately not evaluated: zero is a placeholder, marked
    # explicitly in trainer metrics; it must not be interpreted as actor loss.
    return float(total)/(alg.num_learning_epochs*alg.num_mini_batches), 0., 0., 0, 0, 0, 0.


class RolloutMetrics:
    """Accumulate float32 reductions in float64, matching Python-float sums.

    One host transfer per rollout instead of seven scalar transfers per step.
    Counts are exact within float64's integer range; no training tensor changes.
    """
    def __init__(self, device, quest_mask):
        self.values = torch.zeros(7, device=device, dtype=torch.float64)
        self.quest_mask = quest_mask

    def record_motion_ids(self, ids):
        self.values[0] += self.quest_mask[ids].sum().double()

    def record_step(self, rewards, dones, time_outs, window_state):
        z = rewards.new_zeros(())
        position, turn, distance = ((window_state.position_penalty.mean(), window_state.yaw_penalty.mean(),
                                     window_state.distance_error.mean()) if window_state else (z, z, z))
        self.values[1:] += torch.stack((dones.sum().double(),
                                       (dones.bool() & ~time_outs.bool()).sum().double(),
                                       rewards.mean().double(), position.double(), turn.double(), distance.double()))

    def result(self):
        q, resets, failures, reward, position, turn, distance = self.values.cpu().tolist()
        return int(q), int(resets), int(failures), reward, position, turn, distance
