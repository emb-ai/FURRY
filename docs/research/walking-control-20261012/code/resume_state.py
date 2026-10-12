"""Restore learning state; legacy checkpoints cannot restore physical rollouts."""
import json
import math
from pathlib import Path
import torch


def restore(ck, checkpoint_path, ac, alg, env, cfg, args):
    old_config = json.loads((checkpoint_path.parent/'config.json').read_text())
    old_args = old_config['finetune_overrides']
    for key in ('leg_faults','window_reward','nonlooping','fast_critic'):
        if bool(old_args.get(key)) != bool(getattr(args,key)):
            raise ValueError('Resume condition changed: '+key)
    if ck['phase'] not in ('actor','complete'):
        raise ValueError('Only actor-phase checkpoints can be continued')
    ac.load_state_dict(ck['actor_critic'],strict=True)
    for param in ac.parameters():param.requires_grad_(True)
    ac.std.requires_grad_(not ac.if_fix_std())
    alg.optimizer.load_state_dict(ck['optimizer'])
    rows=[json.loads(x) for x in (checkpoint_path.parent/'metrics.jsonl').read_text().splitlines()]
    actor_updates=int(ck['iteration'])
    rows=[x for x in rows if 'value_loss' in x and (x['phase']=='critic' or x.get('actor_updates',0)<actor_updates)]
    if not rows or rows[-1]['phase']!='actor' or rows[-1]['actor_updates']+1 != actor_updates:
        raise ValueError('Checkpoint and metrics actor counters disagree')
    last=rows[-1]
    state=ck.get('continuation_state')
    alg.counter=int(state['algorithm_counter'] if state else last['iteration'])
    alg.learning_rate=float(state['learning_rate'] if state else last['lr'])
    if any(not math.isclose(g['lr'],alg.learning_rate,rel_tol=1e-12) for g in alg.optimizer.param_groups):
        raise ValueError('Optimizer LR disagrees with continuation LR')
    alg.schedule=old_config['upstream_training']['algorithm']['schedule']
    if state:
        alg.dagger_coef=state['dagger_coef']
        total=state['total_env_steps_counter']
    else:
        alg.dagger_coef=old_config['upstream_training']['algorithm']['dagger_coef']
        for i in range(1,alg.counter+1):
            alg.dagger_coef=(alg.dagger_coef*.5*(1+math.cos(math.pi*i/alg.dagger_coef_anneal_steps))
                             if i<alg.dagger_coef_anneal_steps else alg.dagger_coef_min)
        if last['effective_noise_fraction']>=1:
            raise ValueError('Cannot recover saturated legacy step counter')
        total=round(last['effective_noise_fraction']*cfg.noise.noise_increasing_steps*24)
    env.total_env_steps_counter=int(total)
    env.global_counter=int(state['global_counter'] if state else total)
    env.common_step_counter=int(state['common_step_counter'] if state else total)
    env.delay=torch.tensor(float(last['effective_action_delay_steps']),device=env.device)
    if state:
        for name in ('motion_difficulty','motion_termination_dist'):
            getattr(env,name).copy_(state[name].to(env.device))
        env.mean_motion_difficulty=env.motion_difficulty.mean()
    cm=ck['critic_mean'].to(env.device);cd=ck['critic_divisor'].to(env.device)
    if cm.numel()!=old_config['environment']['env']['num_privileged_obs']+old_config['window_reward']['critic_extra_features'] or cm.shape!=cd.shape or not torch.isfinite(cm).all() or not (cd>0).all():
        raise ValueError('Invalid restored critic normalization')
    report={'actor_updates':actor_updates,'algorithm_counter':alg.counter,'learning_rate':alg.learning_rate,
            'actor_critic_std_optimizer_restored':True,'actor_and_critic_normalization_restored':True,
            'total_env_steps_counter':env.total_env_steps_counter,'legacy_counter_reconstructed':not bool(state),
            'physical_rollout_reset':True,'rng_restored':False,'motion_curriculum_restored':bool(state),
            'legacy_limitation':'Physical state/history and RNG reset; legacy motion difficulty was not saved.' if not state else 'Physical state/history and RNG reset.'}
    return cm,cd,actor_updates,alg.counter,report


def capture(alg,env,cfg):
    return {'algorithm_counter':alg.counter,'learning_rate':alg.learning_rate,'dagger_coef':alg.dagger_coef,
            'total_env_steps_counter':env.total_env_steps_counter,'global_counter':env.global_counter,
            'common_step_counter':env.common_step_counter,'action_delay_schedule':'HumanoidChar.step counter-ramped Bernoulli; 5000..20000 iterations',
            'motion_difficulty':env.motion_difficulty.detach().cpu().clone(),
            'motion_termination_dist':env.motion_termination_dist.detach().cpu().clone()}
