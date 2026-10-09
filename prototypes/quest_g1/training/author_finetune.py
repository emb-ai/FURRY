"""Small-data finetune using pinned TWIST2 simulator, reward and DaggerPPO.

Explicit finetune changes: pretrained/frozen actor normalization, new critic
normalization, critic-only warmup, lower actor LR with warmup, std=.05.
No teacher, added slip/fall reward, or custom simulator approximation.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--smoke', action='store_true', help='Only 2 critic updates + 1 actor update in disposable output; not critic readiness')
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--iterations', type=int, default=1000)
    p.add_argument('--critic-warmup', type=int, default=500)
    p.add_argument('--critic-max', type=int, default=1000)
    p.add_argument('--actor-lr', type=float, default=1e-5)
    p.add_argument('--actor-warmup', type=int, default=100)
    p.add_argument('--initial-std', type=float, default=.05)
    p.add_argument('--save-every', type=int, default=100)
    a, rest = p.parse_known_args()
    root = a.root.resolve(); out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    sys.argv = [sys.argv[0]] + rest
    os.chdir(root/'dataset')
    import isaacgym
    import torch
    import numpy as np
    from legged_gym.envs import G1MimicFuture
    from legged_gym.gym_utils import task_registry, get_args, class_to_dict
    from rsl_rl.modules import ActorCriticFuture
    from rsl_rl.algorithms import DaggerPPO
    from policy import recover
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    args = get_args()
    cfg, tc = task_registry.get_cfgs('g1_stu_future')
    cfg.motion.motion_file = str(root/'dataset/train.yaml')
    cfg.motion.sample_ratio = 1.
    env, cfg = task_registry.make_env('g1_stu_future', args=args, env_cfg=cfg)
    manifest = json.loads((root/'dataset/manifest.json').read_text())
    if manifest.get('schema') != 'twist2-quest-mixture-v2':
        raise ValueError('Dataset must have verified root-oriented PKL export')
    rows = [c for c in manifest['clips'] if c['split']=='train']
    expected = {str((root/'dataset'/c['file']).resolve()) for c in rows}
    actual = {str(Path(f).resolve()) for f in env._motion_lib._motion_files}
    if expected != actual: raise ValueError('Upstream loader skipped or added motions')
    if any('validation' in x for x in actual): raise ValueError('Validation leakage')
    released = recover(root/'TWIST2/assets/ckpts/twist2_1017_20k.onnx',root/'TWIST2/rsl_rl/rsl_rl/modules/actor_critic_future.py').to(env.device)
    ac = ActorCriticFuture(num_observations=env.num_obs,num_critic_observations=env.num_privileged_obs,num_motion_observations=cfg.env.n_mimic_obs,num_motion_steps=len(cfg.env.tar_motion_steps),num_priop_observations=cfg.env.n_proprio,num_history_steps=cfg.env.history_len,num_actions=env.num_actions,**class_to_dict(tc.policy)).to(env.device)
    ac.actor.load_state_dict(released.actor.state_dict(),strict=True)
    ac.actor.eval()
    frozen_actor = {k:v.detach().clone() for k,v in ac.actor.state_dict().items()}
    ac.std.data.fill_(a.initial_std)
    ac.std.requires_grad_(False)
    for param in ac.actor.parameters(): param.requires_grad_(False)
    algorithm = class_to_dict(tc.algorithm)
    algorithm.update(learning_rate=3e-4,schedule='fixed',use_clipped_value_loss=False)
    alg = DaggerPPO(env,ac,ac,teacher_loaded=False,device=env.device,**algorithm)
    steps = tc.runner.num_steps_per_env
    alg.init_storage(env.num_envs,steps,[env.num_obs],[env.num_privileged_obs],[env.num_actions])
    obs = env.get_observations()
    priv = env.get_privileged_observations()
    # Freeze actor affine transform exactly. Fit a separate critic transform once.
    def actor_obs(x): return (x-released.mean)/released.divisor
    s = torch.zeros(env.num_privileged_obs,device=env.device,dtype=torch.float64)
    ss = torch.zeros_like(s); count = 0
    with torch.no_grad():
        for _ in range(128):
            z = priv.double(); s += z.sum(0); ss += z.square().sum(0); count += len(z)
            action = ac.act(actor_obs(obs))
            obs,priv,_,_,_ = env.step(action)
    cm = (s/count).float(); cd = torch.sqrt(torch.clamp(ss/count-(s/count).square(),min=1e-4)).float()
    def critic_obs(x): return (x-cm)/cd
    def emit(record):
        record['wall_seconds'] = time.time()-started
        with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        print(json.dumps(record),flush=True)
    def save(name,phase,iteration):
        # Compatible with our independent 100Hz MuJoCo evaluator/exporter.
        released.actor.load_state_dict(ac.actor.state_dict())
        ck={'policy':released.state_dict(),'actor_critic':ac.state_dict(),'critic_mean':cm,'critic_divisor':cd,'optimizer':alg.optimizer.state_dict(),'phase':phase,'iteration':iteration,'source_revision':manifest['source_revision'],'dataset_manifest_sha256':hashlib.sha256((root/'dataset/manifest.json').read_bytes()).hexdigest(),'critic_restored':False}
        tmp=out/(name+'.tmp');torch.save(ck,tmp);tmp.rename(out/name)
    started=time.time()
    save('baseline.pt','baseline',0)
    config={'source_revision':manifest['source_revision'],'source_actor_sha256':hashlib.sha256((root/'TWIST2/assets/ckpts/twist2_1017_20k.onnx').read_bytes()).hexdigest(),'dataset_manifest_sha256':hashlib.sha256((root/'dataset/manifest.json').read_bytes()).hexdigest(),'environment':class_to_dict(cfg),'upstream_training':class_to_dict(tc),'finetune_overrides':{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},'critic_gate':'On-policy pre-update TD-lambda prediction, 50-iteration windows; proxy only, not independent Monte Carlo certification. Validation never enters optimizer.','deployment':False}
    (out/'config.json').write_text(json.dumps(config,indent=2,default=str))
    counts=torch.zeros(len(rows),device=env.device,dtype=torch.long)
    quest_ids=torch.tensor([i for i,f in enumerate(env._motion_lib._motion_files) if str(f).startswith('./quest/') or str(f).startswith('quest/')],device=env.device)
    phase='critic'; iteration=0; stable=0; window=[]; actor_updates=0
    while True:
        if (out/'STOP').exists():
            emit({'phase':'stopped_by_file','actor_updates':actor_updates});save('stopped.pt',phase,iteration);return
        iteration+=1
        value_loss_previous=[]; resets=0; failures=0; reward_sum=0.; quest_frames=0
        with torch.no_grad():
            for _ in range(steps):
                ids=env._motion_ids.clone(); counts.scatter_add_(0,ids,torch.ones_like(ids))
                quest_frames+=int(torch.isin(ids,quest_ids).sum())
                actions=alg.act(actor_obs(obs),critic_obs(priv),{})
                obs,priv,r,d,info=env.step(actions)
                alg.process_env_step(r,d,info)
                resets+=int(d.sum());failures+=int((d.bool() & ~info['time_outs'].bool()).sum());reward_sum+=float(r.mean())
            alg.compute_returns(critic_obs(priv))
            values=alg.storage.values.flatten(); targets=alg.storage.returns.flatten()
            mse=float((values-targets).square().mean());var=float(targets.var(unbiased=False));ev=1-float((values-targets).var(unbiased=False))/max(var,1e-8)
            nrmse=(mse/max(var,1e-8))**.5
            old_means=alg.storage.mu.detach().clone(); saved_obs=alg.storage.observations.detach().clone()
        if phase=='actor':
            lr=a.actor_lr*min(1.,(actor_updates+1)/a.actor_warmup)
        else:lr=3e-4
        alg.learning_rate=lr
        for group in alg.optimizer.param_groups:group['lr']=lr
        result=alg.update()
        with torch.no_grad():
            current=ac.actor(saved_obs.reshape(-1,env.num_obs)).reshape_as(old_means)
            update_kl=float(((current-old_means).square()/(2*ac.std.square())).sum(-1).mean())
        if not np.isfinite([result[0],result[1],ev,nrmse,update_kl]).all():
            raise FloatingPointError('Nonfinite optimization statistics')
        record={'phase':phase,'iteration':iteration,'actor_updates':actor_updates,'lr':lr,'value_loss':result[0],'surrogate_loss':result[1],'pre_update_critic_ev':ev,'pre_update_critic_nrmse':nrmse,'post_update_kl':update_kl,'reward_rate':reward_sum/steps/env.dt,'resets':resets,'tracking_terminations':failures,'quest_frame_fraction':quest_frames/(steps*env.num_envs),'visited_motions':int((counts>0).sum()),'std_mean':float(ac.std.mean())}
        emit(record)
        if phase=='critic':
            if a.smoke and iteration==2:
                save('smoke-critic.pt',phase,iteration)
                for param in ac.actor.parameters():param.requires_grad_(True)
                phase='actor'
            if any(not torch.equal(v,frozen_actor[k]) for k,v in ac.actor.state_dict().items()):
                raise RuntimeError('Actor weights changed during critic warmup')
            window.append((ev,nrmse)); window=window[-50:]
            if iteration%50==0:
                gate=(iteration>=a.critic_warmup and np.mean([x[0] for x in window])>=.7 and np.mean([x[1] for x in window])<=.65 and bool((counts>0).all()))
                stable=stable+1 if gate else 0
                save('critic-latest.pt',phase,iteration)
                if stable>=2:
                    save('critic-ready.pt',phase,iteration)
                    phase='actor'
                    for param in ac.actor.parameters():param.requires_grad_(True)
                    alg.use_clipped_value_loss=tc.algorithm.use_clipped_value_loss
                    # Fresh Adam state for the newly unfrozen actor, keeping critic moments.
                    emit({'phase':'critic_ready','warmup_iterations':iteration})
                elif iteration>=a.critic_max:
                    emit({'phase':'critic_not_ready','actor_updates':0});return
        else:
            actor_updates+=1
            if actor_updates%a.save_every==0:
                save(f'checkpoint_{actor_updates:06d}.pt',phase,actor_updates)
            if update_kl>.05:
                save('large-update.pt',phase,actor_updates)
                emit({'phase':'stopped_large_policy_update','actor_updates':actor_updates,'kl':update_kl});return
            if actor_updates>=(1 if a.smoke else a.iterations):
                save('final.pt','complete',actor_updates);emit({'phase':'complete','actor_updates':actor_updates});return


if __name__=='__main__':main()
