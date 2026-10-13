"""Offline local-log PPO runner; no cloud tracking or automatic deployment."""
import argparse,json,time,hashlib,os
from pathlib import Path

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--reference',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--device',default='cuda:7');ap.add_argument('--num-envs',type=int,default=4096);ap.add_argument('--iterations',type=int,default=8000)
    ap.add_argument('--seed',type=int,default=42);ap.add_argument('--smoke',action='store_true');args=ap.parse_args()
    world=int(os.environ.get('WORLD_SIZE','1'));rank=int(os.environ.get('RANK','0'))
    if world>1:
        # Isolate Isaac/PhysX to one physical device per torchrun worker.
        visible=os.environ['CUDA_VISIBLE_DEVICES'].split(',')
        os.environ['CUDA_VISIBLE_DEVICES']=visible[int(os.environ['LOCAL_RANK'])]
        args.device='cuda:0'
    from isaaclab.app import AppLauncher
    application=AppLauncher(headless=True,device=args.device).app
    import numpy as np
    import torch
    from isaaclab.envs import ManagerBasedRLEnv
    from rsl_rl.algorithms import PPO
    from rsl_rl.modules import EmpiricalNormalization
    from .isaac_env import make_cfg
    from .network import actor_critic_class
    from .spec import contract,LINKS
    if world>1:
        import torch.distributed as dist
        from datetime import timedelta
        from .distributed import GlobalNormalization,synchronize_policy,replica_hashes
        torch.cuda.set_device(0)
        dist.init_process_group('nccl',timeout=timedelta(minutes=10))
    local_seed=args.seed+rank
    torch.set_num_threads(4);torch.manual_seed(local_seed);np.random.seed(local_seed)
    cfg=make_cfg(args.reference,args.num_envs,args.device);cfg.seed=local_seed
    out=args.output if rank==0 else args.output/f'rank-{rank}';out.mkdir(parents=True,exist_ok=True)
    spec=contract();spec.update(reference_sha256=hashlib.sha256(args.reference.read_bytes()).hexdigest(),seed=args.seed,
        num_envs=args.num_envs,world_size=world,global_rank=rank,global_num_envs=args.num_envs*world,local_seed=local_seed,iterations=args.iterations,smoke=args.smoke,
        implementation_details=['Pinned BeyondMimic G1 body asset and PD constants; no dexterous hand model.',
            'Pre-roll interpreted as starting 0.5–1.5 s before sampled target phase, included in episode duration.',
            'Explicit delayed PD actuators; motor strength/body mass/gains/armature/friction/contact offsets/COM/push/noise randomization.',
            'Body-only G1 asset has no Inspire hand load: Inspire payload/COM randomization and nominal hand-load compensation omitted.'])
    (out/'contract.json').write_text(json.dumps(spec,indent=2)+'\n')
    env=ManagerBasedRLEnv(cfg=cfg)
    obs,_=env.reset();assert obs['policy'].shape==(args.num_envs,157) and obs['critic'].shape==(args.num_envs,286)
    c=env.command_manager.get_term('motion');a=env.action_manager.get_term('joint_pos')
    if c.clip_sampler is not None:
        spec.update(reference_kind='training_manifest',training_references=c.motion.clips,
            initialization='fresh actor, critic, optimizer, observation normalizers',
            sampling='fixed balanced environment quotas; clip-local adaptive phase bins (at least four)',
            environments_per_clip=torch.bincount(c.clip_sampler.clip_ids,minlength=len(c.motion.clips)).cpu().tolist())
        (out/'contract.json').write_text(json.dumps(spec,indent=2)+'\n')
    native={'joint_names':env.scene['robot'].joint_names,'body_names':env.scene['robot'].body_names,
        'action_names':a._joint_names,'default_q':a._offset[0].cpu().tolist(),'scale':a._scale[0].cpu().tolist(),
        'tracked_body_names':list(LINKS),'dt':env.physics_dt,'control_dt':env.step_dt,
        'gpu_dynamics_enabled':env.sim.get_physics_context().is_gpu_dynamics_enabled(),
        'broadphase':env.sim.get_physics_context().get_broadphase_type(),
        'physx_tensor_device':str(env.scene['robot'].root_physx_view.get_dof_positions().device)}
    (out/'native-contract.json').write_text(json.dumps(native,indent=2)+'\n')
    env.sim.forward();env.scene.update(0.)
    position_error=torch.linalg.vector_norm(c.body_pos_w-c.robot_body_pos_w,dim=-1)
    fkmax=float(position_error.max())
    (out/'fk-check.json').write_text(json.dumps({'max_link_position_error_m':fkmax,'mean_m':float(position_error.mean())},indent=2)+'\n')
    if fkmax>.005:raise RuntimeError(f'Reference/model FK mismatch {fkmax:.6f} m')
    policy=actor_critic_class()(157,286,29,init_noise_std=1.,actor_hidden_dims=[512,256,128],critic_hidden_dims=[512,256,128],activation='elu').to(args.device)
    print('ACTIVE_ACTOR',type(policy.actor).__name__,'parameters',sum(p.numel() for p in policy.actor.parameters()),flush=True)
    if world>1:synchronize_policy(policy)
    Normalizer=GlobalNormalization if world>1 else EmpiricalNormalization
    norm=Normalizer(shape=[157],until=1.e8).to(args.device)
    cnorm=Normalizer(shape=[286],until=1.e8).to(args.device)
    alg=PPO(policy,num_learning_epochs=5,num_mini_batches=4,learning_rate=.001,schedule='adaptive',desired_kl=.01,
        clip_param=.2,gamma=.99,lam=.95,entropy_coef=.005,max_grad_norm=1.,value_loss_coef=1.,use_clipped_value_loss=True,device=args.device,
        multi_gpu_cfg={'global_rank':rank,'world_size':world} if world>1 else None)
    alg.init_storage('rl',args.num_envs,24,[157],[286],[29])
    previous={k:v.detach().clone() for k,v in policy.actor.state_dict().items() if v.is_floating_point()}
    cumulative_fail=0;cumulative_complete=0;started=time.monotonic()
    status={'world_size':world,'global_num_envs':args.num_envs*world,'status':'running','iteration':0,'reference':str(args.reference),'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    def save(iteration):
        if rank!=0:return
        payload={'policy':policy.state_dict(),'optimizer':alg.optimizer.state_dict(),'normalizer':norm.state_dict(),
                 'critic_normalizer':cnorm.state_dict(),'iteration':iteration,'contract':spec,'native_contract':native}
        target=out/f'checkpoint-{iteration:05d}.pt';tmp=target.with_suffix('.tmp');torch.save(payload,tmp);tmp.replace(target)
    with (out/'metrics.jsonl').open('w',buffering=1) as log:
        for iteration in range(1,args.iterations+1):
            t=time.monotonic();rewards=[];fails=0;complete=0
            for step in range(24):
                with torch.inference_mode():
                    action=alg.act(norm(obs['policy']),cnorm(obs['critic']))
                    obs,reward,terminated,truncated,info=env.step(action)
                    if not torch.isfinite(obs['policy']).all() or not torch.isfinite(reward).all():raise RuntimeError('Nonfinite environment state')
                    alg.process_env_step(reward,terminated|truncated,{'time_outs':truncated})
                    rewards.append(float(reward.mean()));fails+=int(terminated.sum());complete+=int((truncated&~terminated).sum())
            with torch.inference_mode():alg.compute_returns(cnorm(obs['critic']))
            losses=alg.update()
            if not all(np.isfinite(v) for v in losses.values()):raise RuntimeError('Nonfinite PPO loss')
            cumulative_fail+=fails;cumulative_complete+=complete
            record={'iteration':iteration,'seconds':time.monotonic()-t,'elapsed_s':time.monotonic()-started,'reward_mean':sum(rewards)/24,
                    'terminated':fails,'completed':complete,'total_terminated':cumulative_fail,'total_completed':cumulative_complete,'learning_rate':alg.learning_rate,
                    'anchor_position_error_m':float(c.metrics['error_anchor_pos'].mean()),'losses':losses}
            if c.clip_sampler is not None:
                record['training_clip_ids']=[x['id'] for x in c.motion.clips]
                record['clip_resets']=c.clip_sampler.resets.cpu().tolist()
                record['clip_transitions']=[n*24*iteration for n in spec['environments_per_clip']]
            if world>1:
                totals=torch.tensor([sum(rewards)/24,fails,complete,float(c.metrics['error_anchor_pos'].mean())],device=args.device,dtype=torch.float64)
                dist.all_reduce(totals)
                record['global_reward_mean']=float(totals[0]/world)
                record['global_terminated']=int(totals[1]);record['global_completed']=int(totals[2])
                record['global_anchor_position_error_m']=float(totals[3]/world)
                record['global_transitions']=args.num_envs*world*24*iteration
                if iteration==1 or iteration%100==0 or iteration==args.iterations:
                    record['replica_sha256']=replica_hashes(policy,norm,cnorm)
            log.write(json.dumps(record)+'\n');print(json.dumps(record),flush=True)
            if iteration==1 or iteration%100==0 or iteration==args.iterations:save(iteration)
            status.update(status='completed' if iteration==args.iterations else 'running',**record)
            status['actor_changed']=any(not torch.equal(previous[k],v) for k,v in policy.actor.state_dict().items() if k in previous)
            tmp=out/'status.tmp';tmp.write_text(json.dumps(status,indent=2)+'\n');tmp.replace(out/'status.json')
    if world>1:dist.destroy_process_group()
    env.close();application.close()

if __name__=='__main__':main()
