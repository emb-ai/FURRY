"""Bounded pilot: learn only a reference adapter using upstream PPO.

This is a new hierarchical controller experiment, not a TWIST2 finetune or an
implementation of ReActor. No train/validation reassignment is permitted.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--base-source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--iterations',type=int,default=300)
    p.add_argument('--save-every',type=int,default=100)
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--evaluate',type=Path)
    p.add_argument('--resume',type=Path)
    p.add_argument('--clearance-multiplier',type=float,default=1.)
    p.add_argument('--identity',action='store_true')
    p.add_argument('--eval-steps',type=int,default=600)
    a,rest=p.parse_known_args();root=a.root.resolve();out=a.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(a.base_source.resolve()));sys.argv=[sys.argv[0]]+rest
    os.chdir(root/'dataset')
    import isaacgym
    import torch
    from legged_gym.envs import G1MimicFuture
    from legged_gym.gym_utils import task_registry,get_args,class_to_dict
    from rsl_rl.modules import ActorCritic
    from rsl_rl.algorithms.ppo import PPO
    from policy import load
    from clearance_reward import install,gpu_contract_checks
    from reference_faults import install_nonlooping
    from adapter_env import AdapterEnv
    from continuation import set_clearance_multiplier,restore_optimizer
    if a.resume and (a.evaluate or a.identity):raise ValueError('Resume is training only')
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    args=get_args();torch.manual_seed(args.seed)
    manifest=json.loads((root/'dataset/manifest.json').read_text())
    split='validation' if a.evaluate or a.identity else 'train'
    rows=[r for r in manifest['clips'] if r['source']=='quest' and r['split']==split]
    if len(rows)!=(12 if split=='validation' else 66):raise ValueError('Unexpected verified Quest split')
    for r in rows:
        path=root/'dataset'/r['file']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=r['sha256']:raise ValueError('Dataset digest mismatch')
    yaml=out/'motions.yaml'
    yaml.write_text('root_path: '+str(root/'dataset')+'\nmotions:\n'+''.join(
        '- file: '+json.dumps(str((root/'dataset'/r['file']).resolve()))+'\n  weight: 1.0\n' for r in rows))
    cfg,tc=task_registry.get_cfgs('g1_stu_future')
    cfg.motion.motion_file=str(yaml);cfg.motion.sample_ratio=1.
    cfg.asset.file=str(root/'assets/g1_main_spheres.urdf')
    # Evaluation has deterministic initial RNG and no randomized disturbances.
    if split=='validation':
        for key in ['domain_rand_general','randomize_base_com','randomize_base_mass','randomize_friction',
                    'randomize_gravity','randomize_motor','push_robots','push_end_effector','action_delay']:
            setattr(cfg.domain_rand,key,False)
        cfg.noise.add_noise=False;cfg.motion.motion_curriculum=False
    env,cfg=task_registry.make_env('g1_stu_future',args=args,env_cfg=cfg)
    actual={str(Path(f).resolve()) for f in env._motion_lib._motion_files}
    expected={str((root/'dataset'/r['file']).resolve()) for r in rows}
    if actual!=expected:raise ValueError('Loader dropped/added clips')
    install_nonlooping(env)
    clearance=install(env,root/'assets/g1_main_spheres.urdf',treatment=True)
    checks=gpu_contract_checks(env,clearance)
    checks.update(set_clearance_multiplier(env,clearance,a.clearance_multiplier))
    tracker=load(root/'comparison-baseline.pt',root/'TWIST2/rsl_rl/rsl_rl/modules/actor_critic_future.py',env.device)
    frozen={k:v.clone() for k,v in tracker.state_dict().items()}
    wrapper=AdapterEnv(env,tracker,root/'assets/g1_main_spheres.urdf',root/'runtime-assets/scene.xml',args.seed+1700)
    wrapper.clearance=clearance
    # Independent physical FK and reference-routing checks before any update.
    fk,_=wrapper.fk(env.root_states[:,:3],env.root_states[:,3:7],env.dof_pos)
    body_ids=[wrapper.body_names.index(n) for n in wrapper.key_names+[wrapper.camera_body]]
    fk_error=float((fk-env.rigid_body_states[:,body_ids,:3]).abs().max())
    if fk_error>2e-4:raise ValueError('Key body/camera FK mismatch '+str(fk_error))
    physical_before=env.dof_pos.clone();root_before=env.root_states.clone()
    goal_before=wrapper.goal_pos.clone();goal_quat_before=wrapper.goal_quat.clone()
    wrapper.delta[:]=.03;wrapper.apply_reference()
    if not torch.equal(wrapper.command(True)[:,6:],env._ref_dof_pos):raise ValueError('Reference routing mismatch')
    if not torch.equal(env._ref_dof_pos[:,12:],wrapper.raw['dof_pos'][:,12:]):raise ValueError('Upper body modified')
    if not torch.equal(physical_before,env.dof_pos) or not torch.equal(root_before,env.root_states):raise ValueError('Adapter moved physics state')
    if not torch.equal(goal_before,wrapper.goal_pos) or not torch.equal(goal_quat_before,wrapper.goal_quat):raise ValueError('Adapter moved goal')
    wrapper.delta.zero_();wrapper.apply_reference()
    checks.update(key_body_and_camera_fk_max_abs_m=fk_error,corrected_reference_actor_reward_identical=True,
                  physical_state_unchanged_by_reference=True,independent_goal_unchanged=True,upper_body_reference_unchanged=True)
    obs=wrapper.observation();dim=obs.shape[1]
    ac=ActorCritic(num_prop=dim,num_critic_obs=dim,num_priv_latent=0,num_hist=0,num_actions=12,
        actor_hidden_dims=[256,128],critic_hidden_dims=[256,128],activation='elu',init_noise_std=.3,
        fix_action_std=False,priv_encoder_dims=[],tanh_encoder_output=False).to(env.device)
    # Exactly identity mean correction at initialization, not random foot angles.
    with torch.no_grad():
        ac.actor.actor_backbone[-1].weight.zero_();ac.actor.actor_backbone[-1].bias.zero_()
    loaded=None
    if a.evaluate or a.resume:
        loaded=torch.load(a.evaluate or a.resume,map_location=env.device,weights_only=True)
        if loaded['frozen_tracker_sha256']!=hashlib.sha256((root/'comparison-baseline.pt').read_bytes()).hexdigest():
            raise ValueError('Checkpoint frozen tracker mismatch')
        ac.load_state_dict(loaded['adapter'])
    algorithm=dict(num_learning_epochs=5,num_mini_batches=4,clip_param=.2,gamma=.99,lam=.95,
        value_loss_coef=1.,entropy_coef=.001,learning_rate=2e-4,max_grad_norm=1.,
        use_clipped_value_loss=True,schedule='adaptive',desired_kl=.008)
    alg=PPO(env,ac,device=env.device,**algorithm);steps=24
    start_iteration=restore_optimizer(alg,loaded) if a.resume else 0
    alg.init_storage(env.num_envs,steps,[dim],[dim],[12])
    config=dict(schema='furry-reference-adapter-v1',split=split,clips=[r['file'] for r in rows],
        frozen_tracker_sha256=hashlib.sha256((root/'comparison-baseline.pt').read_bytes()).hexdigest(),
        scene_sha256=hashlib.sha256((root/'runtime-assets/scene.xml').read_bytes()).hexdigest(),
        urdf_sha256=hashlib.sha256((root/'assets/g1_main_spheres.urdf').read_bytes()).hexdigest(),
        source_revision=manifest['source_revision'],environment=class_to_dict(cfg),algorithm=algorithm,
        adapter_observations=dim,adapter_actions=12,amplitude_rad=.5,lowpass_alpha=.2,
        modes={'recorded':.5,'held_position':.25,'held_turn':.25},hz=50,
        clearance_multiplier=a.clearance_multiplier,clearance_reward_scale=env.reward_scales['low_moving_foot'],
        continuation=dict(checkpoint=str(a.resume) if a.resume else None,start_iteration=start_iteration,
            checkpoint_sha256=hashlib.sha256(a.resume.read_bytes()).hexdigest() if a.resume else None,
            optimizer_restored=bool(a.resume),adaptive_learning_rate=alg.learning_rate,
            simulation_state_restored=False,additional_iterations=2 if a.smoke else a.iterations),
        notes=['Only Quest train for adapter pilot; original 33 + mirrored33, not66 independent demos.',
               'Raw Meta/GMR trajectories are precomputed; adapter operates causally on their robot references.',
               'Frozen tracker uses deployed history/normalization; adapter includes simulator state available in Quest simulation.',
               'Source imitation uses corrected q, joint velocities and FK key bodies; servo velocity rewards use commanded velocity.',
               'Additional independent camera-goal progress/distance and correction regularity rewards; new experimental objective.',
               'No weight update to TWIST2. No runtime integration or headset deployment.'])
    # Source config contains numpy.float64 values. Store plain JSON primitives
    # so our own checkpoints remain loadable with weights_only=True.
    config=json.loads(json.dumps(config))
    if a.resume:
        for key in ('clips','frozen_tracker_sha256','scene_sha256','urdf_sha256','adapter_observations','adapter_actions','hz'):
            if loaded['config'][key]!=config[key]:raise ValueError('Resume contract mismatch: '+key)
    (out/'config.json').write_text(json.dumps(config,indent=2));(out/'gpu-contract-checks.json').write_text(json.dumps(checks,indent=2))
    def assert_frozen():
        if any(not torch.equal(v,tracker.state_dict()[k]) for k,v in frozen.items()):raise ValueError('Frozen tracker changed')
        if any(p.grad is not None for p in tracker.parameters()):raise ValueError('Tracker gradient created')
    def save(iteration):
        assert_frozen();tmp=out/'checkpoint.tmp'
        torch.save({'adapter':ac.state_dict(),'optimizer':alg.optimizer.state_dict(),'iteration':iteration,
                    'adaptive_learning_rate':alg.learning_rate,
                    'config':config,'frozen_tracker_sha256':config['frozen_tracker_sha256']},tmp)
        restored=torch.load(tmp,map_location='cpu',weights_only=True)
        if set(restored['adapter'])!=set(ac.state_dict()):raise ValueError('Checkpoint roundtrip failed')
        tmp.rename(out/('adapter_%06d.pt'%iteration))
    started=time.time();metrics=[]
    def emit(record):
        record['elapsed_s']=time.time()-started
        with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        print(json.dumps(record),flush=True)
    if split=='validation':
        # Every clip in every mode, deterministic source start, first episode
        # only. Later divergent resets never enter the comparison statistics.
        if env.num_envs%36:raise ValueError('Evaluation requires a multiple of 12 clips x3 modes')
        ids=torch.arange(env.num_envs,device=env.device)
        env._rand_reset=False
        # G1MimicFuture.reset_idx only accepts env_ids. Select clips at the
        # actual sampling hook while preserving every subclass reset buffer.
        reset_reference=env._reset_ref_motion
        env._reset_ref_motion=lambda env_ids,motion_ids=None:reset_reference(env_ids,motion_ids=env_ids%12)
        try:env.reset_idx(ids)
        finally:env._reset_ref_motion=reset_reference
        wrapper.mode[:]=(ids//12)%3
        wrapper.goal_offset.zero_();wrapper.goal_yaw.zero_()
        sign=torch.where((ids//36)%2==0,1.,-1.)
        wrapper.goal_offset[:,0]=.6*sign*(wrapper.mode==1)
        wrapper.goal_yaw[:]=(torch.pi/6)*sign*(wrapper.mode==2)
        env.gym.refresh_actor_root_state_tensor(env.sim);env.gym.refresh_dof_state_tensor(env.sim)
        env.gym.refresh_rigid_body_state_tensor(env.sim)
        wrapper.original_update();wrapper.capture_raw();wrapper.apply_reference();obs=wrapper.observation()
        initial=dict(motion_ids=env._motion_ids.tolist(),mode=wrapper.mode.tolist(),
            root=env.root_states.tolist(),q=env.dof_pos.tolist(),goal=wrapper.goal_pos.tolist(),
            goal_quat=wrapper.goal_quat.tolist(),obs=obs.tolist())
        (out/'initial-state.json').write_text(json.dumps(initial))
        sums={};counts=torch.zeros(3,device=env.device);fail=torch.zeros_like(counts)
        alive=torch.ones(env.num_envs,dtype=torch.bool,device=env.device)
        final_distance=torch.zeros(env.num_envs,device=env.device);final_yaw=final_distance.clone()
        durations=final_distance.clone();per_env_fail=torch.zeros_like(alive)
        with torch.no_grad():
            for _ in range(a.eval_steps):
                action=torch.zeros(env.num_envs,12,device=env.device) if a.identity else ac.act_inference(obs)
                obs,reward,done,info=wrapper.step(action)
                m=wrapper.last_metrics
                for mode in range(3):
                    mask=(m['mode']==mode)&alive;counts[mode]+=mask.sum();fail[mode]+=m['failure'][mask].sum()
                    for key in ('distance_m','yaw_error_rad','progress_m_s','delta_rms_rad','low_moving_cost','contact_slip_cost'):
                        sums.setdefault(key,torch.zeros(3,device=env.device))[mode]+=m[key][mask].sum()
                final_distance[alive]=m['distance_m'][alive];final_yaw[alive]=m['yaw_error_rad'][alive]
                durations[alive]+=env.dt;per_env_fail|=alive&m['failure'];alive&=~done.bool()
                if not alive.any():break
        assert_frozen()
        result=dict(kind='identity' if a.identity else 'candidate',steps=a.eval_steps,
            frames=counts.tolist(),failures=fail.tolist(),mean={k:(v/counts.clamp(min=1)).tolist() for k,v in sums.items()},
            clearance=clearance.metrics(),frozen_tracker_unchanged=True,
            final_distance_m=final_distance.tolist(),final_yaw_error_rad=final_yaw.tolist(),
            durations_s=durations.tolist(),per_env_failure=per_env_fail.tolist(),
            initial_state_sha256=hashlib.sha256((out/'initial-state.json').read_bytes()).hexdigest(),
            limitation='First episode per paired clip/mode; clearance aggregate includes later reset episodes and is diagnostic only. Dataset is the existing already-inspected validation session, not fresh human data.')
        (out/'result.json').write_text(json.dumps(result,indent=2));emit(result);return
    save(start_iteration);iterations=2 if a.smoke else a.iterations
    initial_actor=ac.actor.actor_backbone[-1].weight.detach().clone()
    for iteration in range(start_iteration+1,start_iteration+iterations+1):
        rates=[];failures=0
        with torch.no_grad():
            for _ in range(steps):
                action=alg.act(obs,obs,{})
                obs,reward,done,info=wrapper.step(action)
                alg.process_env_step(reward,done,info)
                rates.append(reward.mean());failures+=int(wrapper.last_metrics['failure'].sum())
            alg.compute_returns(obs)
        loss=alg.update();assert_frozen()
        if not all(torch.isfinite(p).all() for p in ac.parameters()):raise FloatingPointError('Invalid adapter weights')
        emit(dict(iteration=iteration,reward_rate=float(torch.stack(rates).mean()/env.dt),
            failures=failures,value_loss=loss[0],surrogate_loss=loss[1],lr=alg.learning_rate,
            delta_rms_rad=float(wrapper.delta.square().mean().sqrt()),
            distance_m=float(wrapper.last_metrics['distance_m'].mean()),clearance=clearance.metrics()))
        if iteration%a.save_every==0 or iteration==start_iteration+iterations:save(iteration)
    changed=not torch.equal(initial_actor,ac.actor.actor_backbone[-1].weight)
    if not changed:raise ValueError('Adapter optimizer never changed the actor')
    result=dict(completed=True,iterations=iterations,smoke=a.smoke,adapter_actor_updated=changed,
                start_iteration=start_iteration,final_iteration=start_iteration+iterations,
                frozen_tracker_unchanged=True,headset_deployment=False)
    (out/'result.json').write_text(json.dumps(result,indent=2));emit(result)


if __name__=='__main__':main()
