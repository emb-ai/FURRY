"""Per-reference full manipulation PPO. No pretraining or hidden teacher."""
import argparse,json,time,hashlib,traceback,os
from pathlib import Path

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--reference',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--device',default='cuda:0');ap.add_argument('--num-envs',type=int,default=4096);ap.add_argument('--iterations',type=int,default=12000)
    ap.add_argument('--seed',type=int,default=42);ap.add_argument('--smoke',action='store_true');ap.add_argument('--no-randomize',action='store_true');args=ap.parse_args()
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    def status(**v):
        p=out/'status.tmp';p.write_text(json.dumps(v,indent=2));p.replace(out/'status.json')
    status(status='initializing',pid=os.getpid(),reference=str(args.reference))
    try:
        manifest=json.loads((args.reference.parent/'manifest.json').read_text())
        member=[x for x in manifest['accepted'] if x['file']==args.reference.name]
        if len(member)!=1 or hashlib.sha256(args.reference.read_bytes()).hexdigest()!=member[0]['sha256']:
            raise ValueError('Reference is not a verified accepted demonstration')
        from isaaclab.app import AppLauncher
        app=AppLauncher(headless=True,device=args.device).app
        import numpy as np,torch
        from isaaclab.envs import ManagerBasedRLEnv
        from rsl_rl.algorithms import PPO
        from rsl_rl.modules import EmpiricalNormalization
        from .env import make_cfg
        from .network import actor_critic_class
        from .spec import JOINTS,LINKS,ACTOR_DIM,CRITIC_DIM
        torch.set_num_threads(4);torch.manual_seed(args.seed);np.random.seed(args.seed)
        cfg=make_cfg(args.reference.resolve(),args.num_envs,args.device,not args.no_randomize);cfg.seed=args.seed
        # IsaacLab 2.1 does not expose this importer setting in UrdfFileCfg.
        # Scope the override to conversion; do not patch the shared installation.
        from isaaclab.sim.converters import UrdfConverter
        original_import_config=UrdfConverter._get_urdf_import_config
        def import_inertia(converter):
            result=original_import_config(converter);result.set_import_inertia_tensor(True);return result
        UrdfConverter._get_urdf_import_config=import_inertia
        try:env=ManagerBasedRLEnv(cfg)
        finally:UrdfConverter._get_urdf_import_config=original_import_config
        obs,_=env.reset()
        assert obs['policy'].shape==(args.num_envs,ACTOR_DIM) and obs['critic'].shape==(args.num_envs,CRITIC_DIM),{k:v.shape for k,v in obs.items()}
        c=env.command_manager.get_term('motion');a=env.action_manager.get_term('joint_pos')
        # Check FK at first frame and a recorded grasp pose, independently of dynamics.
        native=dict(joint_names=env.scene['robot'].joint_names,body_names=env.scene['robot'].body_names,action_names=a._joint_names,default_q=a._offset[0].tolist(),scale=a._scale[0].tolist(),dt=env.physics_dt,control_dt=env.step_dt,gpu_dynamics_enabled=env.sim.get_physics_context().is_gpu_dynamics_enabled(),first_env_masses=c.robot.root_physx_view.get_masses()[0].tolist(),cup_mass=c.cup.root_physx_view.get_masses()[0].tolist())
        (out/'native-contract.json').write_text(json.dumps(native,indent=2))
        fk=[]
        for phase in (0,c.motion.time_step_total//2):
            c.time_steps[:]=phase
            root=torch.cat([c.body_pos_w[:,0],c.body_quat_w[:,0],c.body_lin_vel_w[:,0],c.body_ang_vel_w[:,0]],-1)
            c.robot.write_root_state_to_sim(root);c.robot.write_joint_state_to_sim(c.joint_pos,c.joint_vel);c.cup.write_root_state_to_sim(c.object_state)
            env.sim.forward();env.scene.update(0.)
            error=(c.body_pos_w-c.robot_body_pos_w).norm(dim=-1)
            fk.append(dict(phase=phase,max_m=float(error.max()),per_link_m=error.max(dim=0).values.tolist()))
        (out/'fk-check.json').write_text(json.dumps(fk,indent=2))
        if max(x['max_m'] for x in fk)>.005:raise RuntimeError(f'Imported robot FK differs: {fk}')
        obs,_=env.reset()
        spec=dict(schema='furry-dexweave-cup-box-v1',implementation='independent paper reconstruction, not released author code',paper='https://arxiv.org/html/2609.34724v1',
            reference_sha256=hashlib.sha256(args.reference.read_bytes()).hexdigest(),reference=str(args.reference),seed=args.seed,iterations=args.iterations,num_envs=args.num_envs,smoke=args.smoke,
            actor_dim=ACTOR_DIM,critic_dim=CRITIC_DIM,action_dim=43,joints=JOINTS,links=LINKS,
            settings=dict(physics_hz=200,control_hz=50,rollout_steps=24,epochs=5,minibatches=4,learning_rate=.001,desired_kl=.01,entropy=.002,clip=.2,gamma=.99,lam=.95),
            adaptations=['Physical robot recordings directly provide references; no human retargeting.','Three-finger Unitree hands: 14 independent drivers and six fingertips rather than Inspire 12 drivers/ten fingertips.','Archived scene collisions and inertia exported into URDF; PhysX collision solver differs from Quest MuJoCo.','Hand kp=1, kd=.03, torque .35 Nm: recorded hand embodiment, not author Inspire gains; explicit hand armature .001 for 200 Hz integration.','Body PD uses pinned BeyondMimic profile. Nominal hand-load compensation computed from archived link masses and COMs, applied only to waist/arms.','Eight fixed FK marker links use one milligram each to prevent importer unit-mass fallback.','Post-norm attention; torso+waist and waist+arm head routing where author details unspecified.','Separate policy per reference. No held-out demonstration generalization claim.'],
            randomize=not args.no_randomize)
        (out/'contract.json').write_text(json.dumps(spec,indent=2))
        policy=actor_critic_class()(ACTOR_DIM,CRITIC_DIM,43,init_noise_std=1.,actor_hidden_dims=[512,256,128],critic_hidden_dims=[512,256,128],activation='elu').to(args.device)
        print('ACTIVE_ACTOR',type(policy.actor).__name__,sum(p.numel() for p in policy.actor.parameters()),flush=True)
        norm=EmpiricalNormalization(shape=[ACTOR_DIM],until=1.e8).to(args.device);cnorm=EmpiricalNormalization(shape=[CRITIC_DIM],until=1.e8).to(args.device)
        alg=PPO(policy,num_learning_epochs=5,num_mini_batches=4,learning_rate=.001,schedule='adaptive',desired_kl=.01,clip_param=.2,gamma=.99,lam=.95,entropy_coef=.002,max_grad_norm=1.,value_loss_coef=1.,use_clipped_value_loss=True,device=args.device)
        alg.init_storage('rl',args.num_envs,24,[ACTOR_DIM],[CRITIC_DIM],[43])
        initial={k:v.detach().clone() for k,v in policy.actor.named_parameters()};started=time.monotonic()
        with (out/'metrics.jsonl').open('w',buffering=1) as log:
            for iteration in range(1,args.iterations+1):
                t=time.monotonic();rewards=[];fails=0;complete=0;object_error=[]
                for step in range(24):
                    with torch.inference_mode():
                        action=alg.act(norm(obs['policy']),cnorm(obs['critic']))
                        obs,reward,terminated,truncated,info=env.step(action)
                        if not torch.isfinite(obs['policy']).all() or not torch.isfinite(obs['critic']).all() or not torch.isfinite(reward).all():raise RuntimeError('Nonfinite environment state')
                        alg.process_env_step(reward,terminated|truncated,{'time_outs':truncated})
                        rewards.append(float(reward.mean()));fails+=int(terminated.sum());complete+=int((truncated&~terminated).sum())
                        object_error.append(float((c.object_state[:,:3]-c.cup.data.root_state_w[:,:3]).norm(dim=-1).mean()))
                with torch.inference_mode():alg.compute_returns(cnorm(obs['critic']))
                losses=alg.update()
                if not all(np.isfinite(v) for v in losses.values()):raise RuntimeError('Nonfinite PPO losses')
                changed=any(not torch.equal(initial[k],v) for k,v in policy.actor.named_parameters())
                row=dict(iteration=iteration,seconds=time.monotonic()-t,elapsed_s=time.monotonic()-started,reward_mean=sum(rewards)/24,failed=fails,completed_segments=complete,object_error_m=sum(object_error)/24,learning_rate=alg.learning_rate,losses=losses,actor_changed=changed)
                log.write(json.dumps(row)+'\n');print(json.dumps(row),flush=True)
                if iteration==1 or iteration%100==0 or iteration==args.iterations:
                    path=out/f'checkpoint-{iteration:05d}.pt';tmp=path.with_suffix('.tmp')
                    torch.save(dict(policy=policy.state_dict(),optimizer=alg.optimizer.state_dict(),normalizer=norm.state_dict(),critic_normalizer=cnorm.state_dict(),iteration=iteration,contract=spec,native_contract=native),tmp);tmp.replace(path)
                status(status='completed' if iteration==args.iterations else 'running',pid=os.getpid(),**row)
        assert changed
        env.close();app.close()
    except BaseException as e:
        status(status='failed',pid=os.getpid(),error=str(e),traceback=traceback.format_exc());raise
if __name__=='__main__':main()
