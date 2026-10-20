"""Frozen mean-policy diagnostic from frame zero; no mid-rollout state writes.

Continue after tracking thresholds to expose the actual outcome. Training is
untouched. Full-reference tracking is not a native Quest task-success metric.
"""
import argparse, hashlib, json, os, time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reference', type=Path, required=True)
    ap.add_argument('--checkpoints', type=Path, nargs='+', required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--num-envs', type=int, default=1)
    ap.add_argument('--randomize', action='store_true')
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    def save(name, value):
        tmp = args.output / (name + '.tmp')
        tmp.write_text(json.dumps(value, indent=2))
        tmp.replace(args.output / name)
    save('status.json', dict(status='initializing', pid=os.getpid()))
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, device=args.device).app
    import numpy as np
    import torch
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sim.converters import UrdfConverter
    from isaaclab.utils.math import quat_apply, quat_inv, quat_error_magnitude
    from rsl_rl.modules import EmpiricalNormalization
    from .env import make_cfg, failed
    from .network import actor_critic_class
    from .spec import ACTOR_DIM, CRITIC_DIM
    torch.set_num_threads(2)
    torch.manual_seed(42)
    np.random.seed(42)
    cfg = make_cfg(args.reference.resolve(), args.num_envs, args.device, args.randomize)
    cfg.seed = 42
    def never(env):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    cfg.terminations.failure.func = never
    cfg.terminations.end.func = never
    old = UrdfConverter._get_urdf_import_config
    def importer(converter):
        result = old(converter)
        result.set_import_inertia_tensor(True)
        return result
    UrdfConverter._get_urdf_import_config = importer
    try:
        env = ManagerBasedRLEnv(cfg)
    finally:
        UrdfConverter._get_urdf_import_config = old
    c = env.command_manager.get_term('motion')
    a = env.action_manager.get_term('joint_pos')
    policy = actor_critic_class()(ACTOR_DIM, CRITIC_DIM, 43, init_noise_std=1., actor_hidden_dims=[512,256,128], critic_hidden_dims=[512,256,128], activation='elu').to(args.device)
    norm = EmpiricalNormalization(shape=[ACTOR_DIM], until=1.e8).to(args.device)
    summary = dict(protocol='mean policy, frozen normalization, frame-zero physical initialization, no automatic resets, full reference horizon', randomize=args.randomize, device=args.device, num_envs=args.num_envs, reference_sha256=hashlib.sha256(args.reference.read_bytes()).hexdigest(), results=[])
    for checkpoint in args.checkpoints:
        ck = torch.load(checkpoint, map_location=args.device, weights_only=False)
        assert ck['contract']['reference_sha256'] == summary['reference_sha256']
        assert list(a._joint_names) == ck['native_contract']['action_names']
        policy.load_state_dict(ck['policy'], strict=True)
        norm.load_state_dict(ck['normalizer'], strict=True)
        policy.eval(); norm.eval()
        torch.manual_seed(42); np.random.seed(42)
        env.reset()
        c.time_steps[:] = 0; c.starts[:] = 0; c.sampled_start[:] = False
        root = torch.cat([c.body_pos_w[:,0], c.body_quat_w[:,0], c.body_lin_vel_w[:,0], c.body_ang_vel_w[:,0]], -1)
        c.robot.write_root_state_to_sim(root)
        c.robot.write_joint_state_to_sim(c.joint_pos, c.joint_vel)
        c.cup.write_root_state_to_sim(c.object_state)
        env.sim.forward(); env.scene.update(0.)
        c._align(); a.reset()
        anchor = quat_apply(quat_inv(c.robot_anchor_quat_w), c.anchor_pos_w-c.robot_anchor_pos_w)
        c.anchor_history[:] = anchor[:,None]
        c.object_history[:] = c.object_state[:,None,:7]
        c.measurement[:] = c.object_state[:,:7]
        c.delayed_objects[:] = c.object_state[:,None,:7]
        c.last_sample = -1; c.observation_step = -1
        env.episode_length_buf[:] = 0
        fk = float((c.body_pos_w-c.robot_body_pos_w).norm(dim=-1).max())
        assert fk < .005, fk
        obs = env.observation_manager.compute()
        rows = []; first = [None] * args.num_envs
        initial = (c.cup.data.root_pos_w-env.scene.env_origins).clone()
        started = time.monotonic()
        for step in range(c.motion.time_step_total-1):
            with torch.inference_mode():
                action = policy.act_inference(norm(obs['policy']))
                assert torch.isfinite(action).all()
                obs, reward, terminated, truncated, info = env.step(action)
                assert not (terminated | truncated).any(), 'Unexpected automatic reset'
                assert (c.time_steps == step+1).all(), 'Reference phase jumped'
                cp = c.cup.data.root_state_w
                assert torch.isfinite(cp).all() and torch.isfinite(obs['policy']).all()
                bad = failed(env)
                for i in range(args.num_envs):
                    if bool(bad[i]) and first[i] is None:
                        first[i] = (step+1)*env.step_dt
                rq = c.robot.data.root_quat_w
                upright = 1-2*(rq[:,1].square()+rq[:,2].square())
                row = dict(t=(step+1)*env.step_dt, cup=(cp[:,:3]-env.scene.env_origins).cpu().tolist(), reference=(c.object_state[:,:3]-env.scene.env_origins).cpu().tolist(), error=(cp[:,:3]-c.object_state[:,:3]).norm(dim=-1).cpu().tolist(), orientation_error=quat_error_magnitude(cp[:,3:7], c.object_state[:,3:7]).cpu().tolist(), root_z=(c.robot.data.root_pos_w[:,2]-env.scene.env_origins[:,2]).cpu().tolist(), upright=upright.cpu().tolist(), failed=bad.cpu().tolist())
                rows.append(row)
            if step % 100 == 0:
                save('status.json', dict(status='running', checkpoint=str(checkpoint), iteration=ck['iteration'], frame=step+1, total_frames=c.motion.time_step_total-1, first_tracking_failure_s=first, elapsed_s=time.monotonic()-started))
        cups = np.array([r['cup'] for r in rows]); errors = np.array([r['error'] for r in rows])
        refs = np.array([r['reference'] for r in rows])
        result = dict(checkpoint=str(checkpoint), sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(), iteration=ck['iteration'], duration_s=len(rows)*env.step_dt, elapsed_s=time.monotonic()-started, initial_fk_error_m=fk, first_tracking_failure_s=first, full_reference_tracking_pass=[x is None for x in first], final_cup_error_m=errors[-1].tolist(), mean_cup_error_m=errors.mean(0).tolist(), max_cup_lift_m=(cups[:,:,2].max(0)-initial[:,2].cpu().numpy()).tolist(), reference_max_lift_m=(refs[:,:,2].max(0)-initial[:,2].cpu().numpy()).tolist(), final_cup_xyz=cups[-1].tolist(), final_reference_xyz=refs[-1].tolist(), min_root_z_m=np.array([r['root_z'] for r in rows]).min(0).tolist(), task_success_evaluated=False)
        save('rollout-%05d.json' % ck['iteration'], dict(summary=result, rows=rows))
        summary['results'].append(result); save('results.json', summary)
        print(json.dumps(result), flush=True)
    save('status.json', dict(status='completed', results=len(summary['results'])))
    env.close(); app.close()


if __name__ == '__main__':
    main()
