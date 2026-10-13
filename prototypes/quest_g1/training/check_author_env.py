"""Run original TWIST2 Isaac Gym, checking data lineage and released actor parity.

Only frozen-policy inference; never optimizer updates. CWD is the portable dataset.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--steps', type=int, default=500)
    options, rest = p.parse_known_args()
    root = options.root.resolve()
    sys.argv = [sys.argv[0]] + rest
    os.chdir(root / 'dataset')
    import isaacgym  # Must precede torch.
    import torch
    import numpy as np
    import onnxruntime as ort
    from legged_gym.envs import G1MimicFuture
    from legged_gym.gym_utils import task_registry, get_args, class_to_dict
    from rsl_rl.modules import ActorCriticFuture
    from policy import recover

    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    args = get_args()
    cfg, train_cfg = task_registry.get_cfgs('g1_stu_future')
    cfg.motion.motion_file = str(root / 'dataset/train.yaml')
    cfg.motion.sample_ratio = 1.0
    env, cfg = task_registry.make_env('g1_stu_future', args=args, env_cfg=cfg)
    manifest = json.loads((root / 'dataset/manifest.json').read_text())
    if manifest.get('schema') != 'twist2-quest-mixture-v2':
        raise ValueError('Dataset must have verified root-oriented PKL export')
    expected = {str((root / 'dataset' / c['file']).resolve()) for c in manifest['clips'] if c['split'] == 'train'}
    actual = {str(Path(f).resolve()) for f in env._motion_lib._motion_files}
    if expected != actual:
        raise ValueError(f'MotionLib silently changed data: missing={len(expected-actual)}, extra={len(actual-expected)}')
    if any('validation' in f for f in actual):
        raise ValueError('Validation loaded by trainer')
    source = root / 'TWIST2/rsl_rl/rsl_rl/modules/actor_critic_future.py'
    onnx = root / 'TWIST2/assets/ckpts/twist2_1017_20k.onnx'
    actor = recover(onnx, source).to(env.device).eval()
    ort_options = ort.SessionOptions(); ort_options.intra_op_num_threads = 1; ort_options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(onnx), sess_options=ort_options, providers=['CPUExecutionProvider'])
    obs = env.get_observations()
    max_parity = 0.0
    parity_ok = True
    resets = 0
    timeouts = 0
    reward_sum = 0.0
    observations = []
    for step in range(options.steps):
        with torch.no_grad():
            action = actor(obs)
        if step < 10:
            raw = obs[:8].cpu().numpy()
            expected_a = session.run(None, {session.get_inputs()[0].name: raw})[0]
            parity_ok = parity_ok and np.allclose(action[:8].cpu().numpy(), expected_a, atol=2e-4, rtol=2e-4)
            max_parity = max(max_parity, float(np.max(abs(action[:8].cpu().numpy() - expected_a))))
            observations.append(raw)
        if not torch.isfinite(action).all() or not torch.isfinite(obs).all():
            raise ValueError('Nonfinite actor data')
        obs, privileged, rewards, dones, info = env.step(action)
        resets += int(dones.sum())
        timeouts += int((info['time_outs'] & dones.bool()).sum())
        reward_sum += float(rewards.mean())
    if not parity_ok:
        raise ValueError(f'ONNX / Torch actor mismatch {max_parity}')
    report = {'phase':'original_env_frozen_smoke_passed','training_updates':0,'actor_sha256':hashlib.sha256(onnx.read_bytes()).hexdigest(),'actor_parity_max_abs':max_parity,'loaded_train_motions':len(actual),'loaded_validation_motions':0,'num_envs':env.num_envs,'obs':env.num_obs,'privileged_obs':env.num_privileged_obs,'policy_dt':env.dt,'steps':options.steps,'resets':resets,'timeouts':timeouts,'tracking_terminations':resets-timeouts,'reward_rate_mean':reward_sum/options.steps/env.dt,'reward_scales':env.reward_scales,'torch':torch.__version__}
    (root/'frozen-smoke.json').write_text(json.dumps(report,indent=2))
    np.savez_compressed(root/'frozen-smoke-obs.npz', obs=np.concatenate(observations))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
