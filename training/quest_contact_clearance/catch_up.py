"""Training copy of the headset catch-up loop (camera_pose_servo.h, v2 branch).

The Quest app adds a bounded pose correction to the actor's root command:
the reference xy velocity is re-expressed in the robot root frame, the xy
position error (robot frame, capped .4 m/s, total .8 m/s) is added to it, and
3x the yaw error (capped 1, total 1.2 rad/s) is added to the yaw rate.
Without this loop in training the actor never sees a position correction and
ignores it on the headset (111586: 0.17 m/s commanded, ~0.05 achieved).

Differences from the headset, by construction: goal/actual are the reference
and robot ROOT here, the headset uses the operator's mapped head and the robot
ego camera. Reward targets are unchanged; only the actor command is modified.
At a reset frame the source places the robot on the reference while
_ref_root_pos still lacks episode_init_origin, so that frame gets no correction.
"""
import torch


def matrix_xyzw(q):
    x, y, z, w = q.unbind(-1)
    return torch.stack((
        1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
        2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
        2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), -1).reshape(*q.shape[:-1], 3, 3)


def yaw_xyzw(q):
    x, y, z, w = q.unbind(-1)
    return torch.atan2(2*(w*z+x*y), w*w+x*x-y*y-z*z)


def servo(command, ref_pos, ref_quat, robot_pos, robot_quat):
    """CameraPoseServo with a reference quaternion; quaternions are xyzw."""
    out = command.clone()
    R = matrix_xyzw(robot_quat)
    local = torch.zeros(len(command), 3, dtype=R.dtype, device=command.device)
    local[:, :2] = command[:, :2]
    world = (matrix_xyzw(ref_quat) @ local[..., None])[..., 0]
    velocity = (R.transpose(-1, -2) @ world[..., None])[..., 0][:, :2]
    error = torch.zeros_like(local)
    error[:, :2] = ref_pos[:, :2]-robot_pos[:, :2]
    correction = (R.transpose(-1, -2) @ error[..., None])[..., 0][:, :2]
    speed = torch.linalg.vector_norm(correction, dim=-1, keepdim=True)
    correction = correction*torch.where(speed > .4, .4/speed.clamp(min=1e-12), torch.ones_like(speed))
    velocity = velocity+correction
    total = torch.linalg.vector_norm(velocity, dim=-1, keepdim=True)
    velocity = velocity*torch.where(total > .8, .8/total.clamp(min=1e-12), torch.ones_like(total))
    yaw = yaw_xyzw(ref_quat)-yaw_xyzw(robot_quat)
    yaw = torch.atan2(torch.sin(yaw), torch.cos(yaw))
    out[:, :2] = velocity.to(out.dtype)
    out[:, 5] = (command[:, 5]+(3*yaw).clamp(-1, 1).to(out.dtype)).clamp(-1.2, 1.2)
    return out


class CatchUp:
    """Per-environment catch-up, enabled with probability `fraction` at each reset."""
    def __init__(self, count, device, fraction, seed):
        if not 0 < fraction <= 1:
            raise ValueError('Catch-up fraction must be in (0,1]')
        self.fraction = float(fraction)
        self.generator = torch.Generator(device=device).manual_seed(seed)
        self.enabled = torch.zeros(count, dtype=torch.bool, device=device)
        self.fresh = torch.ones(count, dtype=torch.bool, device=device)
        self.stats = torch.zeros(3, dtype=torch.float64, device=device)  # applied frames, |dv| sum, saturated frames

    def reset(self, ids):
        draw = torch.rand(len(ids), generator=self.generator, device=self.enabled.device)
        self.enabled[ids] = draw < self.fraction
        self.fresh[ids] = True

    def apply(self, command, ref_pos, ref_quat, robot_pos, robot_quat):
        corrected = servo(command, ref_pos, ref_quat, robot_pos, robot_quat)
        use = self.enabled & ~self.fresh
        self.fresh[:] = False
        delta = torch.linalg.vector_norm(corrected[:, :2]-command[:, :2], dim=-1)
        self.stats += torch.stack((use.sum(), delta[use].sum(), (use & (delta > .399)).sum())).double()
        return torch.where(use[:, None], corrected, command)


def install(env, fraction, seed):
    """Returns (state, transform) for native_command_bank.install(transform=...)."""
    state = CatchUp(env.num_envs, env.device, fraction, seed)
    state.reset(torch.arange(env.num_envs, device=env.device))
    original_reset = env.reset_idx

    def reset_idx(ids, *args, **kwargs):
        original_reset(ids, *args, **kwargs)
        state.reset(ids)

    def transform(command):
        return state.apply(command, env._ref_root_pos, env._ref_root_rot, env.root_states[:, :3], env.root_states[:, 3:7])

    env.reset_idx = reset_idx
    return state, transform
