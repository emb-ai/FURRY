"""Explicit source-reward ablations; coefficient1 leaves source methods intact."""
import inspect
import textwrap
import torch


def install(cls, coefficient=1.0, yaw_airtime=False, yaw_threshold=.15):
    if not 0 < coefficient <= 1:
        raise ValueError('lower-body coefficient must be in (0,1]')
    originals = {}

    def moving(self):
        return (torch.linalg.vector_norm(self._ref_root_vel[:, :2], dim=-1) > .05) | (self._ref_root_ang_vel[:, 2].abs() > yaw_threshold)

    def joint_weights(self):
        w = torch.ones((self.num_envs, self.num_dof), device=self.device)
        w[:, :12] = torch.where(moving(self), coefficient, 1.)[:, None]
        return w

    def body_weights(self):
        names = list(self.cfg.motion.key_bodies)
        lower = [name in ('left_ankle_roll_link', 'right_ankle_roll_link', 'left_knee_link', 'right_knee_link') for name in names]
        if sum(lower) != 4 or len(names) != len(self._key_body_ids):
            raise ValueError('Unexpected G1 key-body contract')
        mask = torch.tensor(lower, device=self.device)[None, :] & moving(self)[:, None]
        return torch.where(mask, coefficient, 1.)

    cls._locomotion_mask = moving
    cls._leg_joint_weights = joint_weights
    cls._leg_body_weights = body_weights

    def patch(name, old, new):
        fn = getattr(cls, name)
        source = textwrap.dedent(inspect.getsource(fn))
        if source.count(old) != 1:
            raise ValueError('Pinned source formula changed: '+name)
        import linecache
        filename = '<locomotion-ablation:'+name+'>'
        patched = source.replace(old, new)
        linecache.cache[filename] = (len(patched), None, patched.splitlines(True), filename)
        namespace = dict(fn.__globals__)
        exec(compile(source.replace(old, new), '<locomotion-ablation:'+name+'>', 'exec'), namespace)
        originals[name] = fn
        setattr(cls, name, namespace[name])

    if coefficient != 1.:
        patch('_reward_tracking_joint_dof', 'dof_diff * dof_diff', 'dof_diff * dof_diff * self._leg_joint_weights()')
        patch('_reward_tracking_joint_vel', 'vel_diff * vel_diff', 'vel_diff * vel_diff * self._leg_joint_weights()')
        for name in ('_reward_tracking_keybody_pos', '_reward_tracking_keybody_pos_global'):
            patch(name, 'torch.sum(key_body_pos_diff * key_body_pos_diff, dim=-1)', 'torch.sum(key_body_pos_diff * key_body_pos_diff, dim=-1) * self._leg_body_weights()')
        patch('_error_tracking_joint_dof', 'torch.abs(dof_diff)', 'torch.abs(dof_diff) * self._leg_joint_weights()')
        patch('_error_tracking_joint_vel', 'torch.abs(vel_diff)', 'torch.abs(vel_diff) * self._leg_joint_weights()')
        patch('_error_tracking_keybody_pos', 'torch.mean(key_body_pos_diff, dim=-1)', 'torch.mean(key_body_pos_diff * self._leg_body_weights(), dim=-1)')
        patch('check_termination', 'body_pos_dist = torch.sum(body_pos_diff * body_pos_diff, dim=-1)', 'body_pos_dist = torch.sum(body_pos_diff * body_pos_diff, dim=-1) * self._leg_body_weights()')
    if yaw_airtime:
        patch('_reward_feet_air_time', 'torch.norm(self._ref_root_vel[:, :2], dim=1) > 0.05', 'self._locomotion_mask()')
    return originals


def validate_contract(env):
    names = env.gym.get_actor_dof_names(env.envs[0], env.actor_handles[0])
    legs = ['left_'+x+'_joint' for x in ('hip_pitch','hip_roll','hip_yaw','knee','ankle_pitch','ankle_roll')]
    legs += ['right_'+x+'_joint' for x in ('hip_pitch','hip_roll','hip_yaw','knee','ankle_pitch','ankle_roll')]
    if list(names[:12]) != legs or len(names) != 29:
        raise ValueError('G1 leg joint ordering differs')
    return {'leg_dofs':list(names[:12]), 'lower_key_bodies':[x for x in env.cfg.motion.key_bodies if 'knee' in x or 'ankle' in x]}
