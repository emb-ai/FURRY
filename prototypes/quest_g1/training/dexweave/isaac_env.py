"""Body-only paper reconstruction on pinned BeyondMimic / IsaacLab 2.1.

Import only after AppLauncher. The observation/action order is anatomical,
resolved by names, independent of PhysX's breadth-first articulation order.
"""
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import numpy as np
import torch
from isaaclab.utils import configclass
from isaaclab.managers import ObservationGroupCfg,ObservationTermCfg,RewardTermCfg,TerminationTermCfg,EventTermCfg,SceneEntityCfg
from isaaclab.envs.mdp.actions import JointPositionAction,JointPositionActionCfg
from isaaclab.actuators import DelayedPDActuatorCfg,DelayedPDActuator
from isaaclab.utils.math import quat_apply,quat_inv,quat_mul,yaw_quat
import isaaclab.envs.mdp as lab_mdp
import whole_body_tracking.tasks.tracking.mdp as mdp
import whole_body_tracking.tasks.tracking.mdp.commands as command_module
from whole_body_tracking.tasks.tracking.config.g1.flat_env_cfg import G1FlatEnvCfg
from whole_body_tracking.robots.g1 import G1_ACTION_SCALE
from .spec import JOINTS,LINKS
from .sampling import capped_probabilities
from .motion_library import load_bank, ClipSampler

class StrengthPD(DelayedPDActuator):
    def __init__(self,cfg,*args,**kwargs):
        super().__init__(cfg,*args,**kwargs)
        low,high=cfg.strength_range
        self.strength=torch.rand_like(self.effort_limit)*(high-low)+low
    def _clip_effort(self,effort):
        return super()._clip_effort(effort*self.strength)

@configclass
class StrengthPDCfg(DelayedPDActuatorCfg):
    class_type=StrengthPD
    strength_range:tuple[float,float]=(.8,1.2)

class NamedMotion(command_module.MotionCommand):
    def __init__(self,cfg,env):
        robot=env.scene[cfg.asset_name]
        def loader(file,body_indexes,device):
            if Path(file).suffix == '.json':
                arrays,lengths,clips=load_bank(file,robot.joint_names,cfg.body_names)
                return SimpleNamespace(**{k:torch.tensor(v,dtype=torch.float32,device=device) for k,v in arrays.items()},
                    fps=50,time_step_total=sum(lengths),clip_lengths=lengths,clips=clips)
            with np.load(file) as f:
                if int(f['fps'])!=50:raise ValueError('Reference must be 50 Hz')
                joints=f['joint_names'].tolist();bodies=f['body_names'].tolist()
                ji=[joints.index(n) for n in robot.joint_names];bi=[bodies.index(n) for n in cfg.body_names]
                a={key:torch.tensor(f[key][:,ji if key.startswith('joint') else bi],dtype=torch.float32,device=device)
                   for key in ('joint_pos','joint_vel','body_pos_w','body_quat_w','body_lin_vel_w','body_ang_vel_w')}
                return SimpleNamespace(**a,fps=50,time_step_total=len(f['joint_pos']))
        # Parent constructor supplies tracked-body indices to its loader.
        # Scope the replacement to this synchronous constructor only.
        old=command_module.MotionLoader
        command_module.MotionLoader=loader
        try:super().__init__(cfg,env)
        finally:command_module.MotionLoader=old
        self.canonical_ids=torch.tensor([robot.joint_names.index(n) for n in JOINTS],device=self.device)
        self.failures=torch.zeros(self.bin_count,device=self.device)
        self.visits=torch.zeros_like(self.failures)
        self.anchor_history=torch.zeros(self.num_envs,7,3,device=self.device)
        self.delay=torch.zeros(self.num_envs,dtype=torch.long,device=self.device)
        self.history_step=-1
        self.sampled_start=torch.zeros(self.num_envs,dtype=torch.bool,device=self.device)
        self.clip_sampler=None
        self.last_step=self.motion.time_step_total-1
        if hasattr(self.motion,'clip_lengths'):
            self.clip_sampler=ClipSampler(self.motion.clip_lengths,self.num_envs,self.device)
            self.last_step=self.clip_sampler.ends
            self.time_steps[:]=self.clip_sampler.starts[self.clip_sampler.clip_ids]

    def _resample_command(self,env_ids):
        if len(env_ids)==0:return
        ids=torch.as_tensor(env_ids,device=self.device,dtype=torch.long);n=len(ids)
        if self.clip_sampler is not None:
            self.time_steps[ids],self.sampled_start[ids]=self.clip_sampler.sample(ids,self.time_steps,self._env.termination_manager.terminated[ids])
        else:
            old_bins=(self.time_steps[ids]//50).clamp(max=self.bin_count-1)
            self.visits+=torch.bincount(old_bins,minlength=self.bin_count)
            failed=self._env.termination_manager.terminated[ids]
            self.failures+=torch.bincount(old_bins[failed],minlength=self.bin_count)
            weights=torch.sqrt(self.failures/self.visits.clamp(min=1))
            weights=capped_probabilities(weights+1e-8)
            adaptive=torch.multinomial(weights,n,replacement=True)*50+torch.randint(0,50,(n,),device=self.device)
            uniform=torch.randint(0,self.motion.time_step_total,(n,),device=self.device)
            phase=torch.where(torch.rand(n,device=self.device)<.3,adaptive,uniform).clamp(max=self.motion.time_step_total-1)
            sampled=torch.rand(n,device=self.device)>=.5
            # Explicit interpretation of unspecified pre-roll execution: begin
            # tracking 0.5–1.5 s before the sampled phase, bounded at first frame.
            phase=(phase-torch.randint(25,76,(n,),device=self.device)).clamp(min=0)
            self.time_steps[ids]=torch.where(sampled,phase,0);self.sampled_start[ids]=sampled
        self.delay[ids]=torch.randint(0,7,(n,),device=self.device)
        self.anchor_history[ids]=0
        root=torch.cat([self.body_pos_w[ids,0],self.body_quat_w[ids,0],self.body_lin_vel_w[ids,0],self.body_ang_vel_w[ids,0]],-1)
        self.robot.write_joint_state_to_sim(self.joint_pos[ids],self.joint_vel[ids],env_ids=ids)
        self.robot.write_root_state_to_sim(root,env_ids=ids)
        self._align()

    def _align(self):
        delta=self.robot_anchor_pos_w.clone();delta[:,2]=self.anchor_pos_w[:,2]
        rot=yaw_quat(quat_mul(self.robot_anchor_quat_w,quat_inv(self.anchor_quat_w)))
        self.body_quat_relative_w=quat_mul(rot[:,None,:].expand(-1,14,-1),self.body_quat_w)
        self.body_pos_relative_w=delta[:,None,:]+quat_apply(rot[:,None,:].expand(-1,14,-1),self.body_pos_w-self.anchor_pos_w[:,None,:])

    def _update_command(self):
        self.time_steps=torch.minimum(self.time_steps+1,torch.as_tensor(self.last_step,device=self.device))
        self._align()

class BodyAction(JointPositionAction):
    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        assert tuple(self._joint_names)==JOINTS
        self.applied=torch.zeros_like(self._raw_actions);self.previous_applied=self.applied.clone()
        self.previous_raw=self.applied.clone();self.raw_second=self.applied.clone()
        self.previous_target=self._offset.clone()
        self.limit=torch.full((29,),float('inf'),device=self.device);self.limit[:15]=16.;self.limit[[2,8]]=20.
    def process_actions(self,actions):
        self.raw_second[:]=actions-2*self._raw_actions+self.previous_raw
        self.previous_raw[:]=self._raw_actions
        self.previous_applied[:]=self.applied
        super().process_actions(actions)
        # Defaults can be randomized at startup after construction.
        self._offset[:]=self._asset.data.default_joint_pos[:,self._joint_ids]
        desired=self._raw_actions*self._scale+self._offset
        self._processed_actions=torch.maximum(torch.minimum(desired,self.previous_target+self.limit*.02),self.previous_target-self.limit*.02)
        self.previous_target[:]=self._processed_actions
        self.applied[:]=(self._processed_actions-self._offset)/self._scale
    def reset(self,env_ids=None):
        super().reset(env_ids)
        if not hasattr(self,'applied'):return
        ids=slice(None) if env_ids is None else env_ids
        self.applied[ids]=0;self.previous_applied[ids]=0;self.previous_raw[ids]=0;self.raw_second[ids]=0
        self.previous_target[ids]=self._asset.data.joint_pos[ids][:,self._joint_ids]

def terms(env):
    return env.command_manager.get_term('motion'),env.action_manager.get_term('joint_pos')

def actor_obs(env):
    c,a=terms(env);i=c.canonical_ids;r=c.robot.data
    anchor=mdp.motion_anchor_pos_b(env,'motion')
    if c.history_step!=env.common_step_counter:
        c.anchor_history[:,1:]=c.anchor_history[:,:-1].clone();c.anchor_history[:,0]=anchor
        # Fill history for newly reset episodes, so delayed observation does
        # not temporarily point at a fictitious all-zero previous world.
        lengths=getattr(env,'episode_length_buf',None)
        new=torch.ones(env.num_envs,device=env.device,dtype=torch.bool) if lengths is None else lengths<=1
        c.anchor_history[new]=anchor[new,None,:]
        c.history_step=env.common_step_counter
    noisy=env.cfg.observations.policy.enable_corruption
    pos=c.anchor_history[torch.arange(env.num_envs,device=env.device),c.delay] if noisy else anchor
    ori=mdp.motion_anchor_ori_b(env,'motion');omega=r.root_ang_vel_b
    q=r.joint_pos[:,i]-r.default_joint_pos[:,i];dq=r.joint_vel[:,i]
    def noise(x,scale):return x+(torch.rand_like(x)*2-1)*scale if noisy else x
    return torch.cat([c.joint_pos[:,i],c.joint_vel[:,i],noise(pos,.05),noise(ori,.05),noise(omega,.2),noise(q,.01),noise(dq,.5),a.applied],-1)

def critic_obs(env):
    c,a=terms(env);i=c.canonical_ids;r=c.robot.data
    return torch.cat([c.joint_pos[:,i],c.joint_vel[:,i],mdp.motion_anchor_pos_b(env,'motion'),mdp.motion_anchor_ori_b(env,'motion'),
        mdp.robot_body_pos_b(env,'motion'),mdp.robot_body_ori_b(env,'motion'),r.root_lin_vel_b,r.root_ang_vel_b,
        r.joint_pos[:,i]-r.default_joint_pos[:,i],r.joint_vel[:,i],a.applied],-1)

def joint_reward(env):
    c,_=terms(env)
    return torch.exp(-torch.mean((c.joint_pos-c.robot_joint_pos)**2,-1)/.4**2)
def first_difference(env):
    _,a=terms(env);return ((a.applied-a.previous_applied)**2).sum(-1)
def second_difference(env):
    _,a=terms(env);return (a.raw_second[:,12:15]**2).sum(-1)
def reference_end(env):
    c,_=terms(env);return c.time_steps>=c.last_step

@configclass
class PolicyObs(ObservationGroupCfg):
    features=ObservationTermCfg(func=actor_obs)
    enable_corruption=True
    concatenate_terms=True
@configclass
class CriticObs(ObservationGroupCfg):
    features=ObservationTermCfg(func=critic_obs)
    concatenate_terms=True
@configclass
class Observations:
    policy=PolicyObs()
    critic=CriticObs()

def make_cfg(motion_file,num_envs,device,randomize=True):
    cfg=G1FlatEnvCfg();cfg.scene.num_envs=num_envs;cfg.sim.device=device
    # IsaacLab's timestamp + seeded RNG path collides for simultaneous jobs.
    cfg.scene.robot.spawn.usd_dir=tempfile.mkdtemp(prefix='furry-dexweave-urdf-')
    cfg.observations=Observations();cfg.observations.policy.enable_corruption=randomize
    cfg.commands.motion.class_type=NamedMotion;cfg.commands.motion.motion_file=str(motion_file);cfg.commands.motion.debug_vis=False
    cfg.scene.contact_forces.debug_vis=False
    cfg.scene.terrain.visual_material=None;cfg.scene.light=None;cfg.scene.sky_light=None
    cfg.actions.joint_pos=JointPositionActionCfg(class_type=BodyAction,asset_name='robot',joint_names=list(JOINTS),preserve_order=True,scale=G1_ACTION_SCALE,use_default_offset=True)
    for name,act in list(cfg.scene.robot.actuators.items()):
        cfg.scene.robot.actuators[name]=StrengthPDCfg(joint_names_expr=act.joint_names_expr,
            effort_limit=act.effort_limit_sim,effort_limit_sim=act.effort_limit_sim,velocity_limit_sim=act.velocity_limit_sim,
            stiffness=act.stiffness,damping=act.damping,armature=act.armature,min_delay=0,max_delay=2 if randomize else 0,
            strength_range=(.8,1.2) if randomize else (1.,1.))
    cfg.rewards.motion_global_anchor_pos.weight=1.;cfg.rewards.motion_global_anchor_pos.params['std']=.2
    cfg.rewards.motion_global_anchor_ori.weight=.75;cfg.rewards.motion_global_anchor_ori.params['std']=.3
    cfg.rewards.motion_joint_pos=RewardTermCfg(func=joint_reward,weight=1.)
    cfg.rewards.action_rate_l2.func=first_difference
    cfg.rewards.waist_second_difference=RewardTermCfg(func=second_difference,weight=-.05)
    cfg.terminations.reference_end=TerminationTermCfg(func=reference_end,time_out=True)
    if randomize:
        cfg.events.physics_material.params.update(static_friction_range=(.4,1.8),dynamic_friction_range=(.35,1.5),restitution_range=(0.,0.),make_consistent=True)
        cfg.events.add_joint_default_pos.params['pos_distribution_params']=(-.02,.02)
        cfg.events.push_robot.interval_range_s=(1.5,3.)
        cfg.events.push_robot.params['velocity_range']={'x':(-.4,.4),'y':(-.4,.4),'z':(0.,0.),'roll':(-.15,.15),'pitch':(-.15,.15),'yaw':(-.25,.25)}
        cfg.events.mass=EventTermCfg(func=lab_mdp.randomize_rigid_body_mass,mode='startup',params={'asset_cfg':SceneEntityCfg('robot',body_names='.*'),'mass_distribution_params':(.9,1.1),'operation':'scale'})
        cfg.events.gains=EventTermCfg(func=lab_mdp.randomize_actuator_gains,mode='startup',params={'asset_cfg':SceneEntityCfg('robot',joint_names=['.*']),'stiffness_distribution_params':(.85,1.15),'damping_distribution_params':(.75,1.25),'operation':'scale'})
        cfg.events.armature=EventTermCfg(func=lab_mdp.randomize_joint_parameters,mode='startup',params={'asset_cfg':SceneEntityCfg('robot',joint_names=['.*']),'armature_distribution_params':(.8,1.2),'operation':'scale'})
        cfg.events.joint_friction=EventTermCfg(func=lab_mdp.randomize_joint_parameters,mode='startup',params={'asset_cfg':SceneEntityCfg('robot',joint_names=['.*']),'friction_distribution_params':(0.,.08),'operation':'abs'})
        cfg.events.contact_offsets=EventTermCfg(func=lab_mdp.randomize_rigid_body_collider_offsets,mode='startup',params={'asset_cfg':SceneEntityCfg('robot',body_names=['left_ankle_roll_link','right_ankle_roll_link']),'contact_offset_distribution_params':(.005,.02)})
    else:
        cfg.events=None
    return cfg
