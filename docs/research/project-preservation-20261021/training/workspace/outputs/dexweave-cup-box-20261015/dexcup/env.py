"""Full body + physical three-finger hands + dynamic cup, IsaacLab 2.1."""
import json,tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import isaaclab.sim as sim
from isaaclab.assets import AssetBaseCfg,RigidObjectCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply,quat_inv,quat_mul,yaw_quat,matrix_from_quat,quat_error_magnitude,quat_from_euler_xyz
from isaaclab.managers import ObservationGroupCfg,ObservationTermCfg,RewardTermCfg,TerminationTermCfg,SceneEntityCfg,EventTermCfg
from isaaclab.envs.mdp.actions import JointPositionAction,JointPositionActionCfg
from . import base_env as base
from .spec import JOINTS,LINKS

def rot6(q):return matrix_from_quat(q)[...,:,:2].flatten(-2)
def pose(p,q,anchor_p,anchor_q):
    inv=quat_inv(anchor_q)
    return torch.cat([quat_apply(inv,p-anchor_p),rot6(quat_mul(inv,q))],-1)

class Motion(base.NamedMotion):
    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        self.object_ref=torch.tensor(np.load(cfg.motion_file)['object_state'],device=self.device,dtype=torch.float32)
        self.cup=env.scene['cup'];self.starts=torch.zeros(self.num_envs,device=self.device,dtype=torch.long)
        self.object_history=torch.zeros(self.num_envs,4,7,device=self.device)
        self.object_history[:,:,3]=1
        self.measurement=self.object_history[:,0].clone()
        self.object_delay=torch.ones(self.num_envs,device=self.device,dtype=torch.long)
        self.last_sample=-1;self.observation_step=-1
    def _align(self):
        delta=self.robot_anchor_pos_w.clone();delta[:,2]=self.anchor_pos_w[:,2]
        rot=yaw_quat(quat_mul(self.robot_anchor_quat_w,quat_inv(self.anchor_quat_w)))
        qs=rot[:,None,:].expand(-1,len(LINKS),-1)
        self.body_quat_relative_w=quat_mul(qs,self.body_quat_w)
        self.body_pos_relative_w=delta[:,None,:]+quat_apply(qs,self.body_pos_w-self.anchor_pos_w[:,None,:])
    @property
    def object_state(self):
        x=self.object_ref[self.time_steps].clone();x[:,:3]+=self._env.scene.env_origins;return x
    def _resample_command(self,env_ids):
        super()._resample_command(env_ids)
        ids=torch.as_tensor(env_ids,device=self.device,dtype=torch.long)
        self.starts[ids]=self.time_steps[ids]
        state=self.object_state[ids];self.cup.write_root_state_to_sim(state,env_ids=ids)
        self.object_history[ids]=state[:,None,:7];self.measurement[ids]=state[:,:7]
        self.object_delay[ids]=torch.randint(1,4,(len(ids),),device=self.device)
        # Fill the seven-sample history at reset; no zero-world sentinel.
        anchor=quat_apply(quat_inv(self.robot_anchor_quat_w[ids]),self.anchor_pos_w[ids]-self.robot_anchor_pos_w[ids])
        self.anchor_history[ids]=anchor[:,None,:]

class Action(JointPositionAction):
    def __init__(self,cfg,env):
        super().__init__(cfg,env);assert tuple(self._joint_names)==JOINTS
        self.applied=torch.zeros_like(self._raw_actions);self.previous_applied=self.applied.clone()
        self.previous_raw=self.applied.clone();self.second=self.applied.clone()
        self.target=self._offset.clone()
        self.speed=torch.full((43,),float('inf'),device=self.device);self.speed[:15]=16;self.speed[[2,8]]=20
        asset=Path(env.cfg.commands.motion.motion_file).parent/'assets/hand_compensation.json'
        comp=json.loads(asset.read_text())
        self.hand_ids=[self._asset.body_names.index(n) for n in comp['hand_bodies']]
        self.joint_body_ids=[self._asset.body_names.index(n) for n in comp['joint_bodies']]
        self.com=torch.tensor(comp['com'],device=self.device,dtype=torch.float32)
        self.axes=torch.tensor(comp['axis'],device=self.device,dtype=torch.float32)
        self.hand_load=torch.tensor(comp['mass'],device=self.device,dtype=torch.float32)*9.81
        self.descendants=torch.tensor(comp['mask'],device=self.device,dtype=torch.float32)
    def process_actions(self,actions):
        self.second[:]=actions-2*self._raw_actions+self.previous_raw
        self.previous_raw[:]=self._raw_actions;self.previous_applied[:]=self.applied
        super().process_actions(actions)
        self._offset[:]=self._asset.data.default_joint_pos[:,self._joint_ids]
        desired=actions*self._scale+self._offset
        self._processed_actions=torch.minimum(torch.maximum(desired,self.target-self.speed*.02),self.target+self.speed*.02)
        self.target[:]=self._processed_actions;self.applied[:]=(self.target-self._offset)/self._scale
    def apply_actions(self):
        super().apply_actions()
        r=self._asset.data
        hp=r.body_pos_w[:,self.hand_ids]+quat_apply(r.body_quat_w[:,self.hand_ids],self.com[None].expand(self.num_envs,-1,-1))
        jp=r.body_pos_w[:,self.joint_body_ids]
        axes=quat_apply(r.body_quat_w[:,self.joint_body_ids],self.axes[None].expand(self.num_envs,-1,-1))
        lever=hp[:,None]-jp[:,:,None]
        # cross(lever, nominal upward force), then project on joint axis.
        moment=torch.stack([lever[...,1],-lever[...,0],torch.zeros_like(lever[...,0])],-1)
        torques=((moment*axes[:,:,None]).sum(-1)*self.hand_load[None,None]*self.descendants[None]).sum(-1)
        self._asset.set_joint_effort_target(torques,joint_ids=self._joint_ids[12:29])
    def reset(self,env_ids=None):
        super().reset(env_ids)
        if not hasattr(self,'applied'):return
        i=slice(None) if env_ids is None else env_ids
        for v in (self.applied,self.previous_applied,self.previous_raw,self.second):v[i]=0
        self.target[i]=self._asset.data.joint_pos[i][:,self._joint_ids]

def terms(env):return env.command_manager.get_term('motion'),env.action_manager.get_term('joint_pos')
def object_features(c,measured=False):
    actual=c.measurement if measured else c.cup.data.root_state_w[:,:7]
    ref=c.object_state
    x=pose(actual[:,:3],actual[:,3:7],c.robot_anchor_pos_w,c.robot_anchor_quat_w)
    y=pose(ref[:,:3],ref[:,3:7],c.robot_anchor_pos_w,c.robot_anchor_quat_w)
    return torch.cat([x,y,x-y],-1)
def actor_obs(env):
    c,a=terms(env);i=c.canonical_ids;r=c.robot.data;noisy=env.cfg.observations.policy.enable_corruption
    def noise(x,s):return x+(2*torch.rand_like(x)-1)*s if noisy else x
    anchor=base.mdp.motion_anchor_pos_b(env,'motion')
    if c.observation_step!=env.common_step_counter:
        c.anchor_history[:,1:]=c.anchor_history[:,:-1].clone();c.anchor_history[:,0]=noise(anchor,.05)
        # Object samples at 30 Hz, held between samples, then delayed 20–60 ms.
        sample=int(env.common_step_counter*30/50)
        if sample!=c.last_sample:
            actual=c.cup.data.root_state_w[:,:7].clone();actual[:,:3]=noise(actual[:,:3],.01)
            if noisy:
                angles=(torch.rand(env.num_envs,3,device=env.device)*2-1)*.035
                actual[:,3:]=quat_mul(quat_from_euler_xyz(*angles.unbind(-1)),actual[:,3:])
            c.object_history[:,1:]=c.object_history[:,:-1].clone();c.object_history[:,0]=actual;c.last_sample=sample
        # Stored on the control clock, to enforce 1–3 control-step latency.
        if not hasattr(c,'delayed_objects'):c.delayed_objects=c.object_history[:,0,None,:].expand(-1,4,-1).clone()
        c.delayed_objects[:,1:]=c.delayed_objects[:,:-1].clone();c.delayed_objects[:,0]=c.object_history[:,0]
        lengths=getattr(env,'episode_length_buf',None)
        new=torch.ones(env.num_envs,dtype=torch.bool,device=env.device) if lengths is None else lengths<=1
        c.delayed_objects[new]=c.cup.data.root_state_w[new,None,:7]
        c.measurement[:]=c.delayed_objects[torch.arange(env.num_envs,device=env.device),c.object_delay]
        c.observation_step=env.common_step_counter
    hist=c.anchor_history.flatten(1)
    return torch.cat([c.joint_pos[:,i],c.joint_vel[:,i[:29]],hist,noise(base.mdp.motion_anchor_ori_b(env,'motion'),.05),noise(r.root_ang_vel_b,.2),noise(r.joint_pos[:,i]-r.default_joint_pos[:,i],.01),noise(r.joint_vel[:,i[:29]],.5),a.applied,object_features(c,noisy)],-1)
def critic_obs(env):
    c,a=terms(env);i=c.canonical_ids;r=c.robot.data
    return torch.cat([c.joint_pos[:,i],c.joint_vel[:,i],base.mdp.motion_anchor_pos_b(env,'motion'),base.mdp.motion_anchor_ori_b(env,'motion'),base.mdp.robot_body_pos_b(env,'motion'),base.mdp.robot_body_ori_b(env,'motion'),r.root_lin_vel_b,r.root_ang_vel_b,r.joint_pos[:,i]-r.default_joint_pos[:,i],r.joint_vel[:,i],a.applied,object_features(c)],-1)

def err_reward(error,scale):return torch.exp(-error/scale**2)
def reward(env,kind):
    c,a=terms(env);ids=c.canonical_ids
    if kind in ('body_joint','hand_joint'):
        idx=ids[:29] if kind=='body_joint' else ids[29:]
        return err_reward(((c.joint_pos[:,idx]-c.robot_joint_pos[:,idx])**2).mean(-1),.4 if kind=='body_joint' else .5)
    if kind=='first':return ((a.applied-a.previous_applied)**2).sum(-1)
    if kind=='waist_second':return a.second[:,12:15].square().sum(-1)
    if kind=='legs_second':return a.second[:,:12].square().sum(-1)
    obj=c.object_state;act=c.cup.data.root_state_w
    if kind=='object_pos':return err_reward((obj[:,:3]-act[:,:3]).square().sum(-1),.15)
    if kind=='object_ori':return err_reward(quat_error_magnitude(obj[:,3:7],act[:,3:7]).square(),.9)
    values=[]
    for side,wrist in enumerate((10,13)):
        if kind.startswith('hand'):
            idx=list(range(14+side*3,17+side*3));n=3
            refp=c.body_pos_w[:,idx];refq=c.body_quat_w[:,idx];p=c.robot_body_pos_w[:,idx];q=c.robot_body_quat_w[:,idx]
            rp=c.body_pos_w[:,wrist,None,:].expand(-1,n,-1);rq=c.body_quat_w[:,wrist,None,:].expand(-1,n,-1)
            ap=c.robot_body_pos_w[:,wrist,None,:].expand(-1,n,-1);aq=c.robot_body_quat_w[:,wrist,None,:].expand(-1,n,-1)
        else:
            refp=obj[:,:3];refq=obj[:,3:7];p=act[:,:3];q=act[:,3:7]
            rp=c.body_pos_w[:,wrist];rq=c.body_quat_w[:,wrist];ap=c.robot_body_pos_w[:,wrist];aq=c.robot_body_quat_w[:,wrist]
        if kind.endswith('pos'):
            e=(quat_apply(quat_inv(rq),refp-rp)-quat_apply(quat_inv(aq),p-ap)).square().sum(-1)
        else:e=quat_error_magnitude(quat_mul(quat_inv(rq),refq),quat_mul(quat_inv(aq),q)).square()
        if e.ndim>1:e=e.mean(-1)
        values.append(e)
    scale=.4 if kind.endswith('ori') else (.1 if kind.startswith('hand') else .2)
    return err_reward(torch.stack(values,-1).mean(-1),scale)
def failed(env):
    c,_=terms(env);obj=c.object_state;act=c.cup.data.root_state_w
    delta=c.anchor_pos_w-c.robot_anchor_pos_w
    heights=(c.body_pos_w[:,[3,6,10,13],2]-c.robot_body_pos_w[:,[3,6,10,13],2]).abs().amax(-1)
    return (delta[:,2].abs()>.25)|(delta[:,:2].norm(dim=-1)>.5)|(quat_error_magnitude(c.anchor_quat_w,c.robot_anchor_quat_w)>.7)|(heights>.25)|((obj[:,:3]-act[:,:3]).norm(dim=-1)>.3)|(quat_error_magnitude(obj[:,3:7],act[:,3:7])>2)
def ended(env):
    c,_=terms(env)
    return (c.time_steps>=c.last_step)|(c.sampled_start & (c.time_steps-c.starts>=500))

@configclass
class Policy(ObservationGroupCfg):
    features=ObservationTermCfg(func=actor_obs)
    enable_corruption=True
    concatenate_terms=True
@configclass
class Critic(ObservationGroupCfg):
    features=ObservationTermCfg(func=critic_obs)
    concatenate_terms=True
@configclass
class Observations:
    policy=Policy()
    critic=Critic()
@configclass
class Terminations:
    failure=TerminationTermCfg(func=failed)
    end=TerminationTermCfg(func=ended,time_out=True)

def make_cfg(reference,num_envs,device,randomize=True):
    cfg=base.make_cfg(reference,num_envs,device,randomize)
    assets=Path(reference).parent/'assets'
    cfg.scene.env_spacing=4.
    cfg.scene.robot.spawn.asset_path=str(assets/'robot.urdf')
    cfg.scene.robot.spawn.merge_fixed_joints=False
    cfg.scene.robot.spawn.replace_cylinders_with_capsules=False
    cfg.scene.robot.spawn.collision_from_visuals=False
    cfg.scene.robot.spawn.collider_type='convex_hull'
    cfg.scene.robot.spawn.articulation_props.enabled_self_collisions=True
    cfg.scene.robot.spawn.rigid_props.max_depenetration_velocity=1.
    cfg.scene.robot.init_state.joint_pos['.*_hand_.*']=0.
    cfg.scene.robot.actuators['hands']=base.StrengthPDCfg(joint_names_expr=['.*_hand_.*'],effort_limit=.35,effort_limit_sim=.35,velocity_limit_sim=5.,stiffness=1.,damping=.03,armature=.001,min_delay=0,max_delay=2 if randomize else 0,strength_range=(.8,1.2) if randomize else (1.,1.))
    cupspawn=sim.UrdfFileCfg(asset_path=str(assets/'cup.urdf'),fix_base=False,merge_fixed_joints=False,replace_cylinders_with_capsules=False,
        usd_dir=tempfile.mkdtemp(prefix='furry-cup-usd-'),rigid_props=sim.RigidBodyPropertiesCfg(max_depenetration_velocity=1.,linear_damping=0.,angular_damping=0.),
        articulation_props=sim.ArticulationRootPropertiesCfg(articulation_enabled=False),joint_drive=sim.UrdfConverterCfg.JointDriveCfg(gains=sim.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0,damping=0)))
    cfg.scene.cup=RigidObjectCfg(prim_path='{ENV_REGEX_NS}/Cup',spawn=cupspawn,init_state=RigidObjectCfg.InitialStateCfg(pos=(.5,0,.6)))
    for k,s in enumerate(json.loads((assets/'statics.json').read_text())):
        setattr(cfg.scene,f'static_{k}',AssetBaseCfg(prim_path='{ENV_REGEX_NS}/Static'+str(k),spawn=sim.CuboidCfg(size=tuple(s['size']),collision_props=sim.CollisionPropertiesCfg(contact_offset=.001,rest_offset=0)),init_state=AssetBaseCfg.InitialStateCfg(pos=tuple(s['pos']),rot=tuple(s['quat']))))
    cfg.commands.motion.class_type=Motion;cfg.commands.motion.body_names=list(LINKS)
    cfg.observations=Observations();cfg.observations.policy.enable_corruption=randomize
    scale=dict(base.G1_ACTION_SCALE);scale['.*_hand_.*']=.25
    cfg.actions.joint_pos=JointPositionActionCfg(class_type=Action,asset_name='robot',joint_names=list(JOINTS),preserve_order=True,scale=scale,use_default_offset=True)
    cfg.rewards.motion_global_anchor_pos.weight=1.5;cfg.rewards.motion_global_anchor_ori.weight=1.5;cfg.rewards.motion_global_anchor_ori.params['std']=.2
    cfg.rewards.motion_joint_pos=RewardTermCfg(func=reward,weight=1.,params={'kind':'body_joint'})
    for key,weight in [('hand_joint',.3),('hand_pos',.3),('hand_ori',.3),('wrist_pos',1.2),('wrist_ori',1.),('object_pos',1.2),('object_ori',1.),('first',-.1),('waist_second',-.05),('legs_second',-.01)]:
        setattr(cfg.rewards,key,RewardTermCfg(func=reward,weight=weight,params={'kind':key}))
    cfg.rewards.action_rate_l2=None;cfg.rewards.waist_second_difference=None
    cfg.rewards.undesired_contacts.params['sensor_cfg']=SceneEntityCfg('contact_forces',body_names=['^(?!.*(?:ankle_roll|hand_|wrist_yaw)).*'])
    cfg.terminations=Terminations();cfg.episode_length_s=120.
    cfg.sim.dt=.005;cfg.decimation=4
    # Contact distance must remain below the 6 mm cup-wall thickness.
    cfg.scene.robot.spawn.collision_props=sim.CollisionPropertiesCfg(contact_offset=.001,rest_offset=0.)
    cfg.scene.cup.spawn.collision_props=sim.CollisionPropertiesCfg(contact_offset=.001,rest_offset=0.)
    if randomize:
        cfg.events.hand_mass=EventTermCfg(func=base.lab_mdp.randomize_rigid_body_mass,mode='startup',params={'asset_cfg':SceneEntityCfg('robot',body_names=['.*_hand_.*','.*_wrist_yaw_link']),'mass_distribution_params':(.8,1.25),'operation':'scale'})
        cfg.events.hand_com=EventTermCfg(func=base.mdp.randomize_rigid_body_com,mode='startup',params={'asset_cfg':SceneEntityCfg('robot',body_names=['.*_wrist_yaw_link']),'com_range':{k:(-.03,.03) for k in ('x','y','z')}})
        cfg.events.cup_material=EventTermCfg(func=base.lab_mdp.randomize_rigid_body_material,mode='startup',params={'asset_cfg':SceneEntityCfg('cup'),'static_friction_range':(.4,1.8),'dynamic_friction_range':(.35,1.5),'restitution_range':(0.,0.),'num_buckets':64,'make_consistent':True})
    return cfg
