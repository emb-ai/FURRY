"""Meaningful CPU checks against pinned source formulas and termination."""
import isaacgym
import torch
from types import SimpleNamespace as NS
from legged_gym.envs import G1MimicFuture
from locomotion_ablation import install

names=['left_rubber_hand','right_rubber_hand','left_ankle_roll_link','right_ankle_roll_link','left_knee_link','right_knee_link','left_elbow_link','right_elbow_link','head_mocap']

def make(cls, upper_error=False):
    e=object.__new__(cls);e.device='cpu';e.num_envs=4;e.num_dof=29
    e._ref_root_vel=torch.zeros(4,3);e._ref_root_vel[1,0]=.3
    e._ref_root_ang_vel=torch.zeros(4,3);e._ref_root_ang_vel[2:,2]=torch.tensor([.6,-.6])
    e._ref_dof_pos=torch.ones(4,29);e.dof_pos=torch.zeros(4,29)
    e._ref_dof_vel=torch.ones(4,29);e.dof_vel=torch.zeros(4,29);e._dof_err_w=torch.ones(29)
    e._key_body_ids=torch.arange(9);e.rigid_body_states=torch.zeros(4,9,13)
    e.root_states=torch.zeros(4,13);e.root_states[:,6]=1
    e._ref_root_rot=e.root_states[:,3:7].clone();e._ref_root_pos=torch.zeros(4,3)
    e._ref_body_pos=torch.zeros(4,9,3)
    e._ref_body_pos[:,0 if upper_error else 2,0]=1
    e.yaw=torch.zeros(4);e.roll=torch.zeros(4);e.pitch=torch.zeros(4);e.global_obs=False
    e.contact_forces=torch.zeros(4,9,3);e.contact_forces[:,2:4,2]=100
    e.feet_indices=torch.tensor([2,3]);e.last_contacts=torch.zeros(4,2,dtype=torch.bool)
    e.feet_air_time=torch.full((4,2),.2);e.dt=.02
    e.termination_contact_indices=torch.tensor([],dtype=torch.long)
    e.episode_length_buf=torch.ones(4,dtype=torch.long);e.max_episode_length=1000
    e.viewer=None;e._pose_termination=True;e._pose_termination_dist=.7;e._track_root=False
    e._motion_ids=torch.zeros(4,dtype=torch.long)
    e._motion_lib=NS(get_motion_length=lambda ids:torch.full((4,),60.))
    e.cfg=NS(motion=NS(key_bodies=names,use_adaptive_pose_termination=False),rewards=NS(root_height_diff_threshold=.5,termination_roll=4.,termination_pitch=4.,feet_air_time_target=.5))
    return e

class Identity(G1MimicFuture):pass
assert install(Identity,1.,False)=={}
assert Identity.check_termination is G1MimicFuture.check_termination
class Relax(G1MimicFuture):pass
originals=install(Relax,.25,False)
checked=[]
for method in ('_reward_tracking_joint_dof','_reward_tracking_joint_vel','_reward_tracking_keybody_pos','_reward_tracking_keybody_pos_global'):
    b=getattr(make(G1MimicFuture),method)();c=getattr(make(Relax),method)()
    assert torch.equal(b[:1],c[:1]),method+' idle changed'
    assert torch.all(c[1:]>b[1:]),method+' lower errors not relaxed'
    if 'keybody' in method:
        assert torch.equal(getattr(make(G1MimicFuture,True),method)(),getattr(make(Relax,True),method)()),method+' upper changed'
    checked.append(method)
b=make(G1MimicFuture);c=make(Relax);b.check_termination();c.check_termination()
assert b.reset_buf.all() and c.reset_buf.tolist()==[True,False,False,False]
c=make(Relax,True);c.check_termination();assert c.reset_buf.all()
c=make(Relax);c.roll[:]=5;c.check_termination();assert c.reset_buf.all()
class Yaw(G1MimicFuture):pass
install(Yaw,1.,True)
b=make(G1MimicFuture)._reward_feet_air_time();c=make(Yaw)._reward_feet_air_time()
assert torch.equal(b[:2],c[:2]) and torch.equal(b[2:],torch.zeros(2)) and (c[2:]<0).all()
class Hip(G1MimicFuture):pass
install(Hip,1.,False,hip_weight=3.)
assert Hip.check_termination is G1MimicFuture.check_termination and Hip._error_tracking_joint_dof is G1MimicFuture._error_tracking_joint_dof
h=make(Hip);w=h._leg_joint_reward_weights()
assert w[:,[1,2,7,8]].eq(3).all() and w[:,[0,3,4,5,6,9,10,11]].eq(1).all() and w[:,12:].eq(1).all()
assert torch.all(h._reward_tracking_joint_dof()<make(G1MimicFuture)._reward_tracking_joint_dof()),'hip roll/yaw errors not up-weighted'
class RelaxHip(G1MimicFuture):pass
install(RelaxHip,.25,False,hip_weight=2.)
w=make(RelaxHip)._leg_joint_reward_weights();assert w[0,1]==2 and w[1,1]==.5 and w[1,0]==.25
print({'passed':True,'source_reward_methods':checked,'checks':['coefficient1 source identity','idle unchanged','upper-body unchanged','lower pose termination relaxed only during movement','upper/safety termination preserved','yaw both signs airtime active','hip roll/yaw reward weight only']})
