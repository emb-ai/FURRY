"""Engine-independent named observation contract, shared with MuJoCo eval."""
import numpy as np
from scipy.spatial.transform import Rotation

def matrices(wxyz):
    q=np.asarray(wxyz)
    return Rotation.from_quat(q[..., [1,2,3,0]].reshape(-1,4)).as_matrix().reshape(q.shape[:-1]+(3,3))

def build(reference_q,reference_dq,reference_anchor_pos,reference_anchor_quat,
          body_pos,body_quat,base_linear_velocity,base_angular_velocity,
          q,dq,q_default,last_applied):
    """Single environment, 14 named links in spec.LINKS order; world velocities."""
    rot=matrices(body_quat);rt=rot[7].T
    anchor_pos=rt@(reference_anchor_pos-body_pos[7])
    anchor_ori=(rt@matrices(reference_anchor_quat))[:,:2].reshape(6)
    omega=rot[0].T@base_angular_velocity
    qrel=q-q_default
    common=np.r_[reference_q,reference_dq,anchor_pos,anchor_ori]
    actor=np.r_[common,omega,qrel,dq,last_applied].astype(np.float32)
    localpos=(body_pos-body_pos[7])@rot[7]
    localrot=(rt@rot)[...,:2].reshape(-1)
    critic=np.r_[common,localpos.reshape(-1),localrot,rot[0].T@base_linear_velocity,omega,qrel,dq,last_applied].astype(np.float32)
    if actor.shape!=(157,) or critic.shape!=(286,):raise ValueError('Bad observation dimensions')
    if not np.isfinite(actor).all() or not np.isfinite(critic).all():raise ValueError('Nonfinite observations')
    return actor,critic

def process_action(action,previous_target,q_default,scale,dt=.02):
    """Paper body target rate limits, preserving the APPLIED action for history."""
    action=np.asarray(action);target=q_default+scale*action
    limits=np.full(29,np.inf);limits[:15]=16.;limits[[2,8]]=20.
    target=np.clip(target,previous_target-limits*dt,previous_target+limits*dt)
    return target,(target-q_default)/scale
