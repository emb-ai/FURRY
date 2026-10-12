"""Source-PPO reward adapter; no actor inputs, optimizer, DR, or runtime edits.

Desired phase is inferred causally from reference sole geometry only. Achieved
contact never changes it. GPU net body force is only a floor-force proxy, gated
by actual physical sole proximity in the enforced plane/single-actor setting.
Sole-centre finite differences plus omega x (-radius*world_z) include rigid
rotation without guessing whether Isaac Gym linear velocities refer to COM.
"""
from dataclasses import asdict, dataclass
import math
from pathlib import Path
import xml.etree.ElementTree as ET
import torch


@dataclass(frozen=True)
class PhaseConfig:
    stance_enter_height: float = .012
    stance_exit_height: float = .025
    swing_enter_height: float = .035
    swing_exit_height: float = .025
    stance_enter_speed: float = .10
    stance_exit_speed: float = .20
    stance_enter_vertical_speed: float = .12
    stance_exit_vertical_speed: float = .25
    stance_enter_yaw_rate: float = .52
    stance_exit_yaw_rate: float = .90
    stance_dwell_s: float = .08
    swing_dwell_s: float = .04
    max_gap_s: float = .06
    velocity_tau_s: float = .025
    support_band_m: float = .008
    max_reference_penetration_m: float = .01


def qmul(a,b):
    av,bv=a[...,:3],b[...,:3]
    return torch.cat((a[...,3:]*bv+b[...,3:]*av+torch.cross(av,bv,dim=-1),
                      a[...,3:]*b[...,3:]-(av*bv).sum(-1,keepdim=True)),-1)


def qrot(q,v):
    # Retain the scalar quaternion component while broadcasting XYZ vectors.
    quat,vec=q,v
    xyz,vec=torch.broadcast_tensors(quat[...,:3],vec)
    t=2*torch.cross(xyz,vec,dim=-1)
    return vec+quat[...,3:]*t+torch.cross(xyz,t,dim=-1)


def yaw(q):
    x,y,z,w=q.unbind(-1)
    return torch.atan2(2*(w*z+x*y),w*w+x*x-y*y-z*z)


def rpy_quat(rpy):
    r,p,y=[v/2 for v in rpy];cr,sr,cp,sp,cy,sy=math.cos(r),math.sin(r),math.cos(p),math.sin(p),math.cos(y),math.sin(y)
    return [sr*cp*cy-cr*sp*sy,cr*sp*cy+sr*cp*sy,cr*cp*sy-sr*sp*cy,cr*cp*cy+sr*sp*sy]


class SoleKinematics:
    """Two leg-chain FK from the actual sphere-sole URDF, XYZW quaternions."""
    def __init__(self,urdf,dof_names,device='cpu',dtype=torch.float32):
        tree=ET.parse(urdf).getroot();joints={j.find('child').get('link'):j for j in tree.findall('joint')}
        self.chains=[];points=[];radii=[];self.foot_names=['left_ankle_roll_link','right_ankle_roll_link']
        self.tensor=lambda x:torch.tensor(x,device=device,dtype=dtype)
        for foot in self.foot_names:
            chain=[];link=foot
            while link!='pelvis':
                if link not in joints:raise ValueError('Foot chain does not lead to pelvis')
                j=joints[link];origin=j.find('origin');xyz=[0.,0.,0.];rpy=xyz
                if origin is not None:
                    xyz=list(map(float,origin.get('xyz','0 0 0').split()));rpy=list(map(float,origin.get('rpy','0 0 0').split()))
                idx=None;axis=[0.,0.,0.]
                if j.get('type')!='fixed':
                    if j.get('type') not in ['revolute','continuous']:raise ValueError('Unsupported leg joint')
                    idx=list(dof_names).index(j.get('name'));axis=list(map(float,j.find('axis').get('xyz').split()))
                chain.append((self.tensor(xyz),self.tensor(rpy_quat(rpy)),idx,self.tensor(axis)))
                link=j.find('parent').get('link')
            self.chains.append(chain[::-1]);p=[];r=[]
            body=tree.find("link[@name='%s']"%foot)
            for collision in body.findall('collision'):
                sphere=collision.find('geometry/sphere')
                if sphere is None:raise ValueError('Phase training requires sphere-only foot collisions')
                origin=collision.find('origin');p.append(list(map(float,origin.get('xyz','0 0 0').split())) if origin is not None else [0,0,0]);r.append(float(sphere.get('radius')))
            if not p:raise ValueError('Missing physical sole spheres')
            points.append(p);radii.append(r)
        if len(points[0])!=len(points[1]):raise ValueError('Unequal foot sphere count')
        self.points,self.radii=self.tensor(points),self.tensor(radii)

    def world_soles(self,pos,quat):
        quat=quat/torch.linalg.vector_norm(quat,dim=-1,keepdim=True).clamp(min=1e-8)
        return pos[...,None,:]+qrot(quat[...,None,:],self.points[None])

    def reference(self,root_pos,root_quat,dof):
        poses=[];rots=[];root_quat=root_quat/torch.linalg.vector_norm(root_quat,dim=-1,keepdim=True).clamp(min=1e-8)
        for chain in self.chains:
            p,q=root_pos,root_quat
            for origin,fixed,index,axis in chain:
                p=p+qrot(q,origin);q=qmul(q,fixed.expand_as(q))
                if index is not None:
                    half=.5*dof[:,index];rotation=torch.cat((axis[None]*torch.sin(half)[:,None],torch.cos(half)[:,None]),-1)
                    q=qmul(q,rotation)
            poses.append(p);rots.append(q)
        pos,quat=torch.stack(poses,1),torch.stack(rots,1)
        return self.world_soles(pos,quat),yaw(quat)


class ReferencePhase:
    """Vectorized equivalent of g1_sim.contact_phase causal labeler."""
    def __init__(self,count,points,device,config=PhaseConfig(),dtype=torch.float32):
        self.c=config;self.state=torch.full((count,2),-1,device=device,dtype=torch.long)
        self.candidate=self.state.clone();self.duration=torch.zeros((count,2),device=device,dtype=dtype)
        self.previous=torch.zeros((count,2,points,3),device=device,dtype=dtype)
        self.previous_h=torch.zeros((count,2),device=device,dtype=dtype);self.previous_yaw=self.previous_h.clone()
        self.previous_time=torch.full((count,),float('nan'),device=device,dtype=dtype)
        self.previous_good=torch.zeros((count,2),device=device,dtype=torch.bool);self.has_velocity=self.previous_good.clone()
        self.smooth=torch.zeros((count,2,3),device=device,dtype=dtype)
        self.confidence=self.previous_h.clone();self.clearance=self.previous_h.clone()

    def reset(self,ids):
        self.state[ids]=-1;self.candidate[ids]=-1;self.duration[ids]=0
        self.previous_good[ids]=False;self.has_velocity[ids]=False;self.previous_time[ids]=float('nan');self.confidence[ids]=0

    def update(self,time,points,radii,foot_yaw,valid=None):
        c=self.c;dt=time-self.previous_time;continuous=torch.isfinite(dt)&(dt>0)&(dt<=c.max_gap_s)
        safe_dt=torch.where(continuous,dt,torch.ones_like(dt))[:,None]
        bottom=points[...,2]-radii;h=bottom.min(-1).values
        good=torch.isfinite(points).all(-1).all(-1)&torch.isfinite(foot_yaw)&(h>=-c.max_reference_penetration_m)
        if valid is not None:good &= valid
        can=good&self.previous_good&continuous[:,None]
        self.state=torch.where(can,self.state,torch.full_like(self.state,-1))
        self.candidate=torch.where(can,self.candidate,torch.full_like(self.candidate,-1));self.duration=torch.where(can,self.duration,torch.zeros_like(self.duration))
        velocity=(points-self.previous)/safe_dt[...,None,None]
        low=bottom<=h[...,None]+c.support_band_m
        horizontal=torch.sqrt((velocity[...,:2].square().sum(-1)*low).sum(-1)/low.sum(-1).clamp(min=1))
        vertical=(h-self.previous_h)/safe_dt;angle=foot_yaw-self.previous_yaw
        wy=torch.atan2(torch.sin(angle),torch.cos(angle))/safe_dt
        values=torch.stack((horizontal,vertical,wy),-1);alpha=1-torch.exp(-safe_dt/c.velocity_tau_s)
        smooth=torch.where(self.has_velocity[...,None],self.smooth+alpha[...,None]*(values-self.smooth),values)
        smooth=torch.nan_to_num(smooth);speed,vz,wy=smooth.unbind(-1)
        enter=(h<=c.stance_enter_height)&(speed<=c.stance_enter_speed)&(vz.abs()<=c.stance_enter_vertical_speed)&(wy.abs()<=c.stance_enter_yaw_rate)
        hold=(self.state==1)&(h<=c.stance_exit_height)&(speed<=c.stance_exit_speed)&(vz.abs()<=c.stance_exit_vertical_speed)&(wy.abs()<=c.stance_exit_yaw_rate)
        swing=(h>=c.swing_enter_height)|((self.state==0)&(h>=c.swing_exit_height))
        proposed=torch.where(enter|hold,torch.ones_like(self.state),torch.where(swing,torch.zeros_like(self.state),torch.full_like(self.state,-1)))
        proposed=torch.where(can,proposed,torch.full_like(proposed,-1))
        self.duration=torch.where(proposed==self.candidate,self.duration,torch.zeros_like(self.duration))+safe_dt
        self.duration=torch.where(proposed>=0,self.duration,torch.zeros_like(self.duration))
        dwell=torch.where(proposed==1,c.stance_dwell_s,c.swing_dwell_s)
        ready=(proposed==self.state)|(self.duration+1e-7>=dwell)
        self.state=torch.where(ready,proposed,torch.full_like(proposed,-1));self.candidate=proposed
        stance_conf=torch.exp(-h.clamp(min=0)/c.stance_exit_height-speed/c.stance_exit_speed)
        swing_conf=((h-c.stance_enter_height)/(c.swing_enter_height-c.stance_enter_height)).clamp(0,1)
        self.confidence=torch.where(self.state==1,stance_conf,torch.where(self.state==0,swing_conf,torch.zeros_like(h)))
        self.clearance=torch.nan_to_num(h);self.smooth=smooth;self.has_velocity=can
        self.previous=points.clone();self.previous_h=h;self.previous_yaw=foot_yaw.clone();self.previous_good=good;self.previous_time=time.clone()
        return self.state,self.confidence


def contact_terms(state,confidence,ref_clearance,points,previous_points,radii,angular_velocity,force_z,dt,valid,body_weight):
    """Nonnegative penalties; no actual contact argument can alter phase.

    FD of material sphere centres is a 20ms averaged velocity. omega cross the
    bottom-radius arm accounts for the material point instantaneously at floor.
    It is not an exact per-contact solver-force measurement.
    """
    surface_arm=torch.zeros_like(points);surface_arm[...,2]=-radii
    velocity=(points-previous_points)/dt+torch.cross(angular_velocity[...,None,:].expand_as(points),surface_arm,dim=-1)
    bottom=points[...,2]-radii;clearance=bottom.min(-1).values
    near=(bottom<=.008)&(bottom<=clearance[...,None]+.008)
    speed2=velocity[...,:2].square().sum(-1)
    point_speed=torch.sqrt((speed2*near).sum(-1)/near.sum(-1).clamp(min=1))
    actual_floor=(clearance<=.008)&(force_z>5)&valid[:,None]
    load=(force_z.clamp(min=0)/(body_weight/2)).clamp(max=2)
    swing=(state==0)&valid[:,None]
    # Keep source anti-slip supervision on every achieved floor contact, also
    # UNKNOWN/SWING reference phases. Poor reference quality cannot remove it.
    slip=(torch.sqrt(point_speed.clamp(min=0))*load*actual_floor).sum(-1)
    premature=(load*swing*actual_floor*confidence).sum(-1)
    # Track only missing clearance; never reward feet higher than the reference.
    # Capping desired height prevents high-kick reference magnitudes dominating.
    desired=ref_clearance.clamp(min=0,max=.08)
    deficit=((desired-clearance).clamp(min=0)/.04).clamp(max=2)
    clearance_penalty=(deficit.square()*swing*confidence).sum(-1)
    return slip,premature,clearance_penalty


class PhaseContactState:
    feature_dim=6
    def __init__(self,env,urdf):
        self.env=env;self.kinematics=SoleKinematics(urdf,env.gym.get_actor_dof_names(env.envs[0],env.actor_handles[0]),env.device)
        self.phase=ReferencePhase(env.num_envs,self.kinematics.points.shape[1],env.device)
        self.previous_points=torch.zeros((env.num_envs,2,self.kinematics.points.shape[1],3),device=env.device)
        self.valid=torch.zeros(env.num_envs,device=env.device,dtype=torch.bool)
        self.features=torch.zeros((env.num_envs,self.feature_dim),device=env.device)
        self.terms=[torch.zeros(env.num_envs,device=env.device) for _ in range(3)]
        self.counts=torch.zeros(4,device=env.device,dtype=torch.long)
        self.body_weight=torch.tensor([sum(p.mass for p in env.gym.get_actor_rigid_body_properties(e,a))*9.81
                                        for e,a in zip(env.envs,env.actor_handles)],device=env.device)[:,None]

    def reset(self,ids):
        self.phase.reset(ids);self.valid[ids]=False;self.features[ids]=0

    def advance(self):
        env=self.env;k=self.kinematics
        points,yaws=k.reference(env._ref_root_pos,env._ref_root_rot,env._ref_dof_pos)
        states,confidence=self.phase.update(env._get_motion_times(),points,k.radii,yaws)
        rb=env.rigid_body_states[:,env.feet_indices]
        actual=k.world_soles(rb[...,:3],rb[...,3:7])
        self.terms=contact_terms(states,confidence,self.phase.clearance,actual,self.previous_points,k.radii,
                                 rb[...,10:13],env.contact_forces[:,env.feet_indices,2],env.dt,self.valid,self.body_weight)
        self.previous_points=actual.clone();self.valid[:]=True
        self.features=torch.cat((states.float(),confidence,(self.phase.clearance/.08).clamp(-1,5)),dim=-1)
        self.counts+=torch.stack(((states==1).sum(),(states==0).sum(),(states<0).sum(),torch.tensor(states.numel(),device=env.device)))


def install(env,urdf,treatment=False):
    """Same phase observations for both matched control/treatment critics."""
    if abs(env.dt-.02)>1e-8 or env.num_obs!=1432:raise ValueError('Expected existing50Hz/1432 actor contract')
    if env.cfg.terrain.mesh_type!='plane':raise ValueError('Floor proxy requires flat plane training')
    if any(env.gym.get_actor_count(e)!=1 for e in env.envs):raise ValueError('Task objects are not valid floor contacts: expected one robot actor per env')
    if list(env.feet_indices.shape)!=[2]:raise ValueError('Expected two feet')
    body_names=env.gym.get_actor_rigid_body_names(env.envs[0],env.actor_handles[0])
    for j,name in zip(env.feet_indices.tolist(),['left_ankle_roll_link','right_ankle_roll_link']):
        if body_names[j]!=name:raise ValueError('Foot body order mismatch')
    state=PhaseContactState(env,urdf)
    original_reward,original_reset=env.compute_reward,env.reset_idx
    source_slip_scale=float(env.reward_scales['feet_slip'])
    if source_slip_scale>=0:raise ValueError('Expected source slip penalty')
    if treatment:
        index=env.reward_names.index('feet_slip')
        # Source scale/sign retained; replace ankle-centre slip with actual
        # load-weighted sole candidate slip. No change to source imitation terms.
        env.reward_functions[index]=lambda:state.terms[0]
        for name,term in [('phase_early_floor',1),('phase_swing_clearance',2)]:
            env.reward_names.append(name);env.reward_functions.append(lambda term=term:state.terms[term])
            # Reuse the existing source slip scale (already dt-scaled once).
            env.reward_scales[name]=source_slip_scale;env.episode_sums[name]=torch.zeros(env.num_envs,device=env.device)
    def compute_reward():
        state.advance();original_reward()
    def reset(ids,*args,**kwargs):
        original_reset(ids,*args,**kwargs);state.reset(ids)
    env.compute_reward,env.reset_idx=compute_reward,reset
    return state


def training_reference_coverage(env,urdf):
    """Four causal one-second windows per TRAIN clip; no physics/optimizer/RNG.

    This is coverage of geometric desired-phase labels, not achieved contacts.
    Public and Quest groups are reported separately to expose empty Quest signal.
    First 80ms per window intentionally retain causal dwell/reset uncertainty.
    """
    files=list(env._motion_lib._motion_files);n=len(files);device=env.device
    ids=torch.arange(n,device=device).repeat_interleave(4)
    lengths=env._motion_lib.get_motion_length(ids)
    starts=(lengths-1.).clamp(min=0)*torch.tensor([.1,.35,.6,.85],device=device).repeat(n)
    kin=SoleKinematics(urdf,env.gym.get_actor_dof_names(env.envs[0],env.actor_handles[0]),device);phase=ReferencePhase(len(ids),kin.points.shape[1],device)
    counts=torch.zeros((len(ids),4),device=device,dtype=torch.long)
    with torch.no_grad():
        for i in range(50):
            times=starts+i*env.dt;valid=(times<=lengths)[:,None].expand(-1,2)
            root,quat,_,_,dof,*_=env._motion_lib.calc_motion_frame(ids,times.minimum(lengths))
            root=root.clone();root[:,2]+=env.cfg.motion.height_offset
            points,yaws=kin.reference(root,quat,dof)
            state,_=phase.update(times,points,kin.radii,yaws,valid)
            counts[:,0]+=((state==1)&valid).sum(-1);counts[:,1]+=((state==0)&valid).sum(-1)
            counts[:,2]+=((state<0)&valid).sum(-1);counts[:,3]+=valid.sum(-1)
    counts=counts.reshape(n,4,4).sum(1).cpu().tolist()
    rows=[];groups={}
    for path,count in zip(files,counts):
        normalized='/'+str(path).replace('\\','/').lstrip('./')
        group='quest' if '/quest/' in normalized else 'public'
        row=dict(file=str(path),group=group,**dict(zip(('stance','swing','unknown','total'),count)))
        row['known_fraction']=(count[0]+count[1])/max(count[3],1);rows.append(row)
        groupcounts=groups.setdefault(group,dict(clips=0,stance=0,swing=0,unknown=0,total=0))
        groupcounts['clips']+=1
        for key in ('stance','swing','unknown','total'):groupcounts[key]+=row[key]
    for value in groups.values():
        value['known_fraction']=(value['stance']+value['swing'])/max(value['total'],1)
        value['swing_fraction']=value['swing']/max(value['total'],1)
    return dict(method='4 x 1s causal windows at10/35/60/85% of allowable clip start range; reference only; no physics/RNG/optimizer',dt=env.dt,phase_config=asdict(PhaseConfig()),groups=groups,clips=rows,all_train_clips=len(files),actor_inputs=env.num_obs)
