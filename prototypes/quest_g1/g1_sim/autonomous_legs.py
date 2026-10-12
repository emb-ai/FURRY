"""Desktop MuJoCo backends for HOMIE v2 and SONIC three-point teleoperation.

Contracts ported from NVIDIA GR00T-WholeBodyControl b042411f (Apache-2.0).
SONIC uses the released neural kinematic planner, never recorded leg targets.
Weights remain external. This module does not deploy anything to Quest.
"""
from collections import deque
from pathlib import Path
import json
import hashlib
import time
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
import onnxruntime as ort
from scipy.spatial.transform import Rotation, Slerp
from .controller import JOINTS, euler


def rot(q):
    return Rotation.from_quat(np.asarray(q)[[1,2,3,0]])

def quat(r):
    return r.as_quat()[[3,0,1,2]]

def session(path):
    opt=ort.SessionOptions();opt.intra_op_num_threads=1;opt.inter_op_num_threads=1
    return ort.InferenceSession(str(path),sess_options=opt,providers=['CPUExecutionProvider'])

def infer(s,x):
    y=s.run(None,{s.get_inputs()[0].name:np.asarray(x,np.float32).reshape(1,-1)})[0].ravel()
    if not np.isfinite(y).all():raise RuntimeError('Nonfinite policy output')
    return y

def motor_profile(model,path):
    """Apply the same named motor fields as native bodymotors::Apply."""
    for line in Path(path).read_text().splitlines()[1:]:
        name,*v=line.split();a,d,f,lo,hi,flo,fhi=map(float,v)
        j=model.joint(name);i=j.dofadr[0];aid=model.actuator(name).id
        model.dof_armature[i]=a;model.dof_damping[i]=d;model.dof_frictionloss[i]=f
        model.jnt_range[j.id]=[lo,hi];model.jnt_limited[j.id]=True
        model.jnt_actfrcrange[j.id]=[flo,fhi];model.jnt_actfrclimited[j.id]=True
        model.actuator_ctrlrange[aid]=[flo,fhi];model.actuator_ctrllimited[aid]=True
        model.actuator_forcerange[aid]=0;model.actuator_forcelimited[aid]=False

class PDBackend:
    def __init__(self,model):
        self.model=model
        self.qa=np.array([model.joint(n).qposadr[0] for n in JOINTS])
        self.va=np.array([model.joint(n).dofadr[0] for n in JOINTS])
        self.ai=np.array([model.actuator(n).id for n in JOINTS])
        self.torso=model.body('torso_link').id
        self.aux=mujoco.MjData(model)
        self.inference_ms=[]
    def apply(self,data):
        data.ctrl[self.ai]=(self.target-data.qpos[self.qa])*self.kp-data.qvel[self.va]*self.kd
        mujoco.mj_step(self.model,data)
    def initialize(self,data):
        mujoco.mj_setConst(self.model,data)
        mujoco.mj_resetData(self.model,data)
        data.qpos[:7]=[0,0,.793,1,0,0,0];data.qpos[self.qa]=self.home
        data.qpos[self.qa[[16,23]]]=[.2,-.2];data.qpos[self.qa[[18,25]]]=1.2
        self.target=data.qpos[self.qa].copy();mujoco.mj_forward(self.model,data)

class Homie(PDBackend):
    home=np.array([-.1,0,0,.3,-.2,0]*2+[0]*17)
    kp=np.array([150,150,150,200,40,40]*2+[250]*3+[100,100,40,40,20,20,20]*2)
    kd=np.array([2,2,2,4,2,2]*2+[5]*3+[5,5,2,2,2,2,2]*2)
    def __init__(self,model,assets):
        super().__init__(model);motor_profile(model,Path(assets)/'body_motor_profile.txt')
        self.balance=session(Path(assets)/'policy-homie-v2-balance.onnx')
        self.walk=session(Path(assets)/'policy-homie-v2-walk.onnx')
        self.history=deque([np.zeros(86,np.float32) for _ in range(6)],maxlen=6)
        self.previous=np.zeros(15,np.float32)
    def observation(self,data,cmd):
        # The same torso FK as native MakeDecoupledCommand; no reference legs.
        mujoco.mj_resetData(self.model,self.aux)
        # scipy lowercase applies extrinsic: use explicit pitch * roll, matching native.
        self.aux.qpos[3:7]=quat(Rotation.from_rotvec([0,cmd[4],0])*Rotation.from_rotvec([cmd[3],0,0]))
        self.aux.qpos[self.qa[12:15]]=cmd[18:21];mujoco.mj_kinematics(self.model,self.aux)
        torso=Rotation.from_matrix(self.aux.xmat[self.torso].reshape(3,3)).as_euler('xyz')
        return np.r_[cmd[[0,1,5]]*[2,2,.5],cmd[2],torso,data.qvel[3:6]*.5,
                     rot(data.qpos[3:7]).inv().apply([0,0,-1]),data.qpos[self.qa]-self.home,
                     data.qvel[self.va]*.05,self.previous].astype(np.float32)
    def update(self,data,cmd,reference=None):
        self.history.append(self.observation(data,cmd));t=time.perf_counter()
        s=self.walk if np.linalg.norm(cmd[[0,1,5]])>=.05 else self.balance
        self.previous=infer(s,np.concatenate(self.history));self.inference_ms.append((time.perf_counter()-t)*1000)
        self.target=np.r_[self.home[:15]+.25*self.previous,cmd[21:35]]

class Sonic(PDBackend):
    # Exact concatenation order in the matching observation_config.yaml.
    ENCODER_SIZES=[4,290,290,10,1,6,60,120,120,9,12,720,60,60]
    def __init__(self,model,assets,models,constants,seed=11):
        super().__init__(model)
        c=json.loads(Path(constants).read_text())
        self.home=np.array(c['home']);self.kp=np.array(c['kp']);self.kd=np.array(c['kd'])
        self.scale=np.array(c['scale']);self.to_mj=np.array(c['to_mujoco']);self.to_il=np.array(c['to_isaac'])
        config=Path(models)/'observation_config.yaml'
        if hashlib.sha256(config.read_bytes()).hexdigest()!='466d05947c78af6c76388adfb86e3a2a77b2a1d921a64883ed3d085ebf58de1b':
            raise ValueError('SONIC adapter requires the pinned default observation config')
        self.enc=session(Path(models)/'model_encoder.onnx');self.dec=session(Path(models)/'model_decoder.onnx')
        self.planner=session(Path(models)/'planner_sonic.onnx')
        self.previous=np.zeros(29);self.history=deque(maxlen=10)
        # Upstream pads missing state with q=dq=action=omega=0, identity quat.
        for _ in range(10):self.history.append((np.zeros(3),np.zeros(29),np.zeros(29),np.zeros(29),np.array([0,0,-1])))
        self.plan=None;self.plan_start=0.;self.seed=seed;self.planner_calls=0;self.planner_ms=[]
        self.vr_bodies=[model.body(n).id for n in ['left_wrist_yaw_link','right_wrist_yaw_link','torso_link']]
        self.offsets=np.array([[.18,-.025,0],[.18,.025,0],[0,0,.35]])
        # Named body motor profile from the author's SONIC deployment XML.
        motor_profile(model,Path(constants).with_name('sonic_motor_profile.txt'))
    def _at(self,times):
        pos=np.clip((np.asarray(times)-self.plan_start)*30,0,len(self.plan)-1)
        lo=np.floor(pos).astype(int);hi=np.minimum(lo+1,len(self.plan)-1);w=pos-lo
        q=self.plan[lo]*(1-w[:,None])+self.plan[hi]*w[:,None]
        # quaternion interpolation must respect sign and unit length.
        for i,(a,b,t) in enumerate(zip(lo,hi,w)):
            if a==b:q[i,3:7]=self.plan[a,3:7]
            else:q[i,3:7]=quat(Slerp([0,1],Rotation.from_quat(self.plan[[a,b],3:7][:,[1,2,3,0]]))([t])[0])
        return q
    def replan(self,data,cmd):
        now=data.time
        if self.plan is None:
            q=np.r_[data.qpos[:7],data.qpos[self.qa]]
            context=np.tile(q,(4,1))
        else:
            # Four 30 Hz context frames continuing the generated trajectory.
            context=self._at(now+np.arange(4)/30)
        world=rot(data.qpos[3:7]).apply([cmd[0],cmd[1],0]);speed=np.linalg.norm(world[:2])
        heading=rot(context[-1,3:7]).as_euler('xyz')[2]+float(cmd[5])*.3
        movement=world/max(speed,1e-8);movement[2]=0
        mode=0 if speed<.05 else (1 if speed<.5 else 2)
        feed={'context_mujoco_qpos':context[None].astype(np.float32),
              'target_vel':np.array([max(speed,.01)],np.float32),'mode':np.array([mode],np.int64),
              'movement_direction':movement[None].astype(np.float32),
              'facing_direction':np.array([[np.cos(heading),np.sin(heading),0]],np.float32),
              'random_seed':np.array([self.seed],np.int64),'has_specific_target':np.zeros((1,1),np.int64),
              'specific_target_positions':np.zeros((1,4,3),np.float32),'specific_target_headings':np.zeros((1,4),np.float32),
              'allowed_pred_num_tokens':np.array([[1]*6+[0]*5],np.int64),'height':np.array([-1],np.float32)}
        t=time.perf_counter();q,n=self.planner.run(None,feed);self.planner_ms.append((time.perf_counter()-t)*1000)
        n=int(n.item());assert 4<=n<=64
        self.plan=q[0,:n].astype(float);self.plan_start=now;self.planner_calls+=1
        assert np.isfinite(self.plan).all()
    def encoder_observation(self,data,reference):
        future=self._at(data.time+np.arange(10)*.1)
        nextq=self._at(data.time+np.arange(10)*.1+.02)
        fields=[np.zeros(n,np.float32) for n in self.ENCODER_SIZES]
        fields[0][0]=1 # scalar mode id, NOT one-hot
        fields[7]=future[:,7:19].ravel();fields[8]=((nextq[:,7:19]-future[:,7:19])/.02).ravel()
        fields[5]=(rot(data.qpos[3:7]).inv()*rot(future[0,3:7])).as_matrix()[:,:2].ravel()
        # Robot-scaled head/wrists from the same reference FK used for every backend.
        self.aux.qpos[:7]=reference[:7];self.aux.qpos[self.qa]=reference[7:36]
        mujoco.mj_kinematics(self.model,self.aux)
        rr=rot(reference[3:7]);positions=[];orientations=[]
        for bid,offset in zip(self.vr_bodies,self.offsets):
            r=rot(self.aux.xquat[bid]);p=self.aux.xpos[bid]+r.apply(offset)
            positions.extend(rr.inv().apply(p-reference[:3]));orientations.extend(quat(rr.inv()*r))
        fields[9]=positions;fields[10]=orientations
        obs=np.concatenate(fields).astype(np.float32);assert obs.shape==(1762,)
        return obs
    def update(self,data,cmd,reference):
        if self.plan is None or data.time-self.plan_start>=.2-1e-8:self.replan(data,cmd)
        t=time.perf_counter();token=infer(self.enc,self.encoder_observation(data,reference))
        self.history.append((data.qvel[3:6].copy(),(data.qpos[self.qa]-self.home)[self.to_il],
                             data.qvel[self.va][self.to_il].copy(),self.previous.copy(),rot(data.qpos[3:7]).inv().apply([0,0,-1])))
        obs=np.r_[token,*[np.concatenate([h[k] for h in self.history]) for k in range(5)]]
        assert obs.shape==(994,)
        self.previous=infer(self.dec,obs);self.target=self.home+self.scale*self.previous[self.to_mj]
        self.inference_ms.append((time.perf_counter()-t)*1000)
