"""100 Hz MuJoCo tracking environment, matching Quest observations and PD.

Reward equations follow pinned TWIST2 humanoid_mimic.py where comparable;
contact forces are MuJoCo world-frame forces. Not Isaac Gym parity.
"""
import sys,json
from pathlib import Path
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from rewards import RewardConfig, sliding_rate, global_tracking_reward, discrete_rate
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from g1_sim.controller import DEFAULT,KP,KD,JOINTS,euler
DT=.01
FUTURE_SECONDS=np.array([.02]+[i/10 for i in range(1,20)])

def future_reference_features(motion, frame, root_position, root_quaternion):
 """Critic only: 20 reference poses up to 1.9 s, with explicit end padding.

 Commands retain the actor's convention; relative translation/orientation expose
 future tracking errors. Padding repeats the final pose with zero velocities
 and a validity bit, so a short clip cannot masquerade as a longer trajectory.
 """
 indices=frame+np.rint(FUTURE_SECONDS/DT).astype(int);valid=indices<len(motion['q'])
 indices=np.minimum(indices,len(motion['q'])-1);q=motion['q'][indices]
 commands=motion['cmd'][indices].copy();commands[~valid,0:2]=0;commands[~valid,5]=0
 current=Rotation.from_quat(np.asarray(root_quaternion)[[1,2,3,0]])
 delta_position=current.inv().apply(q[:,:3]-root_position)
 delta_rotation=(current.inv()*Rotation.from_quat(q[:,[4,5,6,3]])).as_rotvec()
 return np.c_[commands,delta_position,delta_rotation,valid].astype(np.float32).ravel()

KEYS=['left_wrist_yaw_link','right_wrist_yaw_link','left_ankle_roll_link','right_ankle_roll_link','left_knee_link','right_knee_link','left_elbow_link','right_elbow_link','torso_link']
class Motions:
 def __init__(self,path,split):
  self.root=Path(path);self.manifest=json.loads((self.root/'manifest.json').read_text());self.items=[]
  for c in self.manifest['clips']:
   if c['split']!=split and not (split=='train' and c['split']=='replay'):continue
   d=np.load(self.root/c['file']);q=d['qpos'].copy();fps=float(d['fps']);assert fps==100 and len(q)>100
   rot=Rotation.from_quat(q[:,[4,5,6,3]]);vel=np.gradient(q[:,:3],DT,axis=0);ang=np.zeros((len(q),3));ang[1:]=(rot[:-1].inv()*rot[1:]).as_rotvec()/DT;ang[0]=ang[1]
   cmd=np.c_[rot.inv().apply(vel)[:,:2],q[:,2],rot.as_euler('xyz')[:,:2],ang[:,2],q[:,7:]].astype(np.float32)
   self.items.append({'q':q,'v':np.c_[vel,ang,np.gradient(q[:,7:],DT,axis=0)],'cmd':cmd,'meta':c})
  if not self.items:raise ValueError(f'No motions in {split}')
  self.weights=np.array([len(c['q']) for c in self.items],float);self.weights/=self.weights.sum()
  replay=np.array([c['meta']['split']=='replay' for c in self.items])
  if replay.any() and (~replay).any():
   fraction=self.manifest.get('replay_sampling_fraction',.2);self.weights[replay]*=fraction/self.weights[replay].sum();self.weights[~replay]*=(1-fraction)/self.weights[~replay].sum()
class Env:
 def __init__(self,assets,motions,seed=0,reward_config=None):
  self.reward_config=reward_config or RewardConfig()
  self.model=mujoco.MjModel.from_xml_path(str(Path(assets)/'scene.xml'));self.data=mujoco.MjData(self.model);self.ref=mujoco.MjData(self.model);self.motions=motions;self.rng=np.random.default_rng(seed)
  self.qa=np.array([self.model.joint(n).qposadr[0] for n in JOINTS]);self.va=np.array([self.model.joint(n).dofadr[0] for n in JOINTS]);self.aids=np.array([self.model.actuator(n).id for n in JOINTS]);self.key=np.array([self.model.body(n).id for n in KEYS]);self.feet=np.array([self.model.body(s+'_ankle_roll_link').id for s in ['left','right']]);self.floor=self.model.geom('floor').id
  assert abs(self.model.opt.timestep-.001)<1e-10
  self.contact_data=mujoco.MjData(self.model);self.contact_state_spec=mujoco.mjtState.mjSTATE_INTEGRATION;self.contact_state=np.zeros(mujoco.mj_stateSize(self.model,self.contact_state_spec));self.contact_jac=np.zeros((3,self.model.nv))
  self.ranges=np.array([self.model.joint(n).range for n in JOINTS]);self.torque_limit=KP.copy();self.dt=DT;self.max_steps=2000
 def reset(self,clip=None,start=None,randomize=False):
  self.clip=int(self.rng.choice(len(self.motions.items),p=self.motions.weights)) if clip is None else clip;self.motion=self.motions.items[self.clip];n=len(self.motion['q'])
  self.k=int(self.rng.integers(max(1,n-200))) if start is None else start;self.first=self.k
  mujoco.mj_resetData(self.model,self.data);q=self.motion['q'][self.k];self.data.qpos[:7]=q[:7];self.data.qpos[self.qa]=q[7:];self.data.qvel[:6]=self.motion['v'][self.k,:6];self.data.qvel[self.va]=self.motion['v'][self.k,6:]
  if randomize:self.data.qpos[self.qa]+=self.rng.normal(0,.005,29)
  mujoco.mj_forward(self.model,self.data);self.hist=np.zeros((10,127),np.float32);self.action=np.zeros(29,np.float32);self.air=np.zeros(2);self.lastcontact=np.zeros(2,bool);self.last_vel=self.data.qvel[self.va].copy();self.steps=0;self.return_=0.;self.last_obs=None
  self.slide_path=0.;self.contact_force=np.zeros(2);self.contact_speed=np.zeros(2)
  return self.observation()
 def observation(self):
  cmd=self.motion['cmd'][self.k];v=self.data.qvel[self.va].copy();v[[4,5,10,11]]=0
  current=np.r_[cmd,self.data.qvel[3:6]*.25,euler(self.data.qpos[3:7])[:2],self.data.qpos[self.qa]-DEFAULT,v*.05,self.action].astype(np.float32)
  self.last_obs=np.r_[current,self.hist.ravel(),cmd].astype(np.float32);self.hist[:-1]=self.hist[1:];self.hist[-1]=current
  return self.last_obs
 def contact_sample(self):
  # A shadow state preserves mj_step solver warm-start state and exact dynamics.
  m=self.model;d=self.contact_data
  mujoco.mj_getState(m,self.data,self.contact_state,self.contact_state_spec);mujoco.mj_setState(m,d,self.contact_state,self.contact_state_spec);mujoco.mj_forward(m,d)
  fn=np.zeros(2);v2=np.zeros(2);force=np.zeros(6)
  for j in range(d.ncon):
   c=d.contact[j];g1,g2=int(c.geom1),int(c.geom2)
   if self.floor not in (g1,g2):continue
   body=int(m.geom_bodyid[g2 if g1==self.floor else g1]);sides=np.flatnonzero(self.feet==body)
   if not len(sides):continue
   side=sides[0];mujoco.mj_contactForce(m,d,j,force);normal=max(0.,force[0])
   if normal<=0:continue
   mujoco.mj_jac(m,d,self.contact_jac,None,c.pos,body)
   tangent=c.frame.reshape(3,3)[1:]@(self.contact_jac@d.qvel)
   fn[side]+=normal;v2[side]+=normal*float(tangent@tangent)
  self.contact_force=fn;self.contact_speed=np.sqrt(v2/np.maximum(fn,1e-12))
  return sliding_rate(fn,self.contact_speed,self.reward_config)
 def critic_observation(self):
  # Privileged simulation state is critic-only; the deployed actor is unchanged.
  q=self.motion['q'][self.k];v=self.motion['v'][self.k]
  self.ref.qpos[:7]=q[:7];self.ref.qpos[self.qa]=q[7:];mujoco.mj_forward(self.model,self.ref)
  orientation=(Rotation.from_quat(q[[4,5,6,3]]).inv()*Rotation.from_quat(self.data.qpos[[4,5,6,3]])).as_rotvec()
  return np.r_[self.last_obs,self.data.qpos,self.data.qvel,
   self.data.qpos[:3]-q[:3],self.data.qpos[self.qa]-q[7:],
   self.data.qvel[:6]-v[:6],self.data.qvel[self.va]-v[6:],orientation,
   (self.data.xpos[self.key]-self.ref.xpos[self.key]).ravel(),
   self.contact_force/350.,self.contact_speed,self.air,self.lastcontact.astype(float),self.last_vel,
   (len(self.motion['q'])-1-self.k)*DT,
   future_reference_features(self.motion,self.k,self.data.qpos[:3],self.data.qpos[3:7])].astype(np.float32)
 def step(self,action):
  action=np.asarray(action);assert action.shape==(29,) and np.isfinite(action).all();previous_action=self.action.copy();self.action=action.copy();target=DEFAULT+.5*np.clip(action,-10,10)
  slide_integral=0.
  for _ in range(10):
   self.data.ctrl[self.aids]=np.clip((target-self.data.qpos[self.qa])*KP-self.data.qvel[self.va]*KD,-KP,KP);mujoco.mj_step(self.model,self.data)
   slide_integral+=self.model.opt.timestep*self.contact_sample()
  # mj_step's cached FK precedes the integration; reward evaluates the resulting state.
  mujoco.mj_forward(self.model,self.data)
  self.k=min(self.k+1,len(self.motion['q'])-1);q=self.motion['q'][self.k];v=self.motion['v'][self.k];self.ref.qpos[:7]=q[:7];self.ref.qpos[self.qa]=q[7:];mujoco.mj_forward(self.model,self.ref)
  currentq=self.data.qpos;currentv=self.data.qvel;dq=currentq[self.qa]-q[7:];dv=currentv[self.va]-v[6:]
  rr=Rotation.from_quat(q[[4,5,6,3]]);cr=Rotation.from_quat(currentq[[4,5,6,3]]);angle=(rr.inv()*cr).magnitude();local=Rotation.from_euler('z',euler(currentq[3:7])[2]).inv().apply(self.data.xpos[self.key]-currentq[:3]);ref_local=Rotation.from_euler('z',euler(q[3:7])[2]).inv().apply(self.ref.xpos[self.key]-q[:3])
  forces=np.zeros((2,3));f=np.zeros(6)
  for j in range(self.data.ncon):
   co=self.data.contact[j];g1,g2=int(co.geom1),int(co.geom2)
   if self.floor not in (g1,g2):continue
   other=g2 if g1==self.floor else g1;body=self.model.geom_bodyid[other]
   side=np.flatnonzero(self.feet==body)
   if len(side):
    mujoco.mj_contactForce(self.model,self.data,j,f);world=co.frame.reshape(3,3).T@f[:3];forces[side[0]]+=world if g1==self.floor else -world
  contact=abs(forces[:,2])>5;speed=[]
  for body in self.feet:
   six=np.zeros(6);mujoco.mj_objectVelocity(self.model,self.data,mujoco.mjtObj.mjOBJ_BODY,int(body),six,0);speed.append(np.linalg.norm(six[3:5]))
  speed=np.array(speed);slip=np.sum(np.sqrt(speed)*contact);stumble=float(np.any(np.linalg.norm(forces[:,:2],axis=1)>4*abs(forces[:,2])))
  cf=contact|self.lastcontact;first=(self.air>0)&cf;self.air+=DT;air=np.minimum(self.air-.5,0)*first;airtime=float(air.sum()) if np.linalg.norm(self.motion['cmd'][self.k,:2])>.05 else 0.;self.air[cf]=0;self.lastcontact=contact
  vel=currentv[self.va];acc=(vel-self.last_vel)/DT;self.last_vel=vel.copy();torque=self.data.ctrl[self.aids]
  terms={'joint':2*np.exp(-.15*np.sum(dq*dq)),'joint_velocity':.2*np.exp(-.01*np.sum(dv*dv)),'height':np.exp(-5*(currentq[2]-q[2])**2),'rotation':np.exp(-5*angle**2),'linear_velocity':np.exp(-np.sum((rr.inv().apply(v[:3])-cr.inv().apply(currentv[:3]))**2)),'angular_velocity':np.exp(-np.sum((v[3:6]-currentv[3:6])**2)),'key_local':2*np.exp(-10*np.sum((local-ref_local)**2)),'key_global':global_tracking_reward(self.data.xpos[self.key]-self.ref.xpos[self.key],self.reward_config),'alive':.5,'slip':(-.1*slip if (self.reward_config.legacy or self.reward_config.legacy_slip) else -self.reward_config.slide_weight*slide_integral/DT),'stumble':(-1.25*stumble if self.reward_config.legacy else 0.),'force':-5e-4*max(0,np.linalg.norm(forces[:,2])-350),'position_limit':-5*np.sum(np.maximum(self.ranges[:,0]-currentq[self.qa],0)+np.maximum(currentq[self.qa]-self.ranges[:,1],0)),'torque_limit':-np.sum(np.maximum(abs(torque)/self.torque_limit-.95,0)),'velocity':-1e-4*np.sum(vel**2),'acceleration':-5e-8*np.sum(acc**2),'action_rate':-.05*discrete_rate(np.linalg.norm(action-previous_action),DT,self.reward_config),'airtime':5*discrete_rate(airtime,DT,self.reward_config),'angular_xy':-.01*np.sum(currentv[3:5]**2),'ankle_acceleration':-1e-7*np.sum(acc[[4,5,10,11]]**2),'ankle_velocity':-2e-4*np.sum(vel[[4,5,10,11]]**2)}
  self.reward_terms=terms.copy();self.slide_path+=slide_integral
  reward=float(sum(terms.values())*DT);tilt=float(np.arccos(np.clip(cr.as_matrix()[2,2],-1,1)));bad=any(self.data.warning[j].number for j in [mujoco.mjtWarning.mjWARN_BADQPOS,mujoco.mjtWarning.mjWARN_BADQVEL,mujoco.mjtWarning.mjWARN_BADQACC,mujoco.mjtWarning.mjWARN_BADCTRL]);fall=bool(currentq[2]<.35 or tilt>1.2 or bad or not np.isfinite(reward))
  if bad or not np.isfinite(reward):raise RuntimeError('Invalid MuJoCo numerical rollout')
  self.steps+=1;self.return_+=reward;truncated=self.k>=len(self.motion['q'])-1 or self.steps>=self.max_steps;done=fall or truncated
  info={'fall':fall,'truncated':truncated,'reference_end':self.k>=len(self.motion['q'])-1,'time_limit':self.steps>=self.max_steps,'seconds':self.steps*DT,'return':self.return_,'joint_rmse':float(np.sqrt(np.mean(dq*dq))),'key_rmse_m':float(np.sqrt(np.mean((local-ref_local)**2))),'slip_m_s':float(np.mean(speed[contact])) if contact.any() else 0.,'stumble':stumble,'root_xy_error_m':float(np.linalg.norm(currentq[:2]-q[:2])),'tilt_deg':float(np.degrees(tilt)),'slide_rate':slide_integral/DT,'slide_path_m':self.slide_path,'reward_terms':terms.copy(),'reward_rate':reward/DT}
  return self.observation(),reward,done,info
