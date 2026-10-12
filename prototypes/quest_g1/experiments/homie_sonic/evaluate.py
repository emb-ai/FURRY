"""Paired deterministic evaluation: TWIST2, HOMIE v2, SONIC planner + VR3.

No fitting on these clips. Full survival and matched-prefix metrics are separate.
Contact braking is a reference-independent proxy, not a labelled trip count.
"""
from pathlib import Path
import argparse,sys,json,time,hashlib
import numpy as np
import mujoco
APP=Path(__file__).resolve().parents[2];sys.path.insert(0,str(APP))
from g1_sim.autonomous_legs import Homie,Sonic,rot,quat
from g1_sim.controller import Controller,JOINTS,euler
from scipy.spatial.transform import Rotation,Slerp


def reference_at(q,fps,t):
 x=min(max(t*fps,0),len(q)-1);lo=int(x);hi=min(lo+1,len(q)-1);w=x-lo
 out=q[lo]*(1-w)+q[hi]*w
 if hi!=lo:out[3:7]=quat(Slerp([0,1],Rotation.from_quat(q[[lo,hi],3:7][:,[1,2,3,0]]))([w])[0])
 return out

def command_at(q,fps,t,data,catchup=True):
 ref=reference_at(q,fps,t);nxt=reference_at(q,fps,t+.02)
 rv=rot(ref[3:7]);av=rot(data.qpos[3:7]);vel=(nxt[:3]-ref[:3])/.02
 local=av.inv().apply(vel)
 yaw=rv.as_euler('xyz')[2];dyaw=np.arctan2(np.sin(euler(nxt[3:7])[2]-yaw),np.cos(euler(nxt[3:7])[2]-yaw))/.02
 cmd=np.r_[local[:2],ref[2],euler(ref[3:7])[:2],dyaw,ref[7:36]]
 if catchup:
  correction=av.inv().apply(np.r_[ref[:2]-data.qpos[:2],0])[:2]
  correction*=min(1,.4/max(np.linalg.norm(correction),1e-9));cmd[:2]+=correction
  err=np.arctan2(np.sin(yaw-euler(data.qpos[3:7])[2]),np.cos(yaw-euler(data.qpos[3:7])[2]))
  cmd[5]+=np.clip(3*err,-1,1)
 cmd[:2]*=min(1,.8/max(np.linalg.norm(cmd[:2]),1e-9));cmd[5]=np.clip(cmd[5],-1.2,1.2)
 return cmd,ref

class Metrics:
 def __init__(self,m):
  self.m=m;self.ref=mujoco.MjData(m);self.qa=np.array([m.joint(n).qposadr[0] for n in JOINTS])
  self.feet=[m.body(n+'_ankle_roll_link').id for n in ['left','right']]
  self.upper=[m.body(n).id for n in ['torso_link','left_shoulder_pitch_link','right_shoulder_pitch_link','left_elbow_link','right_elbow_link','left_wrist_yaw_link','right_wrist_yaw_link']]
  self.floor=m.geom('floor').id
  self.foot_geoms=[[i for i in range(m.ngeom) if m.geom_bodyid[i]==b and ((m.geom_contype[i]&m.geom_conaffinity[self.floor]) or (m.geom_conaffinity[i]&m.geom_contype[self.floor]))] for b in self.feet]
  self.rows=[];self.previous_speed=np.zeros(2);self.previous_vertical=np.zeros(2);self.previous_clearance=np.zeros(2);self.prev_contact=np.ones(2,bool);self.last_event=np.full(2,-1.)
 def sample(self,d,q):
  m=self.m;r=self.ref;r.qpos[:7]=q[:7];r.qpos[self.qa]=q[7:36];mujoco.mj_forward(m,r)
  contact=np.zeros(2);weighted=np.zeros(2);speeds=np.zeros(2);vertical=np.zeros(2);clearance=np.ones(2)*10
  for k,b in enumerate(self.feet):
   v=np.zeros(6);mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_BODY,b,v,0);speeds[k]=np.linalg.norm(v[3:5]);vertical[k]=v[5]
   for g in self.foot_geoms[k]:
    # Exact lowest point for the runtime's sphere sole collision geometries.
    if m.geom_type[g]!=mujoco.mjtGeom.mjGEOM_SPHERE:raise ValueError('Expected sphere soles')
    clearance[k]=min(clearance[k],d.geom_xpos[g,2]-m.geom_size[g,0])
  for ci in range(d.ncon):
   c=d.contact[ci];g1,g2=c.geom1,c.geom2
   if self.floor not in [g1,g2]:continue
   other=g2 if g1==self.floor else g1
   for k,geoms in enumerate(self.foot_geoms):
    if other not in geoms:continue
    f=np.zeros(6);mujoco.mj_contactForce(m,d,ci,f);load=max(0,f[0]);b=self.feet[k]
    # Jacobian at actual contact point, including angular contribution.
    jp=np.zeros((3,m.nv));jr=np.zeros((3,m.nv));mujoco.mj_jac(m,d,jp,jr,c.pos,b)
    v=jp@d.qvel;normal=np.asarray(c.frame).reshape(3,3)[0];tan=v-normal*np.dot(v,normal)
    contact[k]+=load;weighted[k]+=load*np.linalg.norm(tan)
  on=contact>10
  # touchdown + preceding horizontal motion + rapid braking; no virtual-leg phase.
  braking=(on&~self.prev_contact&(self.previous_speed>.15)&((self.previous_speed-speeds)>.08)&(d.time-self.last_event>.15))
  upward=braking&(self.previous_vertical>.04)&(self.previous_clearance>.003)
  self.last_event[braking]=d.time
  rr=rot(q[3:7]);ar=rot(d.qpos[3:7]);up_world=np.linalg.norm(d.xpos[self.upper]-r.xpos[self.upper],axis=1)
  up_local=np.linalg.norm(ar.inv().apply(d.xpos[self.upper]-d.qpos[:3])-rr.inv().apply(r.xpos[self.upper]-q[:3]),axis=1)
  angles=[]
  for b in [self.upper[0],*self.upper[-2:]]:angles.append((rot(d.xquat[b]).inv()*rot(r.xquat[b])).magnitude()*180/np.pi)
  yawerr=np.arctan2(np.sin(euler(d.qpos[3:7])[2]-euler(q[3:7])[2]),np.cos(euler(d.qpos[3:7])[2]-euler(q[3:7])[2]))
  row=dict(time=d.time,root_xy_m=float(np.linalg.norm(d.qpos[:2]-q[:2])),yaw_abs_deg=abs(float(yawerr))*180/np.pi,
      root_x=float(d.qpos[0]),root_y=float(d.qpos[1]),ref_x=float(q[0]),ref_y=float(q[1]),
      height=float(d.qpos[2]),tilt_deg=float(np.arccos(np.clip(ar.as_matrix()[2,2],-1,1))*180/np.pi),
      upper_world_mean_m=float(up_world.mean()),upper_local_mean_m=float(up_local.mean()),
      wrist_world_mean_m=float(up_world[-2:].mean()),wrist_local_mean_m=float(up_local[-2:].mean()),
      torso_world_m=float(up_world[0]),torso_orientation_deg=angles[0],wrist_orientation_deg=float(np.mean(angles[1:])),
      upper_joint_rmse_rad=float(np.sqrt(np.mean((d.qpos[self.qa[12:]]-q[19:36])**2))),
      contact_slip_m_s=float(weighted.sum()/max(contact.sum(),1e-9)),normal_load_n=float(contact.sum()),
      slide_foot_fraction=float(np.mean(on&(weighted/np.maximum(contact,1e-9)>.05))),
      low_moving_foot_fraction=float(np.mean((clearance<.01)&(speeds>.15))),
      braking_contact_events=int(braking.sum()),upward_contact_events=int(upward.sum()),clearance_left_m=float(clearance[0]),clearance_right_m=float(clearance[1]))
  self.rows.append(row);self.prev_contact=on;self.previous_speed=speeds;self.previous_vertical=vertical;self.previous_clearance=clearance


def summarize(rows):
 out={'samples':len(rows)}
 if not rows:return out
 for k in (['camera_xy_m'] if 'camera_xy_m' in rows[0] else [])+['root_xy_m','yaw_abs_deg','upper_world_mean_m','upper_local_mean_m','wrist_world_mean_m','wrist_local_mean_m','torso_world_m','torso_orientation_deg','wrist_orientation_deg','upper_joint_rmse_rad','contact_slip_m_s','slide_foot_fraction','low_moving_foot_fraction']:
  a=np.array([r[k] for r in rows]);out[k]={'mean':float(a.mean()),'p95':float(np.percentile(a,95))}
 out['braking_contact_events']=sum(r['braking_contact_events'] for r in rows)
 out['upward_contact_events']=sum(r['upward_contact_events'] for r in rows)
 a=np.array([[r['root_x'],r['root_y']] for r in rows]);r=np.array([[x['ref_x'],x['ref_y']] for x in rows]);dr=np.diff(r,axis=0);da=np.diff(a,axis=0);length=np.linalg.norm(dr,axis=1)
 out['reference_path_m']=float(length.sum());out['actual_path_m']=float(np.linalg.norm(da,axis=1).sum())
 out['directed_progress_ratio']=float(np.sum(np.sum(da*dr,axis=1)/np.maximum(length,1e-8))/max(length.sum(),1e-8))
 return out


def run(args,path,backend):
 raw=np.load(path);q=raw['qpos'].astype(float);fps=float(raw['fps']);duration=min((len(q)-1)/fps,args.seconds)
 m=mujoco.MjModel.from_xml_path(str(args.assets/'scene-stand.xml'));m.opt.timestep=.001
 if backend=='homie':p=Homie(m,args.assets)
 elif backend=='sonic':p=Sonic(m,args.assets,args.models,Path(__file__).with_name('sonic_constants.json'),args.seed)
 else:p=Controller(m,policy=args.assets/args.twist_policy,frequency=50)
 d=mujoco.MjData(m);p.initialize(d)
 qa=np.array([m.joint(n).qposadr[0] for n in JOINTS]);d.qpos[0:2]=q[0,:2];d.qpos[3:7]=quat(Rotation.from_euler('z',euler(q[0,3:7])[2]));mujoco.mj_forward(m,d)
 if 'camera_goal' in raw:
  cid=m.camera('ego').id;g=raw['camera_goal'][0];R=d.cam_xmat[cid].reshape(3,3);yaw=np.arctan2(-R[1,2],-R[0,2]);d.qpos[3:7]=quat(Rotation.from_euler('z',g[2]-yaw)*rot(d.qpos[3:7]));mujoco.mj_forward(m,d);d.qpos[:2]+=g[:2]-d.cam_xpos[cid,:2];mujoco.mj_forward(m,d)
 initial=np.r_[d.qpos[:7],d.qpos[qa]];metric=Metrics(m);camera_id=m.camera('ego').id;states=[];refs=[];commands=[];controls=[];failure=None;start=time.monotonic()
 for i in range(int(duration*50)):
  cmd,ref=command_at(q,fps,i*.02,d,not args.no_catchup)
  if 'commands' in raw:
   index=min(round(i*.02*fps),len(q)-1);cmd=raw['commands'][index].copy();goal=raw['camera_goal'][index]
   # Exact native camera catch-up v2, with this rollout's current camera pose.
   cmd[:2]=rot(d.qpos[3:7]).inv().apply(rot(ref[3:7]).apply([cmd[0],cmd[1],0]))[:2]
   if not args.no_catchup:
    actual=d.cam_xpos[camera_id];R=d.cam_xmat[camera_id].reshape(3,3);yaw=np.arctan2(-R[1,2],-R[0,2])
    correction=rot(d.qpos[3:7]).inv().apply([goal[0]-actual[0],goal[1]-actual[1],0])[:2]
    correction*=min(1,.4/max(np.linalg.norm(correction),1e-9));cmd[:2]+=correction
    dy=np.arctan2(np.sin(goal[2]-yaw),np.cos(goal[2]-yaw));cmd[5]+=np.clip(3*dy,-1,1)
   cmd[:2]*=min(1,.8/max(np.linalg.norm(cmd[:2]),1e-9));cmd[5]=np.clip(cmd[5],-1.2,1.2)
  # Same half-second activation blend for all modes; no future recorded leg inputs to autonomous modes.
  b=min(1,i*.02/.5);cmd[:2]*=b;cmd[5]*=b;cmd[2]=initial[2]+b*(cmd[2]-initial[2]);cmd[6:]=initial[7:]+b*(cmd[6:]-initial[7:])
  if backend!='twist2':p.update(d,cmd,ref)
  for _ in range(20):
   if backend=='twist2':p.step(d,cmd)
   else:p.apply(d)
  mujoco.mj_forward(m,d);metric.sample(d,reference_at(q,fps,d.time));
  if 'camera_goal' in raw:
   goal=raw['camera_goal'][min(round(d.time*fps),len(q)-1)];metric.rows[-1]['camera_xy_m']=float(np.linalg.norm(d.cam_xpos[camera_id,:2]-goal[:2]))
  states.append(np.r_[d.qpos,d.qvel]);refs.append(ref);commands.append(cmd);controls.append(d.ctrl.copy())
  if abs(d.time-(i+1)*.02)>1e-8:failure='physics_time_reset';break
  if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():failure='nonfinite';break
  if any(w.number for w in d.warning):failure='physics_warning';break
  if d.qpos[2]<.35 or metric.rows[-1]['tilt_deg']>65:failure='fall';break
 result=dict(clip=path.stem,backend=backend,twist_policy=args.twist_policy if backend=='twist2' else None,seed=args.seed,planned_seconds=duration,simulated_seconds=float(d.time),failure=failure,wall_seconds=time.monotonic()-start,metrics=summarize(metric.rows),
             inference_ms_median=float(np.median(p.inference_ms)),planner_calls=getattr(p,'planner_calls',0),planner_ms_median=float(np.median(p.planner_ms)) if hasattr(p,'planner_ms') else None)
 target=args.output/f'{backend}-{path.stem}-s{args.seed}';target.parent.mkdir(parents=True,exist_ok=True)
 np.savez_compressed(str(target)+'.npz',state=np.array(states),reference=np.array(refs),commands=np.array(commands),ctrl=np.array(controls),qa=qa)
 Path(str(target)+'.json').write_text(json.dumps(result,indent=2)+'\n');Path(str(target)+'.rows.json').write_text(json.dumps(metric.rows)+'\n')
 print(json.dumps(result),flush=True)
 return result

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--assets',type=Path,default=APP/'android/assets');ap.add_argument('--models',type=Path,required=True);ap.add_argument('--dataset',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--backend',choices=['twist2','homie','sonic'],required=True);ap.add_argument('--seed',type=int,default=11);ap.add_argument('--seconds',type=float,default=1e9);ap.add_argument('--clip');ap.add_argument('--twist-policy',default='policy.onnx');ap.add_argument('--no-catchup',action='store_true');args=ap.parse_args()
 paths=[args.dataset] if args.dataset.is_file() else sorted(args.dataset.glob('*.npz'))
 if args.clip:paths=[p for p in paths if args.clip in p.stem]
 for path in paths:run(args,path,args.backend)
