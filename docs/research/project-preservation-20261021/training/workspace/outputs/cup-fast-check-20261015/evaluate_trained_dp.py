"""Closed-loop visual evaluation with native Quest physics/control and archived models.

Only the initial physical snapshot is restored. No recorded state or command is
injected during policy rollouts. Lower-controller hidden history starts empty;
recorded-command controls quantify this initialization limitation.
"""
from pathlib import Path
import argparse,ctypes,hashlib,json,sys,time
import numpy as np
import mujoco
from PIL import Image
O=Path(__file__).resolve().parent
T=O.parent/'cup-dp-training-20261015'
sys.path.insert(0,str(T))
from capture_snapshot import inspect_episode,columns

lib=ctypes.CDLL(str(O/'bridge.dylib'))
ptr=ctypes.c_void_p;dp=np.ctypeslib.ndpointer(np.float64,flags='C_CONTIGUOUS');fp=np.ctypeslib.ndpointer(np.float32,flags='C_CONTIGUOUS')
lib.create.argtypes=[ctypes.c_char_p];lib.create.restype=ptr
lib.destroy.argtypes=[ptr];lib.last_error.restype=ctypes.c_char_p
lib.restore.argtypes=[ptr,dp,dp,dp,dp,dp];lib.read_state.argtypes=[ptr,dp,dp,dp,dp,dp]
lib.step.argtypes=[ptr,fp,ctypes.c_int];lib.contacts.argtypes=[ptr]

class World:
 def __init__(self,e,render=False):
  self.e=e;self.m=mujoco.MjModel.from_binary_path(str(e['path']/'learning-model.mjb'));self.d=mujoco.MjData(self.m)
  self.p=lib.create(str(e['path']).encode())
  if not self.p:raise RuntimeError(lib.last_error().decode())
  m=self.m;i=e['start'];s=e['scene']
  self.q=columns(e['frames'],'qpos',m.nq)[i].copy();self.v=columns(e['frames'],'qvel',m.nv)[i].copy()
  self.u=np.array([s[f'userdata_{k}'][i] for k in range(m.nuserdata)],dtype=np.float64)
  self.mp=np.array([s[f'mocap_pos_{k}'][i] for k in range(3*m.nmocap)],dtype=np.float64)
  self.mq=np.array([s[f'mocap_quat_{k}'][i] for k in range(4*m.nmocap)],dtype=np.float64)
  assert self.u[3]==0 and self.u[4]==0
  lib.restore(self.p,self.q,self.v,self.u,self.mp,self.mq)
  self.gaze=e['actions'][i,37:39].copy();self.cq=int(m.joint('cup_free').qposadr[0]);self.start=self.q[self.cq:self.cq+3].copy()
  self.goal=self.mp[:3].copy() if m.nmocap else m.numeric('collection_goal_0').data[:3].copy()
  self.cam=m.camera('ego').id;self.renderer=mujoco.Renderer(m,height=360,width=640) if render else None
  self.m.light_castshadow[:]=0;self.read()
 def read(self):
  lib.read_state(self.p,self.q,self.v,self.u,self.mp,self.mq)
  d=self.d;d.qpos[:]=self.q;d.qvel[:]=self.v;d.userdata[:]=self.u;d.mocap_pos.flat[:]=self.mp;d.mocap_quat.flat[:]=self.mq
  mujoco.mj_forward(self.m,d)
 def image(self):
  self.read();yaw,pitch=self.gaze;cy,sy=np.cos(yaw),np.sin(yaw);cp,sp=np.cos(pitch),np.sin(pitch)
  rotation=self.d.cam_xmat[self.cam].reshape(3,3)@np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])@np.array([[1,0,0],[0,cp,-sp],[0,sp,cp]])
  self.renderer.update_scene(self.d,camera=self.cam)
  for c in self.renderer.scene.camera:
   c.pos[:]=self.d.cam_xpos[self.cam];c.forward[:]=-rotation[:,2];c.up[:]=rotation[:,1]
  self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=0
  return np.asarray(Image.fromarray(self.renderer.render()).resize((224,224),Image.Resampling.BILINEAR))
 def step(self,a,n):
  result=lib.step(self.p,np.asarray(a,dtype=np.float32),n);self.gaze=np.asarray(a[37:39]);self.read();return result
 def close(self):
  if self.renderer:self.renderer.close()
  lib.destroy(self.p)

def run(e,mode,model,norm,device,seed,limit,execution=48,label=""):
 import torch
 torch.manual_seed(seed);np.random.seed(seed)
 w=World(e,mode=='policy');started=time.time();trace=[];states=[];commands=[];velocities=[];outcome=0;maxlift=0.;contact_frames=0;minheight=w.q[2];infer=[]
 previous=np.zeros(39,np.float32);prior_image=None;prior_command=previous.copy();actions=None
 maxsteps=int(limit*(100 if mode=='oracle100' else 30));idx=e['indices'];elapsed=0
 try:
  for k in range(maxsteps):
   if mode=='policy':
    if k%execution in (0,execution-1):
     im=w.image()
     if k==0:
      prior_image=im;Image.fromarray(im).save(O/f"initial-{e['path'].name}.png")
     if k%execution==0:
      b={'image':torch.from_numpy(np.stack([prior_image,im]).transpose(0,3,1,2).copy())[None].to(device),
         'command':torch.from_numpy(np.stack([prior_command,previous])*norm['obs/previous_command']['scale']+norm['obs/previous_command']['offset'])[None].to(device)}
      ti=time.time()
      from train_dp import scheduler
      with torch.inference_mode():actions=model.predict(b,scheduler()).cpu().numpy()[0]
      infer.append(time.time()-ti)
      assert np.isfinite(actions).all()
     else:prior_image=im;prior_command=previous.copy()
    a=actions[k%execution];n=round((k+1)*1000/30)-round(k*1000/30)
   elif mode=='oracle100':
    j=min(e['start']+k,e['end']);a=e['actions'][j];n=10
   else:
    a=e['actions'][idx[min(k,len(idx)-1)]];n=round((k+1)*1000/30)-round(k*1000/30)
   outcome=w.step(a,n);elapsed+=n/1000;previous=np.asarray(a,np.float32).copy()
   cup=w.q[w.cq:w.cq+3];maxlift=max(maxlift,float(cup[2]-w.start[2]));contact=lib.contacts(w.p);contact_frames+=contact>0;minheight=min(minheight,float(w.q[2]))
   trace.append([elapsed,*cup,np.linalg.norm(cup[:2]-w.goal[:2]),contact,*w.u]);states.append(w.q.copy());velocities.append(w.v.copy());commands.append(previous.copy())
   if k%300==0:print(json.dumps(dict(progress_s=elapsed,label=label,episode=e["path"].name,execution=execution)),flush=True)
   if outcome:break
  result=dict(execution_steps=execution,label=label,episode=e['path'].name,mode=mode,seed=seed,success=outcome==1,outcome=outcome,elapsed_sim_s=elapsed,wall_s=time.time()-started,lift_m=maxlift,contact_frames=int(contact_frames),final_goal_xy_error_m=float(np.linalg.norm(w.q[w.cq:w.cq+2]-w.goal[:2])),initial_distance_m=float(np.linalg.norm(w.start[:2]-w.goal[:2])),min_pelvis_z=minheight,final_cup=w.q[w.cq:w.cq+3].tolist(),goal=w.goal.tolist(),error=lib.last_error().decode() if outcome<0 else None,inference_mean_s=float(np.mean(infer)) if infer else None)
  name=f"{label}-{mode}-h{execution}-{e['path'].name}-s{seed}"
  np.savez_compressed(O/(name+'.npz'),qpos=states,qvel=velocities,actions=commands,trace=trace,mocap_pos=w.mp,mocap_quat=w.mq)
  (O/(name+'.json')).write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True);return result
 finally:w.close()

def main():
 p=argparse.ArgumentParser();p.add_argument('--denoise-steps',type=int,default=10);p.add_argument('--mode',choices=['policy','oracle100','oracle30'],default='policy');p.add_argument('--device',default='mps');p.add_argument('--seeds',type=int,nargs='+',default=[0]);p.add_argument('--limit',type=float,default=90);p.add_argument('--split',default='valid');p.add_argument('--count',type=int,default=4);p.add_argument("--checkpoint",type=Path,default=O/"best.pt");p.add_argument("--execution-steps",type=int,choices=range(2,49),default=48);p.add_argument("--label",default="check");p.add_argument("--offset",type=int,default=0);args=p.parse_args()
 import torch
 torch.set_num_threads(4)
 manifest=json.loads((O.parent/'cup-training-plan-20261015/training-manifest.json').read_text());rows=manifest['splits'][args.split][args.offset:args.offset+args.count]
 model=norm=None
 if args.mode=='policy':
  from train_dp import Policy
  import train_dp
  ck=torch.load(args.checkpoint,map_location='cpu',weights_only=False);train_dp.PREDICTION_TYPE=ck.get('metadata',{}).get('prediction_type','epsilon');norm=ck['normalizer'];model=Policy(None,norm,load_pretrained=False);model.load_state_dict(ck['ema']);model.eval().to(args.device)
  print(json.dumps(dict(checkpoint_epoch=ck['epoch'],checkpoint_step=ck['step'],validation_noise_mse=ck.get('validation_noise_mse'),device=args.device)),flush=True)
 if model is not None:
  import types
  def predict_fast(self,b,scheduler):
   self.eval();cond=self.encode(b);x=torch.randn((len(cond),64,39),device=cond.device)
   scheduler.set_timesteps(args.denoise_steps,device=cond.device)
   for t in scheduler.timesteps:
    pred=self.unet(x,t,global_cond=cond);x=scheduler.step(pred,t,x).prev_sample
   return (x[:,1:49]-self.action_offset)/self.action_scale
  model.predict=types.MethodType(predict_fast,model)
 results=[]
 for row in rows:
  e=inspect_episode(Path(manifest['raw_root'])/row['episode'])
  for seed in args.seeds:results.append(run(e,args.mode,model,norm,args.device,seed,args.limit,args.execution_steps,args.label))
 (O/f'results-{args.label}-{args.mode}-h{args.execution_steps}-{args.split}.json').write_text(json.dumps(results,indent=2))
if __name__=='__main__':main()
