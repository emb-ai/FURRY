from pathlib import Path
import argparse,ctypes,json,sys,time
import numpy as np,torch
O=Path(__file__).resolve().parent;sys.path.insert(0,str(O.parent/'cup-fast-check-20261015'))
import evaluate_trained_dp as ev
lib=ctypes.CDLL(str(O/'bridge_fresh.dylib'));ptr=ctypes.c_void_p;dp=np.ctypeslib.ndpointer(np.float64,flags='C_CONTIGUOUS');fp=np.ctypeslib.ndpointer(np.float32,flags='C_CONTIGUOUS')
lib.create.argtypes=[ctypes.c_char_p];lib.create.restype=ptr;lib.destroy.argtypes=[ptr];lib.last_error.restype=ctypes.c_char_p;lib.restore.argtypes=[ptr,dp,dp,dp,dp,dp];lib.read_state.argtypes=[ptr,dp,dp,dp,dp,dp];lib.step.argtypes=[ptr,fp,ctypes.c_int];lib.contacts.argtypes=[ptr];lib.random_layout.argtypes=[ptr,ctypes.c_uint];lib.reset_random.argtypes=[ptr,ctypes.c_uint];lib.fresh_reset_flags.argtypes=[ptr];ev.lib=lib

def source():
 name=json.loads((O/'quality-audit.json').read_text())['accepted'][0]
 return ev.inspect_episode(O/'recordings'/name)

def prepare():
 e=source();seeds=np.random.default_rng(20261015).choice(2**31,size=10,replace=False).tolist();cases=[]
 for seed in seeds:
  w=ev.World(e,False)
  try:
   assert lib.reset_random(w.p,seed)==0;assert lib.fresh_reset_flags(w.p)==63;w.gaze=np.zeros(2,np.float32);w.read();assert w.u[3]==0
   cases.append({'seed':seed,'qpos':w.q.tolist(),'qvel':w.v.tolist(),'userdata':w.u.tolist(),'mocap_pos':w.mp.tolist(),'mocap_quat':w.mq.tolist(),'gaze':w.gaze.tolist()})
  finally:w.close()
 report={'schema':'native-cup-full-reset-v2','rng_seed':20261015,'source_episode':e['path'].name,'family':e['family'],'initialization':'Native Simulation.NewSceneTrial(seed): complete robot, physics, HOMIE, grip and task reset. Zero controller history, zero initial velocity, neutral gaze. No demonstration state or action prefix. Policy acts from first simulation step.','cases':cases,'all_59_demonstrations_in_training':True}
 p=O/'fresh-start/scenes.json';p.parent.mkdir(exist_ok=True);assert not p.exists();p.write_text(json.dumps(report,indent=2));print('Prepared 10 fixed native random scenes',flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--kind',choices=['act','dp-epsilon','dp-sample']);p.add_argument('--checkpoint',type=Path);p.add_argument('--stats',type=Path);p.add_argument('--label',default='baseline');p.add_argument('--device',default='cpu');args=p.parse_args()
 if args.prepare:prepare();return
 torch.set_num_threads(4);e=source();cases=json.loads((O/'fresh-start/scenes.json').read_text())['cases'];out=O/'fresh-start'/args.label;out.mkdir(parents=True,exist_ok=True);rows=[];device=args.device
 if args.kind=='act':
  model=torch.jit.load(str(args.checkpoint),map_location=device).eval();stats=json.loads(args.stats.read_text());stats={k:{kk:np.asarray(vv,np.float32) for kk,vv in v.items()} for k,v in stats.items()};execution=8
 else:
  import train_dp_resume as dpmod
  ck=torch.load(args.checkpoint,map_location='cpu',weights_only=False);norm=ck['normalizer'];dpmod.PREDICTION_TYPE=args.kind.split('-')[1];model=dpmod.Policy(None,norm,load_pretrained=False);model.load_state_dict(ck['ema'],strict=True);model.eval().to(device);execution=16;clip=float(json.loads((O/'normalization-range-check.json').read_text())['required_clip_sample_range']);del ck
 rows=json.loads((out/f'{args.kind}-results.json').read_text())
 assert args.kind=='act' and args.label=='finetuned' and len(rows)==9
 assert [r['seed'] for r in rows]==[c['seed'] for c in cases[:9]]
 assert all(r['error'] is None for r in rows)
 for case in cases[9:]:
  seed=case['seed'];torch.manual_seed(seed);np.random.seed(seed);w=ev.World(e,True);start=time.time();infer=[];trace=[];states=[];commands=[];velocities=[];previous=np.zeros(39,np.float32);prior_image=None;prior_command=previous.copy();elapsed=0;outcome=0;maxlift=0;maxtilt=0
  try:
   assert lib.reset_random(w.p,seed)==0;assert lib.fresh_reset_flags(w.p)==63;w.gaze=np.zeros(2,np.float32);w.read()
   for actual,key in [(w.q,'qpos'),(w.v,'qvel'),(w.u,'userdata'),(w.mp,'mocap_pos'),(w.mq,'mocap_quat')]:np.testing.assert_array_equal(actual,np.asarray(case[key]))
   w.start=w.q[w.cq:w.cq+3].copy();w.goal=w.mp[:3].copy()
   for k in range(2700):
    if args.kind=='act':
     if k%execution==0:
      im=w.image().transpose(2,0,1).astype(np.float32)/255;im=(im-np.array([.485,.456,.406],np.float32)[:,None,None])/np.array([.229,.224,.225],np.float32)[:,None,None];st=stats['obs/previous_command'];x=(previous-st['mean'])/st['std'];ti=time.time()
      with torch.inference_mode():actions=model(torch.from_numpy(im)[None].to(device),torch.from_numpy(x)[None].to(device)).cpu().numpy()[0]*stats['actions']['std']+stats['actions']['mean']
      infer.append(time.time()-ti)
    else:
     if k%execution in [0,execution-1]:
      im=w.image()
      if k==0:prior_image=im
      if k%execution==0:
       st=norm['obs/previous_command'];b={'image':torch.from_numpy(np.stack([prior_image,im]).transpose(0,3,1,2).copy())[None].to(device),'command':torch.from_numpy(np.stack([prior_command,previous])*st['scale']+st['offset'])[None].to(device)};ti=time.time();sched=dpmod.scheduler();sched.register_to_config(clip_sample_range=clip);sched.set_timesteps(10,device=device)
       with torch.inference_mode():
        cond=model.encode(b);x=torch.randn((1,64,39),device=device)
        for t in sched.timesteps:x=sched.step(model.unet(x,t,global_cond=cond),t,x).prev_sample
        actions=((x[:,1:49]-model.action_offset)/model.action_scale).cpu().numpy()[0]
       infer.append(time.time()-ti)
      else:prior_image=im;prior_command=previous.copy()
    a=np.ascontiguousarray(actions[k%execution],dtype=np.float32);assert np.isfinite(a).all();n=round((k+1)*1000/30)-round(k*1000/30);outcome=w.step(a,n);elapsed+=n/1000;previous=a.copy();cup=w.q[w.cq:w.cq+3];quat=w.q[w.cq+3:w.cq+7];tilt=float(np.degrees(np.arccos(np.clip(1-2*(quat[1]**2+quat[2]**2),-1,1))));maxlift=max(maxlift,float(cup[2]-w.start[2]));maxtilt=max(maxtilt,tilt)
    assert np.isfinite(w.q).all();trace.append([elapsed,*cup,lib.contacts(w.p),tilt,*w.u]);states.append(w.q.copy());velocities.append(w.v.copy());commands.append(a.copy())
    if outcome:break
   row={'kind':args.kind,'seed':seed,'success':outcome==1,'clean_success':outcome==1 and maxtilt<70 and tilt<=30,'outcome':outcome,'sim_s':elapsed,'wall_s':time.time()-start,'lift_m':maxlift,'max_cup_tilt_deg':maxtilt,'final_cup_tilt_deg':tilt,'goal_xy_error_m':float(np.linalg.norm(cup[:2]-w.goal[:2])),'inference_mean_s':float(np.mean(infer)),'error':lib.last_error().decode() if outcome<0 else None}
   np.savez_compressed(out/f'{args.kind}-{seed}.npz',qpos=states,qvel=velocities,actions=commands,trace=trace,mocap_pos=w.mp,mocap_quat=w.mq);rows.append(row);(out/f'{args.kind}-results.json').write_text(json.dumps(rows,indent=2));print(json.dumps(row),flush=True)
  finally:w.close()
 (out/f'{args.kind}-summary.json').write_text(json.dumps({'status':'completed','successes':sum(r['success'] for r in rows),'n':len(rows),'checkpoint':str(args.checkpoint),'execution_steps':execution,'source':'10 native full-reset random scenes; zero history, no demonstration prefix','results':rows},indent=2))

if __name__=='__main__':main()
