from pathlib import Path
import ctypes,json,sys,time
import numpy as np,mujoco
O=Path(__file__).resolve().parent
sys.path.insert(0,str(O));from evaluate import World,lib,inspect_episode,columns,dp,ptr
lib.teacher_frame.argtypes=[ptr,dp,dp,dp,dp,dp,ctypes.c_int];lib.dump_model.argtypes=[ptr,ctypes.c_char_p]
manifest=json.loads((O.parent/'cup-training-plan-20261015/training-manifest.json').read_text())
rows=[]
for row in manifest['splits']['valid']+manifest['splits']['train'][:1]:
 e=inspect_episode(Path(manifest['raw_root'])/row['episode']);q=columns(e['frames'],'qpos',57);v=columns(e['frames'],'qvel',55)
 for warm in [0,.2,.5]:
  w=World(e);start=e['start'];cut=start+round(warm*100);m=w.m;scene=e['scene']
  def inject(i):
   vals=[q[i].copy(),v[i].copy()]
   for name,count in [('userdata',m.nuserdata),('mocap_pos',m.nmocap*3),('mocap_quat',m.nmocap*4)]:vals.append(np.array([scene[f'{name}_{k}'][i] for k in range(count)],dtype=np.float64))
   lib.teacher_frame(w.p,*vals,int(e['frames']['step'][i]))
  # Only a recorded prefix initializes the controller. No state forcing after cut.
  for i in range(start,cut):inject(i);w.step(e['actions'][i],10)
  if warm:inject(cut)
  w.read();initial=w.q.copy();assert e['scene']['userdata_4'][cut]==0
  errors=[];maxlift=0;out=0
  for k in range(3000):
   i=min(cut+k,e['end']);out=w.step(e['actions'][i],10);maxlift=max(maxlift,float(w.q[w.cq+2]-w.start[2]))
   if i<e['end']:errors.append([k*.01,float(np.linalg.norm(w.q[:3]-q[i+1,:3])),float(np.max(abs(w.q-q[i+1])))])
   if out:break
  r=dict(episode=row['episode'],warm_prefix_s=warm,success=out==1,outcome=out,lift_m=maxlift,final_xy_error_m=float(np.linalg.norm(w.q[w.cq:w.cq+2]-w.goal[:2])),initial_error=float(np.max(abs(initial-q[cut]))),first_second_root_error_m=max((x[1] for x in errors if x[0]<1),default=None),seconds=(k+1)*.01)
  print(json.dumps(r),flush=True);rows.append(r);w.close();(O/'replay-check.json').write_text(json.dumps(rows,indent=2))
