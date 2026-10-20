from pathlib import Path
import ctypes,json,time
import numpy as np
import random_eval as current
O=Path(__file__).resolve().parent;D=O/'restart-diagnostic';D.mkdir(exist_ok=True);ev=current.ev
lib=ctypes.CDLL(str(O/'bridge_audit.dylib'));ptr=ctypes.c_void_p;dp=np.ctypeslib.ndpointer(np.float64,flags='C_CONTIGUOUS');fp=np.ctypeslib.ndpointer(np.float32,flags='C_CONTIGUOUS')
lib.create.argtypes=[ctypes.c_char_p];lib.create.restype=ptr;lib.destroy.argtypes=[ptr];lib.last_error.restype=ctypes.c_char_p;lib.restore.argtypes=[ptr,dp,dp,dp,dp,dp];lib.read_state.argtypes=[ptr,dp,dp,dp,dp,dp];lib.step.argtypes=[ptr,fp,ctypes.c_int];lib.contacts.argtypes=[ptr];lib.teacher_frame.argtypes=[ptr,dp,dp,dp,dp,dp,ctypes.c_int];lib.set_phase.argtypes=[ptr,ctypes.c_int];ev.lib=lib
names=json.loads((O/'train-diagnostic/selection.json').read_text())['episodes'];rows=[]
for name in names:
 e=ev.inspect_episode(O/'recordings'/name);f=e['frames'];q=ev.columns(f,'qpos',57);v=ev.columns(f,'qvel',55);start=e['start'];end=e['end'];warm=50;assert e['scene']['userdata_4'][start+warm]==0
 for mode in ['phase_only','warm500ms']:
  w=ev.World(e,False);lib.set_phase(w.p,int(f['step'][start]));errors=[];t=time.time()
  try:
   if mode=='warm500ms':
    for j in range(start,start+warm):
     w.q[:]=q[j];w.v[:]=v[j]
     for dst,prefix in [(w.u,'userdata'),(w.mp,'mocap_pos'),(w.mq,'mocap_quat')]:dst[:]=[e['scene'][f'{prefix}_{k}'][j] for k in range(len(dst))]
     lib.teacher_frame(w.p,w.q,w.v,w.u,w.mp,w.mq,int(f['step'][j]));lib.step(w.p,np.ascontiguousarray(e['actions'][j],np.float32),10);w.read()
     if j+1<len(q):errors.append({'offset':int(j-start),'root_m':float(np.linalg.norm(w.q[:3]-q[j+1,:3])),'qpos_max_abs':float(np.max(np.abs(w.q-q[j+1]))),'joint_rms':float(np.sqrt(np.mean((w.q[7:36]-q[j+1,7:36])**2)))})
    j=start+warm;w.q[:]=q[j];w.v[:]=v[j]
    for dst,prefix in [(w.u,'userdata'),(w.mp,'mocap_pos'),(w.mq,'mocap_quat')]:dst[:]=[e['scene'][f'{prefix}_{k}'][j] for k in range(len(dst))]
    lib.teacher_frame(w.p,w.q,w.v,w.u,w.mp,w.mq,int(f['step'][j]));first=j
   else:first=start
   outcome=0
   for j in range(first,end+301):
    outcome=w.step(np.ascontiguousarray(e['actions'][min(j,end)],np.float32),10)
    if outcome:break
   row={'episode':name,'mode':mode,'success':outcome==1,'outcome':outcome,'wall_s':time.time()-t,'final_goal_error_m':float(np.linalg.norm(w.q[w.cq:w.cq+2]-w.goal[:2])),'teacher_forced_dynamics_errors':errors}
   rows.append(row);(D/'results.json').write_text(json.dumps(rows,indent=2));print(json.dumps({k:v for k,v in row.items() if k!='teacher_forced_dynamics_errors'}),flush=True)
  finally:w.close()
(D/'summary.json').write_text(json.dumps({m:{'successes':sum(r['success'] for r in rows if r['mode']==m),'n':sum(r['mode']==m for r in rows)} for m in ['phase_only','warm500ms']},indent=2))
