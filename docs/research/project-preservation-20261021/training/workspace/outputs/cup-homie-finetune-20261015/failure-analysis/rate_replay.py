from pathlib import Path
import json,sys,time,ctypes
import numpy as np,h5py,torch
O=Path(__file__).resolve().parents[1];D=Path(__file__).resolve().parent;sys.path.insert(0,str(O));import random_eval as cur
ev=cur.ev;lib=ctypes.CDLL(str(O/'bridge_audit.dylib'));ptr=ctypes.c_void_p;dp=np.ctypeslib.ndpointer(np.float64,flags='C_CONTIGUOUS');fp=np.ctypeslib.ndpointer(np.float32,flags='C_CONTIGUOUS')
lib.create.argtypes=[ctypes.c_char_p];lib.create.restype=ptr;lib.destroy.argtypes=[ptr];lib.last_error.restype=ctypes.c_char_p;lib.restore.argtypes=[ptr,dp,dp,dp,dp,dp];lib.read_state.argtypes=[ptr,dp,dp,dp,dp,dp];lib.step.argtypes=[ptr,fp,ctypes.c_int];lib.contacts.argtypes=[ptr];lib.teacher_frame.argtypes=[ptr,dp,dp,dp,dp,dp,ctypes.c_int];ev.lib=lib
names=json.loads((O/'train-diagnostic/selection.json').read_text())['episodes'];rows=[]
with h5py.File(O/'new-cup.hdf5') as h:
 for ep in names:
  e=ev.inspect_episode(O/'recordings'/ep);demo=next(v for v in h['data'].values() if v.attrs['episode']==ep);a=demo['actions'][:];prev=demo['obs/previous_command'][:];rgb=demo['obs/agentview_image'][:];predicted=a.copy()
  for mode in ['recorded100','linear30_at100']:
   w=ev.World(e,False);q=ev.columns(e['frames'],'qpos',w.m.nq);v=ev.columns(e['frames'],'qvel',w.m.nv);first=int(e['indices'][15]);assert e['scene']['userdata_4'][first]==0
   def frame(j):
    w.q[:]=q[j];w.v[:]=v[j]
    for dst,prefix in [(w.u,'userdata'),(w.mp,'mocap_pos'),(w.mq,'mocap_quat')]:dst[:]=[e['scene'][f'{prefix}_{kk}'][j] for kk in range(len(dst))]
    lib.teacher_frame(w.p,w.q,w.v,w.u,w.mp,w.mq,int(e['frames']['step'][j]))
   for j in range(e['start'],first):frame(j);lib.step(w.p,np.ascontiguousarray(e['actions'][j],np.float32),10)
   frame(first);outcome=0
   try:
    sample_t=e['frames']['sim_time'][e['indices']]
    for k in range(first,e['end']+301):
     t=e['frames']['sim_time'][min(k,e['end'])]
     command=e['actions'][min(k,e['end'])] if mode=='recorded100' else np.array([np.interp(t,sample_t,a[:,dim]) for dim in range(39)],np.float32)
     outcome=w.step(np.ascontiguousarray(command,np.float32),10)
     if outcome:break
    row={'episode':ep,'mode':mode,'success':outcome==1,'outcome':outcome,'error':lib.last_error().decode() if outcome<0 else None,'final_distance_m':float(np.linalg.norm(w.q[w.cq:w.cq+2]-w.goal[:2]))};rows.append(row);(D/'rate-replay-results.json').write_text(json.dumps(rows,indent=2));print(json.dumps(row),flush=True)
   finally:w.close()
report={'protocol':'Same warm0.5s continuation. Original100Hz versus linearly interpolated30Hz sampled operator commands, applied100Hz. Interpolation uses future samples offline; not autonomous success.','summary':{mode:{'successes':sum(x['success'] for x in rows if x['mode']==mode),'n':sum(x['mode']==mode for x in rows)} for mode in ['recorded100','linear30_at100']},'errors':[r for r in rows if r['error']],'results':rows};(D/'rate-replay.json').write_text(json.dumps(report,indent=2))
