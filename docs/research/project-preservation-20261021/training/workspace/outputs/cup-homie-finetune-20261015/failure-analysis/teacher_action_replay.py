from pathlib import Path
import json,sys,time,ctypes
import numpy as np,h5py,torch
O=Path(__file__).resolve().parents[1];D=Path(__file__).resolve().parent;sys.path.insert(0,str(O));import random_eval as cur
ev=cur.ev;lib=ctypes.CDLL(str(O/'bridge_audit.dylib'));ptr=ctypes.c_void_p;dp=np.ctypeslib.ndpointer(np.float64,flags='C_CONTIGUOUS');fp=np.ctypeslib.ndpointer(np.float32,flags='C_CONTIGUOUS')
lib.create.argtypes=[ctypes.c_char_p];lib.create.restype=ptr;lib.destroy.argtypes=[ptr];lib.last_error.restype=ctypes.c_char_p;lib.restore.argtypes=[ptr,dp,dp,dp,dp,dp];lib.read_state.argtypes=[ptr,dp,dp,dp,dp,dp];lib.step.argtypes=[ptr,fp,ctypes.c_int];lib.contacts.argtypes=[ptr];lib.teacher_frame.argtypes=[ptr,dp,dp,dp,dp,dp,ctypes.c_int];ev.lib=lib
s=json.loads((O/'finetuned-weights/act/stats.json').read_text());s={k:{kk:np.array(v,np.float32) for kk,v in x.items()} for k,x in s.items()};torch.set_num_threads(2);model=torch.jit.load(str(O/'finetuned-weights/act/act-inference.pt')).eval();names=json.loads((O/'train-diagnostic/selection.json').read_text())['episodes'];rows=[]
with h5py.File(O/'new-cup.hdf5') as h:
 for ep in names:
  e=ev.inspect_episode(O/'recordings'/ep);demo=next(v for v in h['data'].values() if v.attrs['episode']==ep);a=demo['actions'][:];prev=demo['obs/previous_command'][:];rgb=demo['obs/agentview_image'][:];predicted=a.copy()
  for k in range(15,len(a),8):
   im=rgb[k].transpose(2,0,1).astype(np.float32)/255;im=(im-np.array([.485,.456,.406],np.float32)[:,None,None])/np.array([.229,.224,.225],np.float32)[:,None,None];st=(prev[k]-s['obs/previous_command']['mean'])/s['obs/previous_command']['std']
   with torch.inference_mode():pr=model(torch.from_numpy(im)[None],torch.from_numpy(st)[None]).numpy()[0,:8]*s['actions']['std']+s['actions']['mean']
   n=min(8,len(a)-k);predicted[k:k+n]=pr[:n]
  for mode,commands in [('recorded30',a),('teacher_ACT',predicted),('half_error',.5*a+.5*predicted)]:
   w=ev.World(e,False);q=ev.columns(e['frames'],'qpos',w.m.nq);v=ev.columns(e['frames'],'qvel',w.m.nv);first=int(e['indices'][15]);assert e['scene']['userdata_4'][first]==0
   def frame(j):
    w.q[:]=q[j];w.v[:]=v[j]
    for dst,prefix in [(w.u,'userdata'),(w.mp,'mocap_pos'),(w.mq,'mocap_quat')]:dst[:]=[e['scene'][f'{prefix}_{kk}'][j] for kk in range(len(dst))]
    lib.teacher_frame(w.p,w.q,w.v,w.u,w.mp,w.mq,int(e['frames']['step'][j]))
   for j in range(e['start'],first):frame(j);lib.step(w.p,np.ascontiguousarray(e['actions'][j],np.float32),10)
   frame(first);outcome=0
   try:
    for k in range(15,len(a)+90):
     n=round((k+1)*1000/30)-round(k*1000/30);outcome=w.step(np.ascontiguousarray(commands[min(k,len(a)-1)],np.float32),n)
     if outcome:break
    row={'episode':ep,'mode':mode,'success':outcome==1,'outcome':outcome,'error':lib.last_error().decode() if outcome<0 else None,'final_distance_m':float(np.linalg.norm(w.q[w.cq:w.cq+2]-w.goal[:2]))};rows.append(row);(D/'teacher-action-replay-results.json').write_text(json.dumps(rows,indent=2));print(json.dumps(row),flush=True)
   finally:w.close()
report={'protocol':'Continuation after same0.5s recorded state/control reconstruction. Recorded30 control vs ACT commands predicted on recorded RGB and recorded previous commands; half_error uses mean of recorded and predicted commands. Not autonomous success.','summary':{mode:{'successes':sum(x['success'] for x in rows if x['mode']==mode),'n':sum(x['mode']==mode for x in rows)} for mode in ['recorded30','teacher_ACT','half_error']},'errors':[r for r in rows if r['error']],'results':rows};(D/'teacher-action-replay.json').write_text(json.dumps(report,indent=2))
