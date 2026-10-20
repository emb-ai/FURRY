"""Recorded commands from genuine native resets; dataset diagnostic, not learned-policy eval."""
from pathlib import Path
import ctypes,json,time
import numpy as np,onnxruntime,mujoco
O=Path(__file__).resolve().parent;D=O/'diagnostic-recorded-cold';D.mkdir(exist_ok=False);ctypes.CDLL(str(next((Path(onnxruntime.__file__).parent/'capi').glob('libonnxruntime.so.*'))),mode=ctypes.RTLD_GLOBAL);lib=ctypes.CDLL(str(O/'bridge.so'));ptr=ctypes.c_void_p;dp=np.ctypeslib.ndpointer(np.float64,flags='C_CONTIGUOUS');fp=np.ctypeslib.ndpointer(np.float32,flags='C_CONTIGUOUS');lib.create.argtypes=[ctypes.c_char_p];lib.create.restype=ptr;lib.destroy.argtypes=[ptr];lib.last_error.restype=ctypes.c_char_p;lib.step.argtypes=[ptr,fp,ctypes.c_int];lib.reset_random.argtypes=[ptr,ctypes.c_uint];lib.fresh_reset_flags.argtypes=[ptr];lib.read_state.argtypes=[ptr,dp,dp,dp,dp,dp];rows=[]
selected=json.loads((O/'train-selection.json').read_text())['episodes'];manifest=json.loads((O/'render-input/manifest.json').read_text())
for ep in selected:
 key=next(r['key'] for r in manifest['episodes'] if r['episode']==ep);z=np.load(O/'render-input'/(key+'.npz'));a=z['actions'];w=lib.create(str(O/'episode').encode());assert w;seed=int(z['userdata'][0,0]);assert lib.reset_random(w,seed)==0 and lib.fresh_reset_flags(w)==63;q=np.zeros(57);v=np.zeros(55);u=np.zeros(8);mp=np.zeros(3);mq=np.zeros(4);lib.read_state(w,q,v,u,mp,mq);np.testing.assert_allclose(mp,z['mocap_pos'][0],atol=1e-12);np.testing.assert_allclose(q[50:52],z['qpos'][0,50:52],atol=1e-5);start=time.time()
 try:
  for i in range(len(a)+150):
   outcome=lib.step(w,np.ascontiguousarray(a[min(i,len(a)-1)],np.float32),20)
   if outcome:break
  lib.read_state(w,q,v,u,mp,mq);r={'episode':ep,'success':outcome==1,'outcome':outcome,'wall_s':time.time()-start,'error':lib.last_error().decode() if outcome<0 else None,'final_distance_m':float(np.linalg.norm(q[50:52]-mp[:2]))};rows.append(r);(D/'results.json').write_text(json.dumps(rows,indent=2));print(json.dumps(r),flush=True)
 finally:lib.destroy(w)
s={'horizon':'recorded trajectory duration plus3s holding final command','protocol':'Original50Hz recorded commands immediately after full native reset on matching original8 scenes; no saved robot pose, no prefix; not learned-policy success','n':len(rows),'successes':sum(x['success'] for x in rows),'errors':[x['error'] for x in rows if x['error']],'results':rows};(D/'summary.json').write_text(json.dumps(s,indent=2));print(json.dumps({k:v for k,v in s.items() if k!='results'}),flush=True)
