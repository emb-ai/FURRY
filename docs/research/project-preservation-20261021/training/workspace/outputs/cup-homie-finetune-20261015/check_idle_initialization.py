from pathlib import Path
import ctypes,json,numpy as np
import random_eval as cur
O=Path(__file__).resolve().parent;ev=cur.ev;lib=cur.lib;lib.idle.argtypes=[ctypes.c_void_p,ctypes.c_int];rows=[]
for name in ['episode-1792058834151648','episode-1792058974761240']:
 e=ev.inspect_episode(O/'recordings'/name);w=ev.World(e,False)
 try:
  expected=w.q.copy();idx=0
  w.q[:]=ev.columns(e['frames'],'qpos',w.m.nq)[idx];w.v[:]=ev.columns(e['frames'],'qvel',w.m.nv)[idx]
  for dst,prefix in [(w.u,'userdata'),(w.mp,'mocap_pos'),(w.mq,'mocap_quat')]:dst[:]=[e['scene'][f'{prefix}_{j}'][idx] for j in range(len(dst))]
  lib.restore(w.p,w.q,w.v,w.u,w.mp,w.mq);steps=int(e['frames']['step'][e['start']]);result=lib.idle(w.p,steps);w.read()
  row={'episode':name,'idle_steps':steps,'result':result,'qpos_max_abs_error':float(np.max(np.abs(w.q-expected))),'root_error_m':float(np.linalg.norm(w.q[:3]-expected[:3])),'cup_error_m':float(np.linalg.norm(w.q[w.cq:w.cq+3]-expected[w.cq:w.cq+3]))};rows.append(row);print(json.dumps(row),flush=True)
 finally:w.close()
(O/'train-diagnostic/idle-initialization-check.json').write_text(json.dumps(rows,indent=2))
