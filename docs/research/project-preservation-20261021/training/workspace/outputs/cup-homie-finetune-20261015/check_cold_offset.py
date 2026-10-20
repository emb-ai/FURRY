from pathlib import Path
import ctypes,json,time
import numpy as np
import random_eval as cur
O=Path(__file__).resolve().parent;ev=cur.ev;D=O/'restart-diagnostic';lib=ctypes.CDLL(str(O/'bridge_audit.dylib'));ptr=ctypes.c_void_p;dp=np.ctypeslib.ndpointer(np.float64,flags='C_CONTIGUOUS');fp=np.ctypeslib.ndpointer(np.float32,flags='C_CONTIGUOUS')
lib.create.argtypes=[ctypes.c_char_p];lib.create.restype=ptr;lib.destroy.argtypes=[ptr];lib.last_error.restype=ctypes.c_char_p;lib.restore.argtypes=[ptr,dp,dp,dp,dp,dp];lib.read_state.argtypes=[ptr,dp,dp,dp,dp,dp];lib.step.argtypes=[ptr,fp,ctypes.c_int];lib.set_phase.argtypes=[ptr,ctypes.c_int];ev.lib=lib;rows=[]
for name in json.loads((O/'train-diagnostic/selection.json').read_text())['episodes']:
 e=ev.inspect_episode(O/'recordings'/name);e['start']+=50;w=ev.World(e,False);lib.set_phase(w.p,int(e['frames']['step'][e['start']]));outcome=0
 try:
  for j in range(e['start'],e['end']+301):
   outcome=w.step(np.ascontiguousarray(e['actions'][min(j,e['end'])],np.float32),10)
   if outcome:break
  row={'episode':name,'success':outcome==1,'outcome':outcome};rows.append(row);print(json.dumps(row),flush=True);(D/'cold-offset-results.json').write_text(json.dumps(rows,indent=2))
 finally:w.close()
(D/'cold-offset-summary.json').write_text(json.dumps({'successes':sum(r['success'] for r in rows),'n':len(rows)},indent=2))
