from pathlib import Path
import ctypes,json,sys,inspect,time,os
import numpy as np,torch
import random_eval as cur
O=Path(__file__).resolve().parent;D=O/'warm-act-diagnostic';D.mkdir(exist_ok=True);ev=cur.ev;lib=ctypes.CDLL(str(O/'bridge_audit.dylib'));ptr=ctypes.c_void_p;dp=np.ctypeslib.ndpointer(np.float64,flags='C_CONTIGUOUS');fp=np.ctypeslib.ndpointer(np.float32,flags='C_CONTIGUOUS')
lib.create.argtypes=[ctypes.c_char_p];lib.create.restype=ptr;lib.destroy.argtypes=[ptr];lib.last_error.restype=ctypes.c_char_p;lib.restore.argtypes=[ptr,dp,dp,dp,dp,dp];lib.read_state.argtypes=[ptr,dp,dp,dp,dp,dp];lib.step.argtypes=[ptr,fp,ctypes.c_int];lib.contacts.argtypes=[ptr];lib.teacher_frame.argtypes=[ptr,dp,dp,dp,dp,dp,ctypes.c_int];ev.lib=lib

def warm_world(e,render):
 w=ev.World(e,render);q=ev.columns(e['frames'],'qpos',w.m.nq);v=ev.columns(e['frames'],'qvel',w.m.nv);first=int(e['indices'][15]);assert e['scene']['userdata_4'][first]==0
 def frame(j):
  w.q[:]=q[j];w.v[:]=v[j]
  for dst,prefix in [(w.u,'userdata'),(w.mp,'mocap_pos'),(w.mq,'mocap_quat')]:dst[:]=[e['scene'][f'{prefix}_{k}'][j] for k in range(len(dst))]
  lib.teacher_frame(w.p,w.q,w.v,w.u,w.mp,w.mq,int(e['frames']['step'][j]))
 for j in range(e['start'],first):frame(j);lib.step(w.p,np.ascontiguousarray(e['actions'][j],np.float32),10)
 frame(first);w.gaze=e['actions'][e['indices'][14],37:39].copy();w.read();return w
sys.path.insert(0,str(O.parent/'act-cup-eval-20261015'));import evaluate_act as original
source=inspect.getsource(original.run).replace('previous=np.zeros(39,np.float32)',"previous=e['actions'][e['indices'][14]].astype(np.float32).copy()").replace('checkpoint_step=100000','checkpoint_step=20000')
namespace={'World':warm_world,'lib':lib,'np':np,'torch':torch,'time':time,'json':json,'O':D};exec(source,namespace);run=namespace['run'];torch.set_num_threads(4)
model=torch.jit.load(str(O/'finetuned-weights/act/act-inference.pt')).eval();stats=json.loads((O/'finetuned-weights/act/stats.json').read_text());names=json.loads((O/'train-diagnostic/selection.json').read_text())['episodes'];state={'phase':'running','pid':os.getpid(),'results':[],'protocol':'Teacher-forced first0.5s to reconstruct controller state, then autonomous ACT; continuation diagnostic, not full-start validation'}
def save():
 t=D/'status.tmp';t.write_text(json.dumps(state,indent=2));t.replace(D/'status.json')
try:
 for name in names:
  state['active']=name;save();e=ev.inspect_episode(O/'recordings'/name);state['results'].append(run(e,model,stats,8,90,'warm-train'));save()
 state.update(phase='completed',successes=sum(r['success'] for r in state['results']),n=len(state['results']));save()
except Exception as e:state.update(phase='failed',error=repr(e));save();raise
