"""Continuation diagnostic after recorded prefix; never a full-reset success metric."""
from pathlib import Path
import argparse,ctypes,json,time
import numpy as np,h5py,torch,onnxruntime
from proprio import features
O=Path(__file__).resolve().parent;p=argparse.ArgumentParser();p.add_argument('--weights',type=Path);p.add_argument('--out',type=Path,required=True);args=p.parse_args();args.out.mkdir(exist_ok=False,parents=True);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
ctypes.CDLL(str(next((Path(onnxruntime.__file__).parent/'capi').glob('libonnxruntime.so.*'))),mode=ctypes.RTLD_GLOBAL);lib=ctypes.CDLL(str(O/'bridge.so'));ptr=ctypes.c_void_p;dp=np.ctypeslib.ndpointer(np.float64,flags='C_CONTIGUOUS');fp=np.ctypeslib.ndpointer(np.float32,flags='C_CONTIGUOUS')
lib.create.argtypes=[ctypes.c_char_p];lib.create.restype=ptr;lib.destroy.argtypes=[ptr];lib.last_error.restype=ctypes.c_char_p;lib.step.argtypes=[ptr,fp,ctypes.c_int];lib.teacher_frame.argtypes=[ptr,dp,dp,dp,dp,dp,ctypes.c_int];lib.read_state.argtypes=[ptr,dp,dp,dp,dp,dp]
if args.weights:
 model=torch.jit.load(str(args.weights/'act-inference.pt'),map_location='cuda').eval();stats={k:{kk:np.array(v,np.float32) for kk,v in s.items()} for k,s in json.loads((args.weights/'stats.json').read_text()).items()}
selected=json.loads((O/'train-selection.json').read_text())['episodes'];manifest=json.loads((O/'render-input/manifest.json').read_text());rows=[]
with h5py.File(O/'cup-native50.hdf5') as h:
 for ep in selected:
  key=next(r['key'] for r in manifest['episodes'] if r['episode']==ep);z=np.load(O/'render-input'/(key+'.npz'));a=z['actions'];pred=a.copy();first=25
  if args.weights:
   for i in range(first,len(a),8):
    rgb=h[f'data/{key}/obs/agentview_image'][i];im=rgb.transpose(2,0,1).astype(np.float32)/255;im=(im-np.array([.485,.456,.406],np.float32)[:,None,None])/np.array([.229,.224,.225],np.float32)[:,None,None];s=stats['obs/previous_command'];x=(z['previous_command'][i]-s['mean'])/s['std']
    if 'obs/proprio' in stats:x=np.concatenate([x,(features(z['qpos'][i],z['qvel'][i])-stats['obs/proprio']['mean'])/stats['obs/proprio']['std']])
    with torch.inference_mode():pr=model(torch.from_numpy(im)[None].cuda(),torch.from_numpy(x)[None].cuda()).cpu().numpy()[0,:8]*stats['actions']['std']+stats['actions']['mean']
    n=min(8,len(a)-i);pred[i:i+n]=pr[:n]
  w=lib.create(str(O/'episode').encode());assert w,lib.last_error().decode();q=np.zeros(57);v=np.zeros(55);u=np.zeros(8);mp=np.zeros(3);mq=np.zeros(4)
  def frame(i):
   lib.teacher_frame(w,*[np.ascontiguousarray(z[k][i],np.float64) for k in ['qpos','qvel','userdata','mocap_pos','mocap_quat']],int(z['source_step'][i]))
  start=time.time();assert z['userdata'][first,3]==0 and z['userdata'][first,4]==0
  try:
   for i in range(first):frame(i);assert lib.step(w,np.ascontiguousarray(a[i],np.float32),20)>=0
   frame(first);outcome=0
   for i in range(first,len(a)+150):
    outcome=lib.step(w,np.ascontiguousarray(pred[min(i,len(a)-1)],np.float32),20)
    if outcome:break
   lib.read_state(w,q,v,u,mp,mq);row={'episode':ep,'success':outcome==1,'outcome':outcome,'wall_s':time.time()-start,'arms_mae_deg':float(np.rad2deg(np.abs(pred[first:,21:35]-a[first:,21:35])).mean()),'final_distance_m':float(np.linalg.norm(q[50:52]-mp[:2])),'error':lib.last_error().decode() if outcome<0 else None};rows.append(row);(args.out/'results.json').write_text(json.dumps(rows,indent=2));print(json.dumps(row),flush=True)
  finally:lib.destroy(w)
s={'horizon':'recorded trajectory duration plus3s holding final command','protocol':'CONTINUATION after0.5s recorded state/control prefix; recorded observations for neural predictions; not full-reset policy performance','weights':str(args.weights),'successes':sum(x['success'] for x in rows),'n':len(rows),'errors':[x['error'] for x in rows if x['error']],'results':rows};(args.out/'summary.json').write_text(json.dumps(s,indent=2));print(json.dumps({k:v for k,v in s.items() if k!='results'}),flush=True)
