from pathlib import Path
import json,sys
import h5py,numpy as np,torch
import random_eval as current
O=Path(__file__).resolve().parent;ev=current.ev;D=O/'train-diagnostic'
names=json.loads((D/'selection.json').read_text())['episodes'];rows=[]
models={label:torch.jit.load(str(path)).eval() for label,path in [('before',O.parent/'act-cup-eval-20261015/act-inference.pt'),('after',O/'finetuned-weights/act/act-inference.pt')]}
stats=json.loads((O/'finetuned-weights/act/stats.json').read_text());stats={k:{kk:np.array(vv,np.float32) for kk,vv in v.items()} for k,v in stats.items()};torch.set_num_threads(4)
with h5py.File(O/'new-cup.hdf5','r') as h:
 for name in names:
  d=next(d for d in h['data'].values() if d.attrs['episode']==name);e=ev.inspect_episode(O/'recordings'/name);a=d['actions'][:];prev=d['obs/previous_command'][:];assert np.array_equal(a,e['actions'][e['indices']].astype(np.float32));w=ev.World(e,True)
  row={'episode':name,'start_frame':int(e['start']),'frames':len(a),'robot_start_xyz':w.q[:3].tolist(),'cup_start_xyz':w.start.tolist(),'gaze_start':w.gaze.tolist(),'checks':[]}
  try:
   for k in [0,len(a)//4,len(a)//2]:
    if k:
     idx=e['indices'][k];w.q[:]=ev.columns(e['frames'],'qpos',w.m.nq)[idx];w.v[:]=ev.columns(e['frames'],'qvel',w.m.nv)[idx]
     for dst,prefix in [(w.u,'userdata'),(w.mp,'mocap_pos'),(w.mq,'mocap_quat')]:dst[:]=[e['scene'][f'{prefix}_{j}'][idx] for j in range(len(dst))]
     current.lib.restore(w.p,w.q,w.v,w.u,w.mp,w.mq);w.gaze=prev[k,37:39].copy()
    image=w.image();error=np.abs(image.astype(float)-d['obs/agentview_image'][k].astype(float));c={'frame':k,'rgb_mae':float(error.mean()),'rgb_max_abs':float(error.max())}
    im=image.transpose(2,0,1).astype(np.float32)/255;im=(im-np.array([.485,.456,.406],np.float32)[:,None,None])/np.array([.229,.224,.225],np.float32)[:,None,None];s=stats['obs/previous_command'];st=(prev[k]-s['mean'])/s['std'];target=a[np.minimum(np.arange(k,k+8),len(a)-1)]
    for label,model in models.items():
     with torch.inference_mode():pred=model(torch.from_numpy(im)[None],torch.from_numpy(st)[None]).numpy()[0][:8]*stats['actions']['std']+stats['actions']['mean']
     err=np.abs(pred-target);c[label]={'arms_mae_deg':float(np.rad2deg(err[:,21:35]).mean()),'grip_mae':float(err[:,35:37].mean()),'gaze_mae_deg':float(np.rad2deg(err[:,37:39]).mean()),'navigation_mae':err[:,[0,1,5]].mean(0).tolist()}
    row['checks'].append(c)
  finally:w.close()
  rows.append(row);print(json.dumps(row),flush=True);(D/'observation-action-audit.json').write_text(json.dumps({'episodes':rows,'complete':False},indent=2))
(D/'observation-action-audit.json').write_text(json.dumps({'episodes':rows,'complete':True},indent=2))
