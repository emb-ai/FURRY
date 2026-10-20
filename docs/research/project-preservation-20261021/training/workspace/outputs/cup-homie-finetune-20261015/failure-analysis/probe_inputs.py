from pathlib import Path
import json,time
import h5py,numpy as np,torch
O=Path(__file__).resolve().parents[1];D=Path(__file__).resolve().parent;torch.set_num_threads(4)
s=json.loads((O/'finetuned-weights/act/stats.json').read_text());stats={k:{kk:np.array(v,np.float32) for kk,v in x.items()} for k,x in s.items()}
model=torch.jit.load(str(O/'finetuned-weights/act/act-inference.pt')).eval()
phases=[0,.1,.25,.5,.75];images=[];states=[];targets=[];meta=[]
with h5py.File(O/'new-cup.hdf5') as h:
 names=[v.decode() for v in h['mask/train'][:]]
 for name in names:
  d=h['data'][name];a=d['actions'][:];prev=d['obs/previous_command'][:]
  for phase in phases:
   k=int((len(a)-1)*phase);images.append(d['obs/agentview_image'][k]);states.append(prev[k]);targets.append(a[np.minimum(k+np.arange(8),len(a)-1)]);meta.append({'episode':str(d.attrs['episode']),'k':k,'phase':phase})
raw=np.array(images);state=np.array(states);target=np.array(targets);im=raw.transpose(0,3,1,2).astype(np.float32)/255;im=(im-np.array([.485,.456,.406],np.float32)[None,:,None,None])/np.array([.229,.224,.225],np.float32)[None,:,None,None];st=(state-stats['obs/previous_command']['mean'])/stats['obs/previous_command']['std']
perm=np.roll(np.arange(len(meta)).reshape(-1,len(phases)),1,axis=0).ravel();preds={}
def predict(x,y):
 out=[]
 with torch.inference_mode():
  for k in range(0,len(x),8):out.append(model(torch.from_numpy(x[k:k+8].copy()),torch.from_numpy(y[k:k+8].copy())).numpy()[:,:8])
 return np.concatenate(out)*stats['actions']['std']+stats['actions']['mean']
for name,x,y in [('normal',im,st),('image_other_episode_same_phase',im[perm],st),('image_mean',np.zeros_like(im),st),('state_other_episode_same_phase',im,st[perm])]:
 preds[name]=predict(x,y);print(name,flush=True)
preds['repeat_previous_command']=np.repeat(state[:,None,:],8,axis=1)
def metrics(p,t):
 err=abs(p-t);return {'arms_deg':float(np.rad2deg(err[:,:,21:35]).mean()),'grip':float(err[:,:,35:37].mean()),'nav_xy':float(err[:,:,:2].mean()),'gaze_deg':float(np.rad2deg(err[:,:,37:39]).mean())}
report={'n_frames':len(meta),'episodes':len(names),'phases':phases,'metrics':{},'by_phase':{},'sensitivity':{}}
for key,p in preds.items():report['metrics'][key]=metrics(p,target)
for phase in phases:
 ids=np.array([m['phase']==phase for m in meta]);report['by_phase'][str(phase)]={k:metrics(p[ids],target[ids]) for k,p in preds.items()}
for key,p in preds.items():report['sensitivity'][key]=metrics(p,preds['normal'])
np.savez_compressed(D/'input-probe.npz',images=raw,states=state,targets=target,**preds)
(D/'input-probe-frames.json').write_text(json.dumps(meta,indent=2));(D/'input-probe.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
