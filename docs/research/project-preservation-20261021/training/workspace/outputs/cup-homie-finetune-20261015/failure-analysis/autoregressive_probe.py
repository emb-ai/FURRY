from pathlib import Path
import json
import numpy as np,h5py,torch
O=Path(__file__).resolve().parents[1];D=Path(__file__).resolve().parent;torch.set_num_threads(2)
s=json.loads((O/'finetuned-weights/act/stats.json').read_text());s={k:{kk:np.array(v,np.float32) for kk,v in x.items()} for k,x in s.items()};model=torch.jit.load(str(O/'finetuned-weights/act/act-inference.pt')).eval();names=json.loads((O/'train-diagnostic/selection.json').read_text())['episodes'];rows=[]
with h5py.File(O/'new-cup.hdf5') as h:
 for ep in names:
  demo=next(v for v in h['data'].values() if v.attrs['episode']==ep);a=demo['actions'][:];prev=demo['obs/previous_command'][:];images=demo['obs/agentview_image'][:];selfprev=np.zeros(39,np.float32);errors=[];errs_oracle=[];times=[]
  for k in range(0,len(a),8):
   im=images[k].transpose(2,0,1).astype(np.float32)/255;im=(im-np.array([.485,.456,.406],np.float32)[:,None,None])/np.array([.229,.224,.225],np.float32)[:,None,None]
   st=(np.stack([prev[k],selfprev])-s['obs/previous_command']['mean'])/s['obs/previous_command']['std']
   with torch.inference_mode():pred=model(torch.from_numpy(np.stack([im,im])),torch.from_numpy(st)).numpy()[:,:8]*s['actions']['std']+s['actions']['mean']
   n=min(8,len(a)-k);errors.append(float(np.rad2deg(abs(pred[1,:n,21:35]-a[k:k+n,21:35])).mean()));errs_oracle.append(float(np.rad2deg(abs(pred[0,:n,21:35]-a[k:k+n,21:35])).mean()));selfprev=pred[1,n-1].copy();times.append(k/30)
  rows.append({'episode':ep,'times':times,'own_previous_command_error_deg':errors,'recorded_previous_command_error_deg':errs_oracle});print(ep,round(np.mean(errors),2),round(np.mean(errs_oracle),2),flush=True)
r={'protocol':'Recorded RGB in both branches. Only previous-command input differs: recorded vs recursively predicted. No physics, not task success.','episodes':rows,'mean_own_error_deg':float(np.mean([x for r in rows for x in r['own_previous_command_error_deg']])),'mean_recorded_error_deg':float(np.mean([x for r in rows for x in r['recorded_previous_command_error_deg']]))};(D/'autoregressive-probe.json').write_text(json.dumps(r,indent=2));print(json.dumps({k:v for k,v in r.items() if k!='episodes'}))
