from pathlib import Path
import sys,json,time
import h5py,numpy as np,torch
O=Path(__file__).resolve().parent;sys.path.insert(0,str(O.parent/'cup-dp-training-20261015'))
from train import Policy,scheduler
ck=torch.load(O/'best.pt',map_location='cpu',weights_only=False);norm=ck['normalizer'];m=Policy(None,norm,load_pretrained=False).eval().to('mps');m.load_state_dict(ck['ema']);torch.set_num_threads(4)
rows=[]
with h5py.File(O.parent/'cup-dp-training-20261015/cup44-no-shadow-causal.hdf5') as h:
 for split in ['valid','train']:
  names=[n.decode() for n in h['mask/'+split][:]][:4]
  for name in names:
   d=h['data/'+name];a=d['actions'][:];prev=d['obs/previous_command'][:];ims=d['obs/agentview_image'];ep=d.attrs['episode']
   for t in [0,30,60]:
    ix=np.clip([t-1,t],0,len(a)-1);b={'image':torch.from_numpy(np.stack([ims[int(i)] for i in ix]).transpose(0,3,1,2).copy())[None].to('mps'),'command':torch.from_numpy(prev[ix]*norm['obs/previous_command']['scale']+norm['obs/previous_command']['offset'])[None].to('mps')}
    torch.manual_seed(0)
    with torch.inference_mode():pred=m.predict(b,scheduler()).cpu().numpy()[0]
    truth=a[np.minimum(np.arange(t,t+48),len(a)-1)];err=pred-truth
    def onset(x):
     ids=np.where(np.max(x[:,35:37],axis=1)>.5)[0];return float(ids[0]/30) if len(ids) else None
    row=dict(split=split,episode=ep,time=t/30,arm_mae_degrees=float(np.abs(err[:,21:35]).mean()*180/np.pi),grip_mae=float(np.abs(err[:,35:37]).mean()),gaze_mae_degrees=float(np.abs(err[:,37:39]).mean()*180/np.pi),grip_onset_prediction_s=onset(pred),grip_onset_demo_s=onset(truth))
    rows.append(row);print(json.dumps(row),flush=True)
(O/'teacher-observation-diagnosis.json').write_text(json.dumps(rows,indent=2))
