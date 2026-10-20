from pathlib import Path
import sys,json,torch,numpy as np
O=Path(__file__).resolve().parent;R=O.parent;sys.path.insert(0,str(R));from train import Policy,Sequences,scheduler
rows=[];torch.set_num_threads(4)
for path in [R/'run-seed0/best.pt',R/'run-seed0/latest.pt',O/'single-run/best.pt']:
 ck=torch.load(path,map_location='cpu',weights_only=False);norm=ck['normalizer']
 for weights in (['model','ema'] if 'model' in ck else ['ema']):
  m=Policy(None,norm,load_pretrained=False).cuda().eval();m.load_state_dict(ck[weights])
  for split in ['train','valid']:
   seq=Sequences(R/'cup44-no-shadow-causal.hdf5',split,norm);names=seq.names[:4];ids=[i for i,(n,t) in enumerate(seq.index) if n in names and t in (0,30,60)];b={k:torch.stack([seq[i][k] for i in ids]).cuda() for k in ['image','command','action']}
   with torch.no_grad():
    torch.manual_seed(0);pred=m.predict(b,scheduler());truth=(b['action'][:,1:49]-m.action_offset)/m.action_scale;err=pred-truth
   row=dict(checkpoint=str(path),weights=weights,split=split,arms_deg=float(err[:,:,21:35].abs().mean()*180/np.pi),grip_mae=float(err[:,:,35:37].abs().mean()),per_window_arms_deg=(err[:,:,21:35].abs().mean(dim=(1,2))*180/np.pi).cpu().tolist());rows.append(row);print(json.dumps(row),flush=True)
   seq.handle.close()
  del m
 del ck
(O/'checkpoint-comparison.json').write_text(json.dumps(rows,indent=2))
