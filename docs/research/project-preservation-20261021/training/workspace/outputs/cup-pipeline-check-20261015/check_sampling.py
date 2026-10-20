from pathlib import Path
import sys,json
import numpy as np,torch,h5py
O=Path(__file__).resolve().parent;T=O.parent/'cup-dp-training-20261015';sys.path.insert(0,str(T));from train import Policy,scheduler,Sequences
from diffusers import DDPMScheduler
ck=torch.load(O.parent/'cup-dp-eval-20261015/best.pt',map_location='cpu',weights_only=False);norm=ck['normalizer'];m=Policy(None,norm,load_pretrained=False).eval().to('mps');m.load_state_dict(ck['ema']);torch.set_num_threads(4)
d=Sequences(T/'cup44-no-shadow-causal.hdf5','valid',norm);selection=[i for i,(n,t) in enumerate(d.index) if t in (0,30,60)];samples=[d[i] for i in selection];b={k:torch.stack([s[k] for s in samples]).to('mps') for k in samples[0]}
truth=b['action'][:,1:49];result=[]
with torch.inference_mode():
 cond=m.encode(b)
 for kind,steps in [('leading',10),('leading',50),('trailing',10),('ddpm',50)]:
  s=DDPMScheduler.from_config(scheduler().config) if kind=='ddpm' else scheduler();
  if kind=='trailing':s.register_to_config(timestep_spacing='trailing')
  s.set_timesteps(steps,device='mps');torch.manual_seed(0);x=torch.randn((len(samples),64,39),device='mps')
  for t in s.timesteps:x=s.step(m.unet(x,t,global_cond=cond),t,x).prev_sample
  err=(x[:,1:49]-truth)/m.action_scale
  row=dict(sampler=kind,steps=steps,arms_deg=float(err[:,:,21:35].abs().mean()*180/np.pi),grip_mae=float(err[:,:,35:37].abs().mean()),normalized_mse=float((x[:,1:49]-truth).square().mean()))
  print(row,flush=True);result.append(row)
(O/'sampling-check.json').write_text(json.dumps(result,indent=2))
