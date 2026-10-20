from pathlib import Path
import json,torch,numpy as np
from train import load_data
from models import make
O=Path(__file__).resolve().parent;torch.set_num_threads(4)
meta,stats,obs,act,idx=load_data(O,'cpu');ck=torch.load(O/'run-bc-rnn/best.pt',map_location='cpu',weights_only=False);model=make('bc-rnn',169);model.load_state_dict(ck['ema']);model.eval()
oi,ai=idx['valid'];sel=torch.linspace(0,len(oi)-1,256).long();x=obs[oi[sel]];y=act[ai[sel]];std=torch.tensor(stats['action_std']);mean=torch.tensor(stats['action_mean'])
prev=x[:,-1,-37:]*torch.tensor(stats['obs_std'][-37:])+torch.tensor(stats['obs_mean'][-37:]);prev=(prev-mean)/std;prev=prev[:,None,:].expand(-1,16,-1)
def score(p):
 diff=p-y
 return {'normalized_mse':float(diff.square().mean()),'arm_mae_deg':float(np.degrees((diff*std).abs().mean((0,1))[21:35]).mean()),'grip_mae':float((diff*std).abs().mean((0,1))[35:].mean()),'first4_arm_mae_deg':float(np.degrees((diff[:,:4]*std).abs().mean((0,1))[21:35]).mean())}
with torch.inference_mode():
 p=model(x);r={'learned':score(p),'repeat_previous_command':score(prev)}
 for name,ix in [('cup_goal',list(range(96,114))),('previous_command',list(range(132,169)))]:
  xx=x.clone();xx[:,:,ix]=x.flip(0)[:,:,ix];pp=model(xx);r['shuffle_'+name]=score(pp);r['shuffle_'+name]['prediction_arm_change_deg']=float(np.degrees(((pp-p)*std).abs().mean((0,1))[21:35]).mean())
(O/'prediction-diagnosis.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
