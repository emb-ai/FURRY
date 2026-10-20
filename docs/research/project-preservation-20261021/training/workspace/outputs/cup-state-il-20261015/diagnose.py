from pathlib import Path
import json,numpy as np,mujoco,torch
from state import Encoder
O=Path(__file__).resolve().parent
m=mujoco.MjModel.from_binary_path(str(O/'evaluation-episode/learning-model.mjb'));d=mujoco.MjData(m);e=Encoder(m)
meta=json.loads((O/'dataset.json').read_text());z=np.load(O/'dataset.npz');ck=torch.load(O/'run-bc-rnn/best.pt',map_location='cpu',weights_only=False);stats=ck['stats']
obs=np.concatenate([z[r['key']+'_obs'] for r in meta['episodes'] if r['split']=='train']);act=np.concatenate([z[r['key']+'_action'] for r in meta['episodes'] if r['split']=='train'])
starts=np.stack([z[r['key']+'_obs'][0] for r in meta['episodes'] if r['split']=='train'])
phys=132
norm=(obs[:,:phys]-stats['obs_mean'][:phys])/stats['obs_std'][:phys]
def near(x):
 v=((x[:phys]-stats['obs_mean'][:phys])/stats['obs_std'][:phys]);dist=np.mean((norm-v)**2,axis=1);j=np.argmin(dist)
 return {'rms_z':float(np.sqrt(dist[j])),'arm_error_deg':float(np.degrees(np.abs(x[15:29]-obs[j,15:29])).mean()),'max_z_features':sorted([(meta['observation_names'][i],float(abs(v[i]))) for i in range(phys)],key=lambda t:-t[1])[:5]}
rows=[]
for f in sorted((O/'run-bc-rnn/fresh-evaluation').glob('*.npz')):
 try:
  v=np.load(f);q=v['qpos'];trace=v['trace'];a=v['actions']
 except Exception:continue
 d.qpos[:]=v['initial_q'];d.qvel[:]=v['initial_v'];d.mocap_pos.flat[:]=v['initial_mp'];d.mocap_quat.flat[:]=v['initial_mq'];initial=e(d,np.zeros(37))
 distances=[]
 for i in range(0,len(q),5):
  d.qpos[:]=q[i];mujoco.mj_kinematics(m,d);distances.append([np.linalg.norm(d.xpos[w]-d.xpos[e.cup]) for w in e.wrists])
 tilt=trace[:,5];contacts=trace[:,6]
 rows.append({'seed':f.stem,'min_wrist_cup_m':np.min(distances,axis=0).tolist(),'first_contact_s':float(trace[np.flatnonzero(contacts>0)[0],0]) if (contacts>0).any() else None,'contact_fraction':float((contacts>0).mean()),'grip_min':a[:,35:].min(0).tolist(),'grip_max':a[:,35:].max(0).tolist(),'grip_mean':a[:,35:].mean(0).tolist(),'first_tilt45_s':float(trace[np.flatnonzero(tilt>45)[0],0]) if (tilt>45).any() else None,'lift_flag':float(trace[:,7+4].max()),'initial_nearest_train':near(initial),'initial_cup_relative':initial[96:99].tolist(),'first_action':a[0].tolist()})
held=[]
for r in meta['episodes']:
 if r['split']=='valid':held.append({'episode':r['episode'],'nearest_train':near(z[r['key']+'_obs'][0])})
report={'rollouts':rows,'demo_grip_quantiles':np.quantile(act[:,35:],[0,.1,.5,.9,1],axis=0).tolist(),'heldout_starts':held,'demo_start_time_range':[float(min(z[r['key']+'_time'][0] for r in meta['episodes'])),float(max(z[r['key']+'_time'][0] for r in meta['episodes']))],'checkpoint_step':ck['step']}
(O/'diagnosis.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report))
