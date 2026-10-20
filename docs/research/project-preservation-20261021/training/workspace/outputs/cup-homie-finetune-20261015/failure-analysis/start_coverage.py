from pathlib import Path
import csv,json
import numpy as np,h5py,mujoco as mj
O=Path(__file__).resolve().parents[1];D=Path(__file__).resolve().parent;cases=json.loads((O/'fresh-start/scenes.json').read_text())['cases'];q0=np.array(cases[0]['qpos']);rows=[]
def readrow(path,i):
 with path.open() as f:
  for k,row in enumerate(csv.DictReader(f)):
   if k==i:return {n:float(v) for n,v in row.items()}
with h5py.File(O/'new-cup.hdf5') as h:
 for demo in h['data'].values():
  name=demo.attrs['episode'];folder=O/'recordings'/name;i=int(demo['source_frame_index'][0]);f=readrow(folder/'frames.csv',i);sc=readrow(folder/'scene_state.csv',i)
  if not rows:
   m=mj.MjModel.from_binary_path(str(folder/'learning-model.mjb'));cq=int(m.joint('cup_free').qposadr[0]);names=[f'{s}_{p}_joint' for s in ['left','right'] for p in ['shoulder_pitch','shoulder_roll','shoulder_yaw','elbow','wrist_roll','wrist_pitch','wrist_yaw']];adr=np.array([m.joint(n).qposadr[0] for n in names])
  q=np.array([f['qpos_'+str(k)] for k in range(m.nq)]);cup=q[cq:cq+3];goal=np.array([sc['mocap_pos_'+str(k)] for k in range(3)])
  rows.append({'episode':name,'first_frame':i,'sim_time':f['sim_time'],'root_xyz':q[:3].tolist(),'root_xy_from_reset_m':float(np.linalg.norm(q[:2]-q0[:2])),'arms_mae_from_reset_deg':float(np.rad2deg(abs(q[adr]-q0[adr])).mean()),'cup_xyz':cup.tolist(),'goal_xyz':goal.tolist(),'duration':float(demo['timestamps'][-1]-demo['timestamps'][0])})
report={'n':len(rows),'quantiles':{},'train_cup_xy_bbox':np.array([r['cup_xyz'][:2] for r in rows]).min(0).tolist()+np.array([r['cup_xyz'][:2] for r in rows]).max(0).tolist(),'trials':rows}
for key in ['sim_time','root_xy_from_reset_m','arms_mae_from_reset_deg','duration']:report['quantiles'][key]=np.quantile([r[key] for r in rows],[0,.5,1]).tolist()
report['test_cup_xy']=[np.array(c['qpos'])[cq:cq+2].tolist() for c in cases]
(D/'start-coverage.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='trials'},indent=2))
