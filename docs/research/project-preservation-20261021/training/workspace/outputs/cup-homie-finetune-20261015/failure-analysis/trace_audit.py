from pathlib import Path
import json,numpy as np,h5py,mujoco as mj
O=Path(__file__).resolve().parents[1];D=Path(__file__).resolve().parent
name=json.loads((O/'quality-audit.json').read_text())['accepted'][0];m=mj.MjModel.from_binary_path(str(O/'recordings'/name/'learning-model.mjb'));d=mj.MjData(m)
# Body action dimensions21:35 correspond to14 arm joints, actuator body indices15:29.
names=[f'{side}_{part}_joint' for side in ['left','right'] for part in ['shoulder_pitch','shoulder_roll','shoulder_yaw','elbow','wrist_roll','wrist_pitch','wrist_yaw']]
qadr=np.array([m.joint(n).qposadr[0] for n in names])
report={'trials':[]}
for row in json.loads((O/'fresh-start/finetuned/act-summary.json').read_text())['results']:
 z=np.load(O/'fresh-start/finetuned'/f"act-{row['seed']}.npz");a=z['actions'];q=z['qpos'];t=z['trace'][:,0];late=t>max(0,t[-1]-10)
 rr={'seed':row['seed'],'success':row['success'],'duration':float(t[-1]),'arms_tracking_mae_deg':float(np.rad2deg(abs(a[:,21:35]-q[:,qadr])).mean()),'late_command_arm_std_deg':float(np.rad2deg(a[late,21:35].std(0)).mean()),'late_gaze_deg':np.rad2deg(a[late,37:39].mean(0)).tolist(),'late_grip_mean':a[late,35:37].mean(0).tolist(),'root_displacement_m':float(np.linalg.norm(q[-1,:2]-q[0,:2])),'max_lift_m':row['lift_m'],'final_distance_m':row['goal_xy_error_m']}
 report['trials'].append(rr)
with h5py.File(O/'new-cup.hdf5') as h:
 a=np.concatenate([x['actions'][:] for x in h['data'].values()]);first=np.array([x['actions'][0] for x in h['data'].values()]);report['train_gaze_deg_quantiles']=np.rad2deg(np.quantile(a[:,37:39],[0,.1,.5,.9,1],axis=0)).tolist();report['first_train_gaze_deg_quantiles']=np.rad2deg(np.quantile(first[:,37:39],[0,.1,.5,.9,1],axis=0)).tolist()
(D/'trace-audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
