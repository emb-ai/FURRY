from pathlib import Path
import json,sys,hashlib
import numpy as np
import mujoco
O=Path(__file__).resolve().parent
sys.path.insert(0,str(O.parent/'cup-dp-training-20261015'))
from capture_snapshot import inspect_episode,table,columns
rows=[]
for name in json.loads((O/'selected-source-episodes.json').read_text()):
 p=O/'recordings'/name;row={'episode':name,'reasons':[]}
 try:
  e=inspect_episode(p);m=mujoco.MjModel.from_binary_path(str(p/'learning-model.mjb'));q=columns(e['frames'],'qpos',m.nq);sl=slice(e['start'],e['end']+1);t=e['frames']['sim_time'][sl];q=q[sl];adr=int(m.joint('cup_free').qposadr[0]);quat=q[:,adr+3:adr+7];tilt=np.degrees(np.arccos(np.clip(1-2*(quat[:,1]**2+quat[:,2]**2),-1,1)))
  root=np.degrees(np.arccos(np.clip(1-2*(q[:,4]**2+q[:,5]**2),-1,1)));dt=np.r_[np.diff(t),0]
  side_s=float(dt[tilt>=70].sum());final_tilt=float(np.max(tilt[t>=t[-1]-.5]));maxlift=float(q[:,adr+2].max()-q[0,adr+2])
  if side_s>=.15:row['reasons'].append('cup_tilt_ge70_for_ge150ms')
  if final_tilt>30:row['reasons'].append('final_cup_tilt_gt30deg')
  if q[:,2].min()<.35 or root.max()>45:row['reasons'].append('robot_unstable')
  row.update(duration_s=float(t[-1]-t[0]),frames=len(t),samples=len(e['indices']),max_cup_tilt_deg=float(tilt.max()),cup_side_s=side_s,final_cup_tilt_deg=final_tilt,max_lift_m=maxlift,min_root_z=float(q[:,2].min()),max_root_tilt_deg=float(root.max()),worst_frame=int(e['start']+np.argmax(tilt)),family=e['family'],identity=e['identity'],max_abs_velocity_command=np.max(np.abs(e['actions'][e['indices']][:,[0,1,5]]),axis=0).tolist())
 except Exception as exc:row['reasons'].append('integrity_or_capture:'+str(exc))
 row['accepted']=not row['reasons'];rows.append(row);print(json.dumps(row),flush=True)
report={'criteria':{'sideways':'cup axis tilt >=70 degrees for >=0.15 seconds total','final':'cup tilt <=30 degrees during final 0.5 seconds','robot':'pelvis height >=0.35m and tilt <=45 degrees','integrity':'existing strict source stream, reset, gap, success and model-hash validation'},'accepted':[r['episode'] for r in rows if r['accepted']],'rejected':[r['episode'] for r in rows if not r['accepted']],'rows':rows}
(O/'quality-audit.json').write_text(json.dumps(report,indent=2));print('ACCEPTED',len(report['accepted']),'REJECTED',len(report['rejected']),flush=True)
