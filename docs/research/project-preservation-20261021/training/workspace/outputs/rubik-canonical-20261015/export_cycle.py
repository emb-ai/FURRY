from pathlib import Path
import numpy as np,mujoco,json,hashlib
from physical_check import check
p=Path(__file__).resolve().parent;z=np.load(p/'macro-follow07.npz');n=2797;subset={k:z[k][:n] for k in z.files};np.savez_compressed(p/'single-cycle.npz',**subset)
headers=['time_s']+[f'{side}_{a}' for side in ['left','right'] for a in ['x_m','y_m','z_m','qw','qx','qy','qz']]+['left_grip','right_grip']
np.savetxt(p/'single-cycle-wrist-commands.csv',np.c_[subset['trace'][:,0],subset['commands']],delimiter=',',header=','.join(headers),comments='',fmt='%.12g')
m=mujoco.MjModel.from_xml_path(str(p.parent/'rubik-sequences-20261015/project/android/assets/scene-rubik.xml'));d=mujoco.MjData(m);hands=[];cube=[]
for g in range(m.ngeom):
 if m.geom_contype[g]==0:continue
 name=m.body(int(m.geom_bodyid[g])).name
 if name.startswith('right_hand_') or name=='right_wrist_yaw_link':hands.append(g)
 if name.startswith('rubik_cubie_'):cube.append(g)
t=subset['trace'];mask=(t[:,0]>14.97)&(t[:,13]==0)&(subset['commands'][:,-1]<.1);starts=np.where(np.diff(np.r_[False,mask,False].astype(int))==1)[0];ends=np.where(np.diff(np.r_[False,mask,False].astype(int))==-1)[0];longest=int(max(ends-starts,default=0))
clearance=[]
for k in np.flatnonzero(mask)[::10]:
 d.qpos[:]=subset['qpos'][k];mujoco.mj_kinematics(m,d);gap=min(mujoco.mj_geomDistance(m,d,a,b,.3,None) for a in hands for b in cube);clearance.append([float(t[k,0]),gap])
np.savetxt(p/'single-cycle-right-clearance.csv',clearance,delimiter=',',header='time_s,min_geometry_gap_m',comments='')
r=dict(status='single_cycle_only_arbitrary_sequences_not_achieved',policy='HOMIE Standing',environment='Unmodified installed Rubik model and native physics; desktop offline simulation.',scene_sha256=hashlib.sha256((p.parent/'rubik-sequences-20261015/project/android/assets/scene-rubik.xml').read_bytes()).hexdigest(),warmup_seconds=2.5,command_hz=100,physics_hz=1000,policy_hz=50,seed=0,initial_pickup_end_s=11.,cycle_end_s=27.97,cycle_metrics=check(subset,1099,n-1),right_contact_free_longest_s=longest/100,right_max_geometry_clearance_m=max((v[1] for v in clearance),default=0),non_hand_support_frames_during_cycle=int((subset['support'][1100:]>0).sum()),both_hands_in_contact_at_end=bool(np.all(t[-1,12:14]>0)),replay=json.loads((p/'macro-follow07-replay.json').read_text()),series_success=False,limitation='Next physical move after this return loses the grasp. The cube pose returns closely, but wrist poses differ by 1.6–2.8 cm. This is not a reusable move basis.')
(p/'single-cycle-summary.json').write_text(json.dumps(r,ensure_ascii=False,indent=2));print(json.dumps(r,ensure_ascii=False,indent=2))
