from pathlib import Path
import os,json,numpy as np,mujoco
from scipy.spatial.transform import Rotation as R
p=Path(__file__).resolve().parent;name=os.environ.get('VERIFY_GRASP_RUN','planned-five');bounds=json.loads(os.environ.get('VERIFY_GRASP_BOUNDS','[40.5,59.32,78.17,93.02,117.01,131.9]'));z=dict(np.load(p/(name+'.npz')));m=mujoco.MjModel.from_xml_path(str(p.parent/'rubik-sequences-20261015/project/android/assets/scene-rubik.xml'));d=mujoco.MjData(m);cq=int(m.joint('rubik_free').qposadr[0]);bids=[m.body('rubik_cubie_'+str(i)).id for i in range(8)];handids=[[i for i in range(m.nbody) if (m.body(i).name or '').startswith(h+'_hand_')] for h in ['left','right']];group=R.create_group('O')
reference=40.5;d.qpos[:]=z['qpos'][round(reference*100)-1];mujoco.mj_kinematics(m,d);homeR=R.from_quat(d.qpos[cq+3:cq+7][[1,2,3,0]]);homeC=d.xpos[bids].mean(0);homeF=[homeR.inv().apply(d.xpos[ids]-homeC) for ids in handids];homeFR=[homeR.inv()*R.from_quat(d.xquat[ids][:,[1,2,3,0]]) for ids in handids]
results=[]
for t in bounds:
 k=round(t*100)-1;d.qpos[:]=z['qpos'][k];mujoco.mj_kinematics(m,d);choices=R.from_quat(d.qpos[cq+3:cq+7][[1,2,3,0]])*group;r=choices[np.argmin((homeR.inv()*choices).magnitude())];c=d.xpos[bids].mean(0);f=[r.inv().apply(d.xpos[ids]-c) for ids in handids];fr=[(r.inv()*R.from_quat(d.xquat[ids][:,[1,2,3,0]])*homeFR[h].inv()).magnitude() for h,ids in enumerate(handids)]
 results.append(dict(time_s=t,cube_center_error_mm=float(np.linalg.norm(c-homeC)*1000),cube_orientation_error_deg=float(np.rad2deg((r*homeR.inv()).magnitude())),hand_position_rms_mm=[float(np.sqrt(np.mean(np.sum((f[h]-homeF[h])**2,axis=1)))*1000) for h in range(2)],hand_orientation_rms_deg=[float(np.rad2deg(np.sqrt(np.mean(e**2)))) for e in fr]))
a,b=round(bounds[0]*100),round(bounds[-1]*100)
result=dict(reference_time_s=reference,boundaries=results,other_support_frames=int(np.count_nonzero(z['support'][a:b])),both_hands_without_cube_contact_frames=int(np.sum(np.all(z['trace'][a:b,12:14]==0,axis=1))),policy='HOMIE Standing',wrist_and_scalar_grip_only=True,physics_modified=False,headset_tested=False,objective_complete=False,reason='Same-face sequence only; arbitrary continuation, other faces, and robustness remain unresolved')
(p/(name+'-full-grasp.json')).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
