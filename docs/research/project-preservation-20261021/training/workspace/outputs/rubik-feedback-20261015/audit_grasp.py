from pathlib import Path
import numpy as np,mujoco,json
from scipy.spatial.transform import Rotation as R
p=Path(__file__).resolve().parent;z=dict(np.load(p.parent/'rubik-stable-20261015/five-cycles.npz'));m=mujoco.MjModel.from_xml_path(str(p.parent/'rubik-sequences-20261015/project/android/assets/scene-rubik.xml'));d=mujoco.MjData(m);cq=int(m.joint('rubik_free').qposadr[0]);homeQ=z['qpos'][2089];homeR=R.from_quat(homeQ[cq+3:cq+7][[1,2,3,0]]);group=R.create_group('O');out=[]
for t in [20.9,40.5,59.32,78.17,93.02,117.01,139.59]:
 q=z['qpos'][round(t*100)-1];d.qpos[:]=q;mujoco.mj_kinematics(m,d);rr=R.from_quat(q[cq+3:cq+7][[1,2,3,0]])*group;r=rr[np.argmin((homeR.inv()*rr).magnitude())];row=dict(time=t,hands=[])
 for side in ['left','right']:
  joints=[j for j in range(m.njnt) if (m.joint(j).name or '').startswith(side+'_hand_')];ids=[m.body(side+'_hand_'+s).id for s in ['thumb_2_link','index_1_link','middle_1_link']];w=m.body(side+'_wrist_yaw_link').id
  hand=dict(side=side,finger_angles_deg={m.joint(j).name:round(float(np.rad2deg(q[m.jnt_qposadr[j]])),2) for j in joints},wrist_cube_local_quat=(r.inv()*R.from_quat(d.xquat[w][[1,2,3,0]])).as_quat().tolist(),tip_cube_local_xyz=r.inv().apply(d.xpos[ids]-q[cq:cq+3]).tolist());row['hands'].append(hand)
 out.append(row)
(p/'grasp-audit.json').write_text(json.dumps(out,indent=2));print(json.dumps(out[-2:],indent=2))
