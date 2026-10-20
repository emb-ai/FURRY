from pathlib import Path
import numpy as np,mujoco,json
from scipy.spatial.transform import Rotation as R
p=Path(__file__).resolve().parent;z=dict(np.load(p/'mpc-five.npz'));m=mujoco.MjModel.from_xml_path(str(p.parent/'rubik-sequences-20261015/project/android/assets/scene-rubik.xml'));d=mujoco.MjData(m);cq=int(m.joint('rubik_free').qposadr[0]);bids=[m.body('rubik_cubie_'+str(i)).id for i in range(8)];hands=[[b for b in range(m.nbody) if (m.body(b).name or '').startswith(side+'_hand_')] for side in ['left','right']];group=R.create_group('O');homeR=R.from_quat(z['qpos'][2089,cq+3:cq+7][[1,2,3,0]])
def grab(t):
 d.qpos[:]=z['qpos'][round(t*100)-1];mujoco.mj_kinematics(m,d);choices=R.from_quat(d.qpos[cq+3:cq+7][[1,2,3,0]])*group;r=choices[np.argmin((homeR.inv()*choices).magnitude())];c=d.xpos[bids].mean(0);return [r.inv().apply(d.xpos[h]-c) for h in hands],[r.inv()*R.from_quat(d.xquat[h][:,[1,2,3,0]]) for h in hands]
refF,refR=grab(40.5);rows=[]
for t in [20.9,40.5,59.32,78.17,93.02,117.01,139.59,159.39]:
 f,r=grab(t);err=[np.rad2deg((r[h]*refR[h].inv()).magnitude()) for h in range(2)];rows.append(dict(time=t,position_rms_mm=[float(np.sqrt(np.mean(np.sum((f[h]-refF[h])**2,axis=1)))*1000) for h in range(2)],orientation_rms_deg=[float(np.sqrt(np.mean(e**2))) for e in err],orientation_max_deg=[float(e.max()) for e in err]))
(p/'full-pose-audit.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
