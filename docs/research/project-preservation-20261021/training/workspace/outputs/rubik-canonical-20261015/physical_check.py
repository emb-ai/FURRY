from pathlib import Path
import numpy as np,mujoco,json,sys
from scipy.spatial.transform import Rotation as R
p=Path(__file__).resolve().parent;m=mujoco.MjModel.from_xml_path(str(p.parent/'rubik-sequences-20261015/project/android/assets/scene-rubik.xml'));d=mujoco.MjData(m);ids=[m.body('rubik_cubie_'+str(i)).id for i in range(8)];wr=[m.body(h+'_wrist_yaw_link').id for h in ['left','right']];core=m.body('rubik_core').id;group=R.create_group('O')
def state(q):
 d.qpos[:]=q;mujoco.mj_kinematics(m,d)
 return d.xpos[ids].copy(),R.from_quat(d.xquat[ids][:,[1,2,3,0]]),R.from_quat(d.xquat[core][[1,2,3,0]]),d.xpos[wr].copy(),R.from_quat(d.xquat[wr][:,[1,2,3,0]])
def check(z,a,b,vertical=False):
 P0,Q0,C0,W0,V0=state(z['qpos'][a]);P1,Q1,C1,W1,V1=state(z['qpos'][b]);center0=P0.mean(axis=0);center1=P1.mean(axis=0);relative=C0*group;grid=relative[np.argmin(relative.magnitude())];axis=grid.apply([0,0,-1] if vertical else [0,1,0]);selector=grid.apply([0,0,1] if vertical else [0,-1,0]);moving=np.argsort((P0-center0)@selector)[-4:];rot=R.from_rotvec(axis*np.pi/2);expectedP=P0-center0;expectedP[moving]=rot.apply(expectedP[moving]);expectedQ=Q0.as_quat();expectedQ[moving]=(rot*Q0[moving]).as_quat();expectedQ=R.from_quat(expectedQ)
 common,rmsd=R.align_vectors(P1-center1,expectedP);residual=(Q1*(common*expectedQ).inv()).magnitude()*180/np.pi;posErr=np.linalg.norm(P1-center1-common.apply(expectedP),axis=1)
 return dict(start_s=float(z['trace'][a,0]),end_s=float(z['trace'][b,0]),corner_position_max_error_m=float(posErr.max()),corner_orientation_max_error_deg=float(residual.max()),common_rotation_deg=float(np.rad2deg(common.magnitude())),center_error_m=float(np.linalg.norm(center1-center0)),wrist_position_error_m=np.linalg.norm(W1-W0,axis=1).tolist(),wrist_orientation_error_deg=np.rad2deg((V1*V0.inv()).magnitude()).tolist(),moving_corner_ids=moving.tolist())
if __name__=='__main__':
 name=sys.argv[1];z=np.load(p/(name+'.npz'));a=int(round(float(sys.argv[2])*100))-1;b=int(round(float(sys.argv[3])*100))-1;print(json.dumps(check(z,a,b,'vertical' in sys.argv[4:]),indent=2))
