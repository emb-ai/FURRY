from pathlib import Path
import sys,os,json,numpy as np,mujoco
from scipy.spatial.transform import Rotation as R
p=Path(__file__).resolve().parent;sys.path.insert(0,str(p.parent/'rubik-canonical-20261015'));os.environ['BRIDGE']='bridge-original-snapshot.dylib'
from common import ASSETS
name=os.environ.get('VERIFY_RUN','stable_recover');out=os.environ.get('VERIFY_OUT','three-cycles');z=dict(np.load(p/(name+'.npz')));m=mujoco.MjModel.from_xml_path(str(ASSETS/'scene-rubik.xml'));d=mujoco.MjData(m);cq=int(m.joint('rubik_free').qposadr[0]);bids=[m.body('rubik_cubie_'+str(i)).id for i in range(8)];group=R.create_group('O');home=R.from_quat(z['qpos'][2089,cq+3:cq+7][[1,2,3,0]])
def load(t):
 i=round(t*100)-1;d.qpos[:]=z['qpos'][i];mujoco.mj_kinematics(m,d);choices=R.from_quat(d.qpos[cq+3:cq+7][[1,2,3,0]])*group;r=choices[np.argmin((home.inv()*choices).magnitude())];return d.xpos[bids].copy(),R.from_quat(d.xquat[bids][:,[1,2,3,0]]),r
right=[g for g in range(m.ngeom) if (m.geom_contype[g] or m.geom_conaffinity[g]) and ((m.body(m.geom_bodyid[g]).name or '').startswith('right_hand_') or m.body(m.geom_bodyid[g]).name=='right_wrist_yaw_link')];cube=[g for g in range(m.ngeom) if m.geom_bodyid[g] in bids and (m.geom_contype[g] or m.geom_conaffinity[g])]
results=[]
bounds=json.loads(os.environ.get('VERIFY_BOUNDS','[40.5,59.32,78.17,93.02]'))
for start,end in zip(bounds,bounds[1:]):
 a,ar,grid=load(start);b,br,_=load(end);c=a.mean(axis=0);moving=np.argsort(grid.inv().apply(a-c)[:,1])[:4];turn=R.from_rotvec(grid.apply([0,np.pi/2,0]));expected=a.copy();expected[moving]=c+turn.apply(a[moving]-c);er=ar.as_quat();er[moving]=(turn*ar[moving]).as_quat();er=R.from_quat(er);fit,_=R.align_vectors(b-b.mean(axis=0),expected-expected.mean(axis=0));pr=np.linalg.norm(fit.apply(expected-expected.mean(axis=0))-(b-b.mean(axis=0)),axis=1);rr=(br*(fit*er).inv()).magnitude();i,j=round(start*100),round(end*100);free=(z['trace'][i:j,13]==0)&(z['commands'][i:j,-1]<.1);longest=streak=0
 for v in free:streak=streak+1 if v else 0;longest=max(longest,streak)
 gaps=[]
 for k in range(i,j,10):
  if z['trace'][k,13] or z['commands'][k,-1]>=.1:continue
  d.qpos[:]=z['qpos'][k];mujoco.mj_kinematics(m,d);gap=min(mujoco.mj_geomDistance(m,d,g,h,1,None) for g in right for h in cube);gaps.append(gap)
 results.append(dict(start_s=start,end_s=end,expected_rotation_deg=90,corner_position_max_mm=float(pr.max()*1000),corner_orientation_max_deg=float(np.rad2deg(rr).max()),common_rotation_deg=float(np.rad2deg(fit.magnitude())),cube_center_drift_mm=float(np.linalg.norm(b.mean(axis=0)-a.mean(axis=0))*1000),other_support_frames=int(np.count_nonzero(z['support'][i:j])),open_contact_free_longest_s=longest/100,right_max_clearance_m_10hz=max(gaps),corners_rotated=moving.tolist()))
end=round(bounds[-1]*100);np.savez_compressed(p/(out+'.npz'),**{k:z[k][:end] for k in z});np.savetxt(p/(out+'-wrist-commands.csv'),np.c_[z['trace'][:end,0],z['commands'][:end]],delimiter=',',header='t,lx,ly,lz,lqw,lqx,lqy,lqz,rx,ry,rz,rqw,rqx,rqy,rqz,left_grip,right_grip',comments='')
result=dict(run=name,cycles=results,continuous=True,initial_warmup_s=2.5,policy='homie-standing',physics_modified=False,complete_basis_of_moves=False,limitations=['Repeated turns of the same physical face only','Complete move basis and robustness not established','Desktop native simulation; no new headset installation'])
(p/(out+'-verification.json')).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
