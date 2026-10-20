from pathlib import Path
import os,sys,numpy as np,mujoco,json
from scipy.spatial.transform import Rotation as R
from scipy.optimize import differential_evolution
p=Path(__file__).resolve().parent;prev=p.parent/'rubik-regrasp-20261015';m=mujoco.MjModel.from_xml_path(str(p.parent/'rubik-sequences-20261015/project/android/assets/scene-rubik.xml'));d=mujoco.MjData(m);z=dict(np.load(p/'contact-fifth.npz'));base=dict(np.load(prev/'four-cycles.npz'));k=12289;d.qpos[:]=z['qpos'][k];mujoco.mj_kinematics(m,d);q=d.qpos.copy();w=m.body('right_wrist_yaw_link').id;actual=np.r_[d.xpos[w],d.xquat[w]];cq=int(m.joint('rubik_free').qposadr[0]);bodies=[m.body('rubik_cubie_'+str(i)).id for i in range(8)];c=d.xpos[bodies].mean(axis=0);r=R.from_quat(q[cq+3:cq+7][[1,2,3,0]]);home=base['commands'][2089,:14].reshape(2,7);hd=mujoco.MjData(m);hd.qpos[:]=base['qpos'][2089];mujoco.mj_kinematics(m,hd);hc=hd.xpos[bodies].mean(axis=0);hr=R.from_quat(hd.qpos[cq+3:cq+7][[1,2,3,0]]);delta=r*hr.inv();goal=np.r_[c+delta.apply(home[1,:3]-hc),(delta*R.from_quat(home[1,[4,5,6,3]])).as_quat()[[3,0,1,2]]];current=z['commands'][k,:14].reshape(2,7)[1];wp=goal[:3]+actual[:3]-current[:3];wr=R.from_quat(actual[[4,5,6,3]])*R.from_quat(current[[4,5,6,3]]).inv()*R.from_quat(goal[[4,5,6,3]])
hg=[g for g in range(m.ngeom) if (m.geom_contype[g] or m.geom_conaffinity[g]) and (m.body(m.geom_bodyid[g]).name or '').startswith('right_hand_')];cg=[g for g in range(m.ngeom) if m.geom_bodyid[g] in bodies and (m.geom_contype[g] or m.geom_conaffinity[g])];qa=[int(m.joint('right_hand_thumb_'+str(j)+'_joint').qposadr[0]) for j in range(3)]
def quality(angles):
 d.qpos[:]=q;d.qpos[qa]=angles;mujoco.mj_kinematics(m,d);w0=d.xpos[w].copy();r0=R.from_quat(d.xquat[w][[1,2,3,0]]);rot=wr*r0.inv();pos=d.geom_xpos[hg].copy();mat=d.geom_xmat[hg].copy();mat=(rot*R.from_matrix(mat.reshape(-1,3,3))).as_matrix().reshape(-1,9);worst=1.;pair=None
 for u in np.linspace(0,1,7):
  desired=actual[:3]*(1-u)+wp*u;d.geom_xpos[hg]=desired+rot.apply(pos-w0);d.geom_xmat[hg]=mat
  for g in hg:
   for h in cg:
    gap=mujoco.mj_geomDistance(m,d,g,h,.05,None)
    if gap<worst:worst=gap;pair=[m.geom(g).name,m.geom(h).name,float(u)]
 return float(worst),pair
results=[]
for label,bounds in [('existing_control',[(0,0),(-.8,0),(-1.392,0)]),('joint_limits',[(-1.0472,1.0472),(-1.0472,.724312),(-1.74533,0)])]:
 opt=differential_evolution(lambda a:-quality(a)[0]+.00005*np.linalg.norm(a),bounds,popsize=6,maxiter=20,tol=.001,seed=7,polish=False);gap,pair=quality(opt.x);result=dict(domain=label,angles=opt.x.tolist(),min_clearance_m=gap,limiting_pair=pair,evaluations=opt.nfev);results.append(result);print(json.dumps(result),flush=True)
(p/'geometry-probe.json').write_text(json.dumps(results,indent=2))
