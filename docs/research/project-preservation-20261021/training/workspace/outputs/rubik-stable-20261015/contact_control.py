"""Contact-aware approach: optimize finger preshape in a kinematic mirror only."""
from pathlib import Path
import os
OUT=Path(__file__).resolve().parent;PREV=OUT.parent/'rubik-regrasp-20261015'
source=(OUT/'search_fifth.py').read_text().split('results=[];best=None')[0]
source=source.replace("BRIDGE='bridge-original-snapshot.dylib'","BRIDGE='bridge-contacts.dylib'")
exec(source)
from scipy.optimize import minimize_scalar
lib.cube_contacts.argtypes=[ptr,dp,ctypes.c_int];lib.cube_contacts.restype=ctypes.c_int
handGeoms=[i for i in range(m.ngeom) if (m.geom_contype[i] or m.geom_conaffinity[i]) and ((m.body(m.geom_bodyid[i]).name or '').startswith('right_hand_') or m.body(m.geom_bodyid[i]).name=='right_wrist_yaw_link')]
cubeGeoms=[i for i in range(m.ngeom) if m.geom_bodyid[i] in cubeids and (m.geom_contype[i] or m.geom_conaffinity[i])]
thumb=m.joint('right_hand_thumb_2_joint').id;thumbqa=m.jnt_qposadr[thumb];thumbidx=[m.actuator(a).name for a in aids].index('right_hand_thumb_2_joint')
plan=mujoco.MjData(m);controlLog=[]
def contact_force():
 buf=np.zeros((256,17));n=lib.cube_contacts(s.p,buf.ravel(),256);b=buf[:n];return b, b[b[:,0]==1,11:14].sum(axis=0) if n else np.zeros(3)

def preshape(goal):
 # Evaluate one-dimensional finger geometry along the entire remaining approach.
 actual=s.poses()[1];currentGoal=goals[1];wp=goal[:3]+actual[:3]-currentGoal[:3];wr=R.from_quat(actual[[4,5,6,3]])*R.from_quat(currentGoal[[4,5,6,3]]).inv()*R.from_quat(goal[[4,5,6,3]])
 def quality(angle):
  plan.qpos[:]=q;plan.qpos[thumbqa]=angle;mujoco.mj_kinematics(m,plan);w0=plan.xpos[s.wr[1]].copy();r0=R.from_quat(plan.xquat[s.wr[1]][[1,2,3,0]]);delta=wr*r0.inv();pos=plan.geom_xpos[handGeoms].copy();mat=plan.geom_xmat[handGeoms].copy();rhand=(delta*R.from_matrix(mat.reshape(-1,3,3))).as_matrix().reshape(-1,9);worst=1.
  for u in [0.,.25,.5,.75,1.]:
   desired=actual[:3]*(1-u)+wp*u;plan.geom_xpos[handGeoms]=desired+delta.apply(pos-w0);plan.geom_xmat[handGeoms]=rhand
   gap=min(mujoco.mj_geomDistance(m,plan,g,h,.05,None) for g in handGeoms for h in cubeGeoms);worst=min(worst,gap)
  return float(worst)
 optimum=minimize_scalar(lambda angle:-quality(angle)+.0001*abs(angle),bounds=(-1.392,0),method='bounded',options={'xatol':.015,'maxiter':20});choices=[(0.,quality(0.)),(-1.392,quality(-1.392)),(float(optimum.x),quality(optimum.x))];angle,gap=max(choices,key=lambda x:x[1]);controlLog.append(dict(kind='preshape',time=elapsed,angle=angle,predicted_clearance_m=gap,candidates=choices));print('PRESHAPE '+json.dumps(controlLog[-1]),flush=True);return angle

def controlled_return():
 global phase
 phase+=1;g=goals.copy();move(g,.8,gr=[1,0]);c,r=frame();g[1,:3]+=r.apply([0,-offset,0]);move(g,.8)
 c,r=frame();delta=r*homeR.inv();g[1,:3]=c+delta.apply(homeG[1,:3]-homeC)+r.apply([0,-offset,0]);g[1,3:]=(delta*R.from_quat(homeG[1,[4,5,6,3]])).as_quat()[[3,0,1,2]];move(g,1)
 c,r=frame();delta=r*homeR.inv();target=homeG.copy();target[1,:3]=c+delta.apply(homeG[1,:3]-homeC);target[1,3:]=(delta*R.from_quat(homeG[1,[4,5,6,3]])).as_quat()[[3,0,1,2]]
 angle=preshape(target[1]);fingerTargets[thumbidx]=angle;move(goals,.5,gr=[1,0]);note('preshape-ready',thumb_actual=float(q[thumbqa]))
 # Progress is suspended on excessive downward force, instead of pushing through.
 distance=offset;stalled=0;maxDown=0.;baseZ=q[cq+2]
 for k in range(600):
  b,force=contact_force();maxDown=max(maxDown,-force[2]);c,r=frame();delta=r*homeR.inv();g=homeG.copy();g[1,:3]=c+delta.apply(homeG[1,:3]-homeC)+r.apply([0,-distance,0]);g[1,3:]=(delta*R.from_quat(homeG[1,[4,5,6,3]])).as_quat()[[3,0,1,2]]
  if force[2]<-.5 or q[cq+2]<baseZ-.015:stalled+=1
  else:distance=max(0,distance-.00035);stalled=0
  tick(g,[1,0])
  if distance<=0 or stalled>=15:break
 note('force-limited-approach',distance=distance,max_down_force=maxDown,**posemetric())
 # Thumb was already flexed before approaching; close the other fingers gradually.
 start=goals.copy()
 for k in range(150):
  u=(k+1)/150;u=u*u*(3-2*u);fingerTargets[thumbidx]=angle*(1-u)-1.392*u;c,r=frame();delta=r*homeR.inv();g=homeG.copy();g[1,:3]=c+delta.apply(homeG[1,:3]-homeC)+r.apply([0,-distance,0]);g[1,3:]=(delta*R.from_quat(homeG[1,[4,5,6,3]])).as_quat()[[3,0,1,2]];tick(g,[1,u])
 fingerTargets[:]=np.nan;move(goals,1);note('contact-closed',**posemetric())
try:
 controlled_return()
 np.savez_compressed(O/'contact-fifth.npz',trace=trace,qpos=qs,qvel=vs,userdata=states,support=support,commands=commands,actions=actions,finger_commands=fingerCommands)
 (O/'contact-fifth.json').write_text(json.dumps(events,indent=2));(O/'contact-controller-log.json').write_text(json.dumps(controlLog,indent=2))
finally:lib.free_snapshot(snapshot);s.close()
