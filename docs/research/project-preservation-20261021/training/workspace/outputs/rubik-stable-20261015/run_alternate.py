"""Transfer support and regrasp left using only wrist goals and scalar Grip."""
from scipy.optimize import minimize
tick=originalTick
z=dict(np.load(O/'wrist-only-grasp.npz'))
for k in range(len(qs),len(z['qpos'])):
 phase=int(z['trace'][k,1]);cc=z['commands'][k];tick(cc[:14].reshape(2,7),cc[-2:])
assert np.max(abs(q-z['qpos'][-1]))==0
allLog=[]
def forces():
 buf=np.zeros((256,17));nn=lib.cube_contacts(s.p,buf.ravel(),256);buf=buf[:nn]
 return np.array([buf[buf[:,0]==h,11:14].sum(axis=0) for h in range(2)])
# Establish upward support in the right hand before releasing the left.
g=goals.copy();lift=0.
for k in range(200):
 ff=forces()
 if ff[1,2]>.9:break
 lift+=.0001;g[1,2]+=.0001;tick(g,[1,1])
note('right-support-transfer',force=forces().tolist(),lift=lift,residual=residual(),**posemetric())
phase+=1;g=goals.copy();move(g,1.,gr=[0,1]);note('left-released',force=forces().tolist(),residual=residual(),**posemetric())
if q[cq+2]<.97:raise RuntimeError('Right hand did not retain cube')
c,r=frame();g=goals.copy();g[0,:3]+=r.apply([0,offset,0]);move(g,.8)
c,r=frame();delta=r*homeR.inv();g[0,:3]=c+delta.apply(homeG[0,:3]-homeC)+r.apply([0,offset,0]);g[0,3:]=(delta*R.from_quat(homeG[0,[4,5,6,3]])).as_quat()[[3,0,1,2]];move(g,1.)
fixedRight=goals[1].copy();c,r=frame();delta=r*homeR.inv();nominal=goals.copy();nominal[0,:3]=c+delta.apply(homeG[0,:3]-homeC);nominal[0,3:]=(delta*R.from_quat(homeG[0,[4,5,6,3]])).as_quat()[[3,0,1,2]]
actual=s.poses()[0];bias=actual[:3]-goals[0,:3];wr=R.from_quat(actual[[4,5,6,3]]);nr=R.from_quat(nominal[0,[4,5,6,3]]);rb=wr*R.from_quat(goals[0,[4,5,6,3]]).inv()
hg=[i for i in range(m.ngeom) if (m.geom_contype[i] or m.geom_conaffinity[i]) and ((m.body(m.geom_bodyid[i]).name or '').startswith('left_hand_') or m.body(m.geom_bodyid[i]).name=='left_wrist_yaw_link')]
plan.qpos[:]=q;mujoco.mj_kinematics(m,plan);pos=plan.geom_xpos[hg].copy();mat=plan.geom_xmat[hg].copy().reshape(-1,3,3);bestPose=None;scale=np.array([.02,.02,.02,.3,.3,.3])
def gap(x):
 global bestPose
 target=nominal[0,:3]+bias+r.apply(x[:3]);rot=rb*R.from_rotvec(r.apply(x[3:]))*nr*wr.inv();plan.geom_xmat[hg]=(rot*R.from_matrix(mat)).as_matrix().reshape(-1,9);worst=.05
 for u in [0,.5,1.]:
  wp=actual[:3]*(1-u)+target*u;plan.geom_xpos[hg]=wp+rot.apply(pos-actual[:3])
  for a in hg:
   for b in cubeGeoms:worst=min(worst,mujoco.mj_geomDistance(m,plan,a,b,.05,None))
 score=float(np.sum((x/scale)**2))
 if worst>=.00295 and (bestPose is None or score<bestPose[0]):bestPose=(score,x.copy(),float(worst))
 return worst
for xx in [-.02,-.04,-.06]:
 for yy in [0,.02,.04]:
  for zz in [-.02,0,.02]:gap(np.array([xx,yy,zz,0,0,0.]))
if bestPose is None:raise RuntimeError('No collision-free left approach seed')
opt=minimize(lambda y:float(np.dot(y,y)),bestPose[1]/scale,method='COBYLA',bounds=[(-3,2),(-1.5,2.5),(-2,2)]+[(-3,3)]*3,constraints=[{'type':'ineq','fun':lambda y:100*(gap(y*scale)-.003)}],options={'maxiter':1200,'tol':1e-4,'catol':1e-5,'rhobeg':.5})
x=bestPose[1];print('LEFT WRIST PLAN',x,bestPose[2],flush=True)
def target_left(distance):
 c,rr=frame();dd=rr*homeR.inv();g=goals.copy();g[1]=fixedRight;g[0,:3]=c+dd.apply(homeG[0,:3]-homeC)+rr.apply(x[:3]+[0,distance,0]);g[0,3:]=(R.from_rotvec(rr.apply(x[3:]))*dd*R.from_quat(homeG[0,[4,5,6,3]])).as_quat()[[3,0,1,2]];return g
move(target_left(offset),1.,gr=[0,1]);baseZ=q[cq+2];distance=offset;stalled=0
for k in range(800):
 ff=forces();allLog.append([elapsed,*ff.ravel(),distance,q[cq+2]])
 if ff[0,2]<-1.5 or q[cq+2]<baseZ-.015:stalled+=1
 else:distance=max(0,distance-.0002);stalled=0
 tick(target_left(distance),[0,1])
 if distance<=0 or stalled>=20:break
note('left-approached',distance=distance,**posemetric())
if distance==0:move(target_left(0),2.,gr=[1,1]);move(goals,1.)
note('alternating-return',residual=residual(),**posemetric())
np.savez_compressed(O/'wrist-alternate.npz',trace=trace,qpos=qs,qvel=vs,userdata=states,support=support,commands=commands,actions=actions)
(O/'wrist-alternate.json').write_text(json.dumps(events,indent=2));(O/'wrist-alternate-control.json').write_text(json.dumps(dict(individual_finger_commands=False,left_offset=x.tolist(),clearance=bestPose[2],residual=residual(),metric=posemetric(),force_log=allLog),indent=2))
