def regrasp_hand(h):
 global phase
 """Contact-aware return using only original wrist goals and scalar Grip."""
 from scipy.optimize import minimize
 assert not hasattr(lib,'fingers')
 # No finger override is allowed, even when using a diagnostic bridge.
 held=1-h;side=1 if h==0 else -1;holdLeft=goals[held].copy();openGrip=np.ones(2);openGrip[h]=0
 handGeoms=[i for i in range(m.ngeom) if (m.geom_contype[i] or m.geom_conaffinity[i]) and ((m.body(m.geom_bodyid[i]).name or "").startswith(("left" if h==0 else "right")+"_hand_") or (m.body(m.geom_bodyid[i]).name or "")==("left" if h==0 else "right")+"_wrist_yaw_link")]
 phase+=1;g=goals.copy()
 if True:
  move(g,.8,gr=openGrip);c,r=frame();radial=s.poses()[h,:3]-c;radial/=np.linalg.norm(radial);g[h,:3]+=radial*offset;move(g,1.2)
 c,r=frame();delta=r*homeR.inv();g[h,:3]=c+delta.apply(homeG[h,:3]-homeC)+r.apply([0,side*offset,0]);g[h,3:]=(delta*R.from_quat(homeG[h,[4,5,6,3]])).as_quat()[[3,0,1,2]];move(g,1)
 # Optimize a rigid wrist pose with measured open-hand joint angles held fixed.
 c,r=frame();delta=r*homeR.inv();nominal=goals.copy();nominal[h,:3]=c+delta.apply(homeG[h,:3]-homeC);nominal[h,3:]=(delta*R.from_quat(homeG[h,[4,5,6,3]])).as_quat()[[3,0,1,2]]
 actual=s.poses()[h];bias=actual[:3]-goals[h,:3];wr=R.from_quat(actual[[4,5,6,3]]);nr=R.from_quat(nominal[h,[4,5,6,3]]);rb=wr*R.from_quat(goals[h,[4,5,6,3]]).inv()
 plan.qpos[:]=q;mujoco.mj_kinematics(m,plan);pos=plan.geom_xpos[handGeoms].copy();mat=plan.geom_xmat[handGeoms].copy().reshape(-1,3,3);calls=0
 
 bestWrist=None
 def clearance(x):
  nonlocal calls,bestWrist
  calls+=1;target=nominal[h,:3]+bias+r.apply(x[:3]);rr=R.from_rotvec(r.apply(x[3:]));rot=rb*rr*nr*wr.inv();plan.geom_xmat[handGeoms]=(rot*R.from_matrix(mat)).as_matrix().reshape(-1,9);worst=.05
  for u in [0.,.5,1.]:
   wp=actual[:3]*(1-u)+target*u;plan.geom_xpos[handGeoms]=wp+rot.apply(pos-actual[:3])
   for a in handGeoms:
    for b in cubeGeoms:worst=min(worst,mujoco.mj_geomDistance(m,plan,a,b,.05,None))
  score=float(np.sum((x/np.array([.02,.02,.02,.3,.3,.3]))**2))
  if worst>=.00295 and (bestWrist is None or score<bestWrist[0]):bestWrist=(score,x.copy(),float(worst))
  return worst
 scale=np.array([.02,.02,.02,.3,.3,.3])
 for xx in [-.02,-.04,-.06]:
  for yy in [0,side*.02,side*.04]:
   for zz in [-.02,0,.02]:clearance(np.array([xx,yy,zz,0,0,0.]))
 print('GEOMETRIC FEASIBLE SEED',bestWrist,flush=True)
 opt=minimize(lambda y:float(np.dot(y,y)),(bestWrist[1]/scale if bestWrist else np.array([-1.5,-1.,.5,0,0,0.])),method='COBYLA',bounds=[(-3,2),((-1.5,2.5) if h==0 else (-2.5,1.5)),(-2,2)]+[(-3,3)]*3,constraints=[{'type':'ineq','fun':lambda y:100*(clearance(y*scale)-.003)}],options={'maxiter':300,'tol':1e-4,'catol':1e-5,'rhobeg':.5})
 opt.x=bestWrist[1] if bestWrist else opt.x*scale
 result=dict(success=bool(opt.success),message=str(opt.message),offset_m=opt.x[:3].tolist(),rotation_rad=opt.x[3:].tolist(),clearance_m=float(clearance(opt.x)),evaluations=calls,individual_finger_commands=False)
 print('WRIST PLAN '+json.dumps(result),flush=True)
 x=opt.x;forceLog=[]
 if result['clearance_m']<.001:raise RuntimeError('No valid wrist approach')
 def wrist_goal(distance,blend=1.):
  c,r=frame();delta=r*homeR.inv();g=goals.copy();g[held]=holdLeft;g[h,:3]=c+delta.apply(homeG[h,:3]-homeC)+r.apply(x[:3]*blend+[0,side*distance,0]);g[h,3:]=(R.from_rotvec(r.apply(x[3:]*blend))*delta*R.from_quat(homeG[h,[4,5,6,3]])).as_quat()[[3,0,1,2]];return g
 move(wrist_goal(offset),1.,gr=openGrip);baseZ=q[cq+2];distance=offset;stalled=0
 for k in range(800):
  b,_=contact_force();force=b[b[:,0]==h,11:14].sum(0);forceLog.append([elapsed,*force,distance,q[cq+2]])
  if force[2]<-1.5 or q[cq+2]<min(baseZ-.015,homeC[2]-.02):stalled+=1
  else:distance=max(0,distance-.0002);stalled=0
  tick(wrist_goal(distance),openGrip)
  if distance<=0 or stalled>=20:break
 note('wrist-only-approach',remaining_distance=distance,max_down_force=-min(a[3] for a in forceLog),**posemetric())
 if distance==0:
  closeC,closeR=frame();closeDelta=closeR*homeR.inv();closePosition=closeC+closeDelta.apply(homeG[h,:3]-homeC);closeRotation=closeDelta*R.from_quat(homeG[h,[4,5,6,3]])
  for k in range(200):
   u=(k+1)/200;u=u*u*(3-2*u);g=goals.copy();g[held]=holdLeft;g[h,:3]=closePosition+closeR.apply(x[:3]*(1-u));g[h,3:]=(R.from_rotvec(closeR.apply(x[3:]*(1-u)))*closeRotation).as_quat()[[3,0,1,2]];closeGrip=np.ones(2);closeGrip[h]=u;tick(g,closeGrip)
  move(goals,1.);note('coupled-grip-close',**posemetric())
 
 return distance==0
