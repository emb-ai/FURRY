"""Select a scalar Grip and outward wrist preload by native contact rollouts."""
def planned_turn(direction=1):
 global phase
 snapshot=lib.snapshot(s.p);start=goals.copy();c,r=frame();axis=r.apply([0,1,0]);base=R.from_quat(d.xquat[cubeids][:,[1,2,3,0]]);coords=r.inv().apply(d.xpos[cubeids]-c);moving=np.argsort(coords[:,1])[:4];holding=np.argsort(coords[:,1])[4:];trials=[];chosen=None
 try:
  for leftGrip,grip in [(.65,.7),(.65,1.),(.65,.65),(.55,.7),(.55,.65),(.75,.7),(.75,.65)]:
   out=0.
   lib.restore(s.p,snapshot);s.read();cmds=[];target=start.copy();target[1,:3]+=r.apply([0,-out,0]);gr=np.array([leftGrip,grip]);a=0.;progress=0.;recent=[];maximum=0.;valid=False;reason='timeout'
   def advance(g,gg):s.step(g,gg);cmds.append((g.copy(),gg.copy()))
   for k in range(90):
    u=min(1,(k+1)/60);u=u*u*(3-2*u);g=start.copy();g[:,:3]=start[:,:3]*(1-u)+target[:,:3]*u;advance(g,np.ones(2)*(1-u)+gr*u)
   for k in range(800):
    err=90*direction-progress;speed=float(np.clip(2*err,-25,25))
    if abs(err)>1 and abs(speed)<7:speed=np.copysign(7,err)
    if abs(err)<=1:speed=0
    a=np.clip(a+speed*.01,-140,140);rot=R.from_rotvec(axis*np.deg2rad(a));g=target.copy();g[1,:3]=c+rot.apply(target[1,:3]-c);g[1,3:]=(rot*R.from_quat(target[1,[4,5,6,3]])).as_quat()[[3,0,1,2]];advance(g,gr)
    delta=R.from_quat(d.xquat[cubeids][:,[1,2,3,0]])*base.inv();A=delta[moving].mean();B=delta[holding].mean();progress=float(np.rad2deg((B.inv()*A).as_rotvec()@axis));disp=float(np.rad2deg(np.r_[(A.inv()*delta[moving]).magnitude(),(B.inv()*delta[holding]).magnitude()]).max());recent.append(progress);maximum=max(maximum,progress*direction)
    if k>50 and progress*direction>80 and k%5==0:
     stopSnapshot=lib.snapshot(s.p);saved=(progress,disp)
     try:
      for stopGrip in [1.,.5,.35,0.]:
       lib.restore(s.p,stopSnapshot);s.read();stopRecent=[];stopCommands=[]
       for hold in range(100):
        gg=np.array([1.,stopGrip]);s.step(g,gg);stopCommands.append((g.copy(),gg.copy()));delta=R.from_quat(d.xquat[cubeids][:,[1,2,3,0]])*base.inv();A=delta[moving].mean();B=delta[holding].mean();stopProgress=float(np.rad2deg((B.inv()*A).as_rotvec()@axis));stopDisp=float(np.rad2deg(np.r_[(A.inv()*delta[moving]).magnitude(),(B.inv()*delta[holding]).magnitude()]).max());stopRecent.append(stopProgress)
       stopValid=abs(stopProgress-90*direction)<2 and np.ptp(stopRecent[-30:])<1 and stopDisp<3 and q[cq+2]>.95 and lib.rubik_contacts(s.p,0)!=0
       print('STOP PREVIEW '+json.dumps(dict(k=k,start_progress=saved[0],grip=stopGrip,end_progress=stopProgress,valid=bool(stopValid))),flush=True)
       if stopValid:valid=True;reason='predictive_stop';progress=stopProgress;disp=stopDisp;cmds.extend(stopCommands);break
      if not valid:lib.restore(s.p,stopSnapshot);s.read();progress,disp=saved
     finally:lib.free_snapshot(stopSnapshot)
     if valid:break
    if q[cq+2]<.95 or k>150 and not lib.rubik_contacts(s.p,0):reason='support';break
    if k>150 and maximum-progress*direction>8:reason='lost_progress';break
    if k>250 and np.ptp(recent[-100:])<1 and abs(progress)<80:reason='stalled';break
   met=posemetric();rec=dict(outward_m=out,left_grip=leftGrip,grip=grip,valid=bool(valid),reason=reason,progress_deg=progress,dispersion_deg=disp,max_progress_deg=maximum,metric=met);trials.append(rec);print('CONTACT TURN '+json.dumps(rec),flush=True)
   if valid:chosen=(cmds,q.copy(),rec);break
  lib.restore(s.p,snapshot);s.read()
  if chosen:
   phase+=1
   for g,gg in chosen[0]:tick(g,gg)
   assert np.max(abs(q-chosen[1]))<1e-9
   note('contact-planned-turn',**chosen[2]);return True,trials
  note('contact-turn-plan-failed');return False,trials
 finally:lib.free_snapshot(snapshot)
