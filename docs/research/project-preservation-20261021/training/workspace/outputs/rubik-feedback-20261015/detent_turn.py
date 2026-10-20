"""Stop wrist drive in the physical detent window, then release and withdraw."""
def face_turn(direction=1):
 global phase,goals,right_released
 right_released=False;phase+=1;c,r=frame();axis=r.apply([0,1,0]);start=goals.copy();pivot=c;base=R.from_quat(d.xquat[cubeids][:,[1,2,3,0]]);coords=r.inv().apply(d.xpos[cubeids]-c);moving=np.argsort(coords[:,1])[:4];holding=np.argsort(coords[:,1])[4:];a=0.;progress=0
 def measured():
  delta=R.from_quat(d.xquat[cubeids][:,[1,2,3,0]])*base.inv();A=delta[moving].mean();B=delta[holding].mean();progress=float(np.rad2deg((B.inv()*A).as_rotvec()@axis));disp=float(np.rad2deg(np.r_[(A.inv()*delta[moving]).magnitude(),(B.inv()*delta[holding]).magnitude()]).max());return progress,disp
 for k in range(1000):
  err=90*direction-progress;speed=np.clip(2*err,-25,25)
  if abs(err)>1 and abs(speed)<7:speed=np.copysign(7,err)
  a=np.clip(a+speed*.01,-125,125);rot=R.from_rotvec(axis*np.deg2rad(a));g=start.copy();g[1,:3]=pivot+rot.apply(start[1,:3]-pivot);g[1,3:]=(rot*R.from_quat(start[1,[4,5,6,3]])).as_quat()[[3,0,1,2]];tick(g,[1,1]);progress,disp=measured()
  if k%100==0:print('TURN '+json.dumps(dict(k=k,angle=a,progress=progress,dispersion=disp)),flush=True)
  if progress*direction>=84 and disp<5:break
  if q[cq+2]<.95 or k>300 and not lib.rubik_contacts(s.p,0):break
 if progress*direction<84:note('turn-failed',progress=progress,dispersion=disp,**posemetric());return False
 note('detent-entry',progress=progress,dispersion=disp,**posemetric())
 move(goals,.6,gr=[1,0]);c,r=frame();g=goals.copy();g[1,:3]+=r.apply([0,-offset,0]);move(g,.8);right_released=True
 recent=[]
 for k in range(200):
  tick(goals,[1,0]);progress,disp=measured();recent.append(progress)
  if k>=50 and abs(progress-90*direction)<2 and np.ptp(recent[-50:])<1 and disp<5:
   note('detent-turn-settled',progress=progress,dispersion=disp,**posemetric());return True
 note('detent-turn-unfinished',progress=progress,dispersion=disp,**posemetric());return False
