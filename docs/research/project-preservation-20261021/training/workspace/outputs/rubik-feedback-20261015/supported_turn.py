"""Turn while compensating motion of the held half through common wrist commands."""
def face_turn(direction=1):
 global phase,goals,right_released
 right_released=False;phase+=1;c,r=frame();axis=r.apply([0,1,0]);start=goals.copy();pivot=c.copy();base=R.from_quat(d.xquat[cubeids][:,[1,2,3,0]]);basePos=d.xpos[cubeids].copy();coords=r.inv().apply(basePos-c);moving=np.argsort(coords[:,1])[:4];holding=np.argsort(coords[:,1])[4:];a=0.;progress=0.;corrPos=np.zeros(3);corrRot=R.identity();recent=[]
 def measured():
  delta=R.from_quat(d.xquat[cubeids][:,[1,2,3,0]])*base.inv();A=delta[moving].mean();B=delta[holding].mean();progress=float(np.rad2deg((B.inv()*A).as_rotvec()@axis));disp=float(np.rad2deg(np.r_[(A.inv()*delta[moving]).magnitude(),(B.inv()*delta[holding]).magnitude()]).max());center=d.xpos[np.array(cubeids)[holding]].mean(0)-B.apply(basePos[holding].mean(0)-pivot);return progress,disp,B,center
 for k in range(1200):
  progress,disp,B,center=measured();v=2*(pivot-center);v*=min(1,.03/max(np.linalg.norm(v),1e-12));omega=-1.5*B.as_rotvec();omega*=min(1,np.deg2rad(10)/max(np.linalg.norm(omega),1e-12));corrPos+=.01*v;corrRot=R.from_rotvec(.01*omega)*corrRot
  err=90*direction-progress;speed=np.clip(2*err,-25,25)
  if abs(err)>1 and abs(speed)<7:speed=np.copysign(7,err)
  if abs(err)<=1:speed=0
  a=np.clip(a+speed*.01,-140,140);rot=R.from_rotvec(axis*np.deg2rad(a));g=start.copy()
  for h in range(2):
   rotation=corrRot*(rot if h else R.identity());g[h,:3]=pivot+rotation.apply(start[h,:3]-pivot)+corrPos;g[h,3:]=(rotation*R.from_quat(start[h,[4,5,6,3]])).as_quat()[[3,0,1,2]]
  tick(g,[1,1]);progress,disp,B,center=measured();recent.append(progress)
  if k%100==0:print('SUPPORTED TURN '+json.dumps(dict(k=k,angle=a,progress=progress,dispersion=disp,holding_center_mm=float(np.linalg.norm(center-pivot)*1000),holding_rotation_deg=float(np.rad2deg(B.magnitude())))),flush=True)
  if k>50 and abs(progress-90*direction)<2 and np.ptp(recent[-50:])<1 and disp<5:note('supported-turn-settled',progress=progress,dispersion=disp,**posemetric());return True
  if q[cq+2]<.95 or k>300 and not lib.rubik_contacts(s.p,0):break
 note('supported-turn-failed',progress=progress,dispersion=disp,**posemetric());return False
