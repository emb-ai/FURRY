def face_turn(direction=1):
 global phase,goals
 phase+=1;c,r=frame();axis=r.apply([0,1,0]);start=goals.copy();pivot=c+r.apply([float(os.environ.get('PIVOT_X','.03')),0,0]);base=R.from_quat(d.xquat[cubeids][:,[1,2,3,0]]);coords=r.inv().apply(d.xpos[cubeids]-c);moving=np.argsort(coords[:,1])[:4];holding=np.argsort(coords[:,1])[4:];a=0.;progress=0;recent=[]
 for k in range(1200):
  err=90*direction-progress;speed=np.clip(2*err,-25,25)
  if abs(err)>1 and abs(speed)<7:speed=np.copysign(7,err)
  if abs(err)<=1:speed=0
  a=np.clip(a+speed*.01,-180,180);g=start.copy()
  for h,fraction in enumerate([-.5,.5]):
   rot=R.from_rotvec(axis*np.deg2rad(a*fraction));g[h,:3]=pivot+rot.apply(start[h,:3]-pivot);g[h,3:]=(rot*R.from_quat(start[h,[4,5,6,3]])).as_quat()[[3,0,1,2]]
  tick(g,[1,1])
  delta=R.from_quat(d.xquat[cubeids][:,[1,2,3,0]])*base.inv();A=delta[moving].mean();B=delta[holding].mean();progress=float(np.rad2deg((B.inv()*A).as_rotvec()@axis));disp=float(np.rad2deg(np.r_[(A.inv()*delta[moving]).magnitude(),(B.inv()*delta[holding]).magnitude()]).max());recent.append(progress)
  if k%200==0:print(json.dumps(dict(k=k,angle=a,progress=progress,dispersion=disp)),flush=True)
  if k>50 and abs(progress-90*direction)<2 and np.ptp(recent[-50:])<1 and disp<5:
   note('turn-settled',progress=progress,dispersion=disp,**posemetric());return True
  if q[cq+2]<.94 or k>300 and not lib.rubik_contacts(s.p,0):break
 note('turn-failed',progress=progress,dispersion=disp,**posemetric());return False

