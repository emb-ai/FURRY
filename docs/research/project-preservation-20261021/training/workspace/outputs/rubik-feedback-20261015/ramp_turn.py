def face_turn(direction=1):
 global phase,goals
 phase+=1;c,r=frame();axis=r.apply([0,1,0]);start=goals.copy();pivot=c+r.apply([float(os.environ.get('PIVOT_X','.03')),0,0]);base=R.from_quat(d.xquat[cubeids][:,[1,2,3,0]]);coords=r.inv().apply(d.xpos[cubeids]-c);moving=np.argsort(coords[:,1])[:4];holding=np.argsort(coords[:,1])[4:];a=0.;progress=0;recent=[];previousWrists=R.from_quat(s.poses()[:,[4,5,6,3]])
 for k in range(1200):
  err=90*direction-progress;speed=np.clip(2*err,-25,25)
  if abs(err)>1 and abs(speed)<7:speed=np.copysign(7,err)
  if abs(err)<=1:speed=0
  speed*=min(1,k/100)
  a=np.clip(a+speed*.01,-140,140);rot=R.from_rotvec(axis*np.deg2rad(a));g=start.copy();g[1,:3]=pivot+rot.apply(start[1,:3]-pivot);g[1,3:]=(rot*R.from_quat(start[1,[4,5,6,3]])).as_quat()[[3,0,1,2]];tick(g,[1,1])
  delta=R.from_quat(d.xquat[cubeids][:,[1,2,3,0]])*base.inv();A=delta[moving].mean();B=delta[holding].mean();progress=float(np.rad2deg((B.inv()*A).as_rotvec()@axis));disp=float(np.rad2deg(np.r_[(A.inv()*delta[moving]).magnitude(),(B.inv()*delta[holding]).magnitude()]).max());recent.append(progress)
  wrists=R.from_quat(s.poses()[:,[4,5,6,3]]);omega=(wrists*previousWrists.inv()).as_rotvec()/.01;previousWrists=wrists
  if k%25==0:print('RAMP '+json.dumps(dict(k=k,angle=a,progress=progress,dispersion=disp,active=int(s.state()[0]),candidate=int(s.state()[5]),relative_omega_local=r.inv().apply(omega[1]-omega[0]).tolist())),flush=True)
  if k>50 and abs(progress-90*direction)<2 and np.ptp(recent[-50:])<1 and disp<5:
   note('turn-settled',progress=progress,dispersion=disp,**posemetric());return True
  if q[cq+2]<.94 or k>300 and not lib.rubik_contacts(s.p,0):break
 note('turn-failed',progress=progress,dispersion=disp,**posemetric());return False

