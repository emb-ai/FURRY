from pathlib import Path
import os
p=Path(__file__).resolve().parent
exec((p/'regrasp.py').read_text().split('try:\n for cycle')[0])
cubeGroup=R.create_group('O')
def frame():
 r=R.from_quat(q[cq+3:cq+7][[1,2,3,0]]);choices=r*cubeGroup;r=choices[np.argmin((homeR.inv()*choices).magnitude())]
 return d.xpos[cubeids].mean(axis=0).copy(),r

def face_turn(direction=1):
 global phase,goals
 phase+=1;c,r=frame();axis=r.apply([0,1,0]);start=goals.copy();pivot=c+r.apply([float(os.environ.get('PIVOT_X','.03')),0,0]);base=R.from_quat(d.xquat[cubeids][:,[1,2,3,0]]);coords=r.inv().apply(d.xpos[cubeids]-c);moving=np.argsort(coords[:,1])[:4];holding=np.argsort(coords[:,1])[4:];a=0.;progress=0;recent=[]
 for k in range(1200):
  err=90*direction-progress;speed=np.clip(2*err,-25,25)
  if abs(err)>1 and abs(speed)<7:speed=np.copysign(7,err)
  if abs(err)<=1:speed=0
  a=np.clip(a+speed*.01,-140,140);rot=R.from_rotvec(axis*np.deg2rad(a));g=start.copy();g[1,:3]=pivot+rot.apply(start[1,:3]-pivot);g[1,3:]=(rot*R.from_quat(start[1,[4,5,6,3]])).as_quat()[[3,0,1,2]];tick(g,[1,1])
  delta=R.from_quat(d.xquat[cubeids][:,[1,2,3,0]])*base.inv();A=delta[moving].mean();B=delta[holding].mean();progress=float(np.rad2deg((B.inv()*A).as_rotvec()@axis));disp=float(np.rad2deg(np.r_[(A.inv()*delta[moving]).magnitude(),(B.inv()*delta[holding]).magnitude()]).max());recent.append(progress)
  if k%200==0:print(json.dumps(dict(k=k,angle=a,progress=progress,dispersion=disp)),flush=True)
  if k>50 and abs(progress-90*direction)<2 and np.ptp(recent[-50:])<1 and disp<5:
   note('turn-settled',progress=progress,dispersion=disp,**posemetric());return True
  if q[cq+2]<.94 or k>300 and not lib.rubik_contacts(s.p,0):break
 note('turn-failed',progress=progress,dispersion=disp,**posemetric());return False

def return_after_turn():
 global phase
 phase+=1;g=goals.copy()
 if elapsed>=float(os.environ.get('SUPPORT_AFTER','10000')) and posemetric()['finger_rms_mm'][0]>=float(os.environ.get('SUPPORT_METRIC','0')):
  g[0,:3]+=np.array([float(os.environ.get('SUPPORT_X','0')),float(os.environ.get('SUPPORT_Y','0')),float(os.environ.get('SUPPORT_Z','0'))]);move(g,.4)
 move(g,.8,gr=[1,0]);c,r=frame();g[1,:3]+=r.apply([0,-offset,0]);move(g,.8)
 c,r=frame();delta=r*homeR.inv();g[1,:3]=c+delta.apply(homeG[1,:3]-homeC)+r.apply([0,-offset,0]);g[1,3:]=(delta*R.from_quat(homeG[1,[4,5,6,3]])).as_quat()[[3,0,1,2]];move(g,1)
 stage(1.,True,0,False,'approach');stage(1.,False,0,True,'close');stage(1.,False,0,True,'settled')
try:
 for cycle in range(int(os.environ.get('WARM_CYCLES','5'))):
  stage(.8,True,0,False,'open');stage(.8,True,offset,False,'retreat');stage(.3,True,offset,False,'outside');stage(1.,True,0,False,'approach');stage(1.,False,0,True,'close');stage(1.,False,0,True,'settled')
  if cycle==0 and int(os.environ.get('REBASE','0')):
   homeC,homeR=frame();homeG=goals.copy();homeP=s.poses();homeF=fingers();homeLocal=homeP.copy()
   for h in range(2):homeLocal[h,:3]=homeR.inv().apply(homeP[h,:3]-homeC);homeLocal[h,3:]=(homeR.inv()*R.from_quat(homeP[h,[4,5,6,3]])).as_quat()[[3,0,1,2]]
   note('canonical-reset',**posemetric())
 note('start-turns')
 for cycle in range(int(os.environ.get('TURNS','4'))):
  if not face_turn(int(os.environ.get('DIRECTION','1'))):break
  return_after_turn()
  for attempt in range(int(os.environ.get('RECOVER','0'))):
   e=posemetric()['finger_rms_mm']
   if max(e)<float(os.environ.get('TOL_MM','8')):break
   stage(.8,True,0,False,'open');stage(.8,True,offset,False,'retreat');stage(.3,True,offset,False,'outside');stage(1.,True,0,False,'approach');stage(1.,False,0,True,'close');stage(1.,False,0,True,'settled')
   note('recovery',attempt=attempt,**posemetric())
  final=posemetric()
  if max(final['finger_rms_mm'])>float(os.environ.get('TOL_MM','8')) or q[cq+2]<.94 or min(lib.rubik_contacts(s.p,h) for h in range(2))==0:
   note('canonical-return-failed',**final);break
  note('cycle-complete',**final)
finally:
 np.savez_compressed(O/f'{run}.npz',trace=trace,qpos=qs,qvel=vs,userdata=states,support=support,commands=commands,actions=actions)
 (O/f'{run}.json').write_text(json.dumps(events,indent=2));s.close()
