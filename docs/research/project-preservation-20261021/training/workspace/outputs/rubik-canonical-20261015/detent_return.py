from pathlib import Path
import os
os.environ['BRIDGE']='bridge-original-snapshot.dylib'
p=Path(__file__).resolve().parent
source=(p/'search_return.py').read_text().split('try:\n obj,Q=pickup')[0]
source=source.replace('global goals,grips,elapsed','global goals,grips,elapsed,lastApplied')
source=source.replace('fail=s.step(None if direct else applied,gr);','if follow:applied[holdingHand,2]=followGoalZ+float(os.environ.get("FOLLOW_GAIN","1"))*(q[cq+2]-followCubeZ)\n lastApplied=applied.copy();fail=s.step(None if direct else applied,gr);')
source=source.replace('commands.append(np.r_[goals.ravel(),grips])','commands.append(np.r_[applied.ravel(),grips])')
exec(source)
follow=False;lastApplied=goals.copy();followGoalZ=0.;followCubeZ=0.

lib.snapshot.argtypes=[ptr];lib.snapshot.restype=ptr;lib.restore.argtypes=[ptr,ptr];lib.free_snapshot.argtypes=[ptr]
vertical=os.environ.get('MODE','horizontal')=='vertical';movingHand=0 if vertical else 1;holdingHand=1-movingHand;axis=np.array([0,0,-1.]) if vertical else np.array([0,1.,0]);bodies=[m.body('rubik_cubie_'+str(i)).id for i in range(8)];search_log=[]
obj,Q=pickup(True)
if vertical:
 pivot=obj+[0,0,.14];start=goals.copy()
 for k in range(400):
  rot=R.from_rotvec([np.deg2rad(90*(k+1)/400),0,0]);g=start.copy()
  for h in range(2):g[h,:3]=pivot+rot.apply(start[h,:3]-pivot);g[h,3:]=(rot*R.from_quat(start[h,[4,5,6,3]])).as_quat()[[3,0,1,2]]
  tick(g,[1,1])
 move(g,2);canonical=goals.copy();note('vertical-grasp')
else:
 canonical=goals.copy()

homeCenter=d.xpos[bodies].mean(axis=0).copy();cubeGroup=R.create_group('O')
def grid():
 choices=R.from_quat(q[cq+3:cq+7][[1,2,3,0]])*cubeGroup
 return choices[np.argmin(choices.magnitude())]
homeGrid=grid();homeGoals=goals.copy()
def align():
 global follow,goals
 follow=False;goals=lastApplied.copy()
 for repeat in range(2):
  center=d.xpos[bodies].mean(axis=0);rot=homeGrid*grid().inv();g=goals.copy()
  for h in range(2):g[h,:3]=homeCenter+rot.apply(goals[h,:3]-center);g[h,3:]=(rot*R.from_quat(goals[h,[4,5,6,3]])).as_quat()[[3,0,1,2]]
  move(g,1.5);move(g,.5)
 note('aligned')
def regrip():
 global phase,follow,followGoalZ,followCubeZ,goals
 phase+=1;align();g=goals.copy();g[0,0]+=.008;move(g,.4);followGoalZ=g[0,2];followCubeZ=q[cq+2];follow=True
 move(g,.7,gr=[1,0]);g[1,1]-=.12;move(g,.8)
 rot=grid()*homeGrid.inv();center=d.xpos[bodies].mean(axis=0);target=center+rot.apply(homeGoals[1,:3]-homeCenter);ori=(rot*R.from_quat(homeGoals[1,[4,5,6,3]])).as_quat()[[3,0,1,2]]
 g[1,:3]=target+[0,-.12,0];g[1,3:]=ori;move(g,1)
 rot=grid()*homeGrid.inv();center=d.xpos[bodies].mean(axis=0);g[1,:3]=center+rot.apply(homeGoals[1,:3]-homeCenter);g[1,3:]=(rot*R.from_quat(homeGoals[1,[4,5,6,3]])).as_quat()[[3,0,1,2]]
 g[1,2]+=float(os.environ.get('GRASP_DZ','0'));move(g,.8);move(g,.8,gr=[1,1]);move(g,.5);align();
 if os.environ.get('RESET_GOALS','1')=='1':move(homeGoals,1.5);move(homeGoals,1)
 note('regrip')

def face_search(cycle):
 global phase,elapsed,goals,grips,follow,followGoalZ,followCubeZ
 followGoalZ=goals[0,2];followCubeZ=q[cq+2];follow=True;phase+=1;start=goals.copy();pivot=obj+[0,0,.14];basePos=d.xpos[bodies].copy();baseR=R.from_quat(d.xquat[bodies][:,[1,2,3,0]]);moving=np.argsort(-basePos[:,1])[-4:];holding=np.argsort(-basePos[:,1])[:4];a=0.;progress=0.;recent=[]
 for k in range(1200):
  error=90-progress;speed=np.clip(2*error,-25,25)
  if abs(error)>1 and abs(speed)<7:speed=np.copysign(7,error)
  if abs(error)<=1:speed=0
  if os.environ.get('CONTROL','servo')=='fixed':a=min((k+1)*.25,130)
  else:a=np.clip(a+speed*.01,-25,140)
  rot=R.from_rotvec([0,np.deg2rad(a),0]);g=start.copy();g[1,:3]=pivot+rot.apply(start[1,:3]-pivot);g[1,3:]=(rot*R.from_quat(start[1,[4,5,6,3]])).as_quat()[[3,0,1,2]]
  if os.environ.get('CONTROL','servo')=='fixed' and recent and abs(90-recent[-1])<1:g=goals.copy()
  tick(g,[1,1]);delta=R.from_quat(d.xquat[bodies][:,[1,2,3,0]])*baseR.inv();A=delta[moving].mean();B=delta[holding].mean();progress=float(np.rad2deg((B.inv()*A).as_rotvec()[1]));disp=float(np.rad2deg(np.r_[(A.inv()*delta[moving]).magnitude(),(B.inv()*delta[holding]).magnitude()]).max());recent.append(progress)
  if k%100==0:print(json.dumps(dict(cycle=cycle,k=k,a=a,progress=progress,disp=disp)),flush=True)
  if len(recent)>50 and abs(progress-90)<8 and np.ptp(recent[-50:])<1.5 and disp<5 and min(int(lib.rubik_contacts(s.p,h)).bit_count() for h in range(2))>=2:
   note('pre-detent-return',progress=progress,disp=disp,goal_angle=a);return True
  if q[cq+2]<.94:break
 note('physical-final',progress=progress,disp=disp,goal_angle=a);return False
try:
 for cycle in range(int(os.environ.get('CYCLES','3'))):
  start_frame=len(qs)-1
  if not face_search(cycle):break
  regrip()
  from physical_check import check
  result=check({'qpos':np.array(qs),'trace':np.array(trace)},start_frame,len(qs)-1)
  accepted=result['corner_position_max_error_m']<.0015 and result['corner_orientation_max_error_deg']<3 and result['common_rotation_deg']<15 and result['center_error_m']<.03
  note('complete-cycle-check',accepted=accepted,metrics=result)
  if not accepted:break
  if q[cq+2]<.94 or not lib.rubik_contacts(s.p,holdingHand):break
finally:
 run=os.environ.get('RUBIK_RUN','physical-mpc');np.savez_compressed(O/f'{run}.npz',trace=trace,qpos=qs,qvel=vs,userdata=states,support=support,commands=commands,actions=actions)
 (O/f'{run}.json').write_text(json.dumps(events,indent=2));(O/f'{run}-search.json').write_text(json.dumps(search_log,indent=2));s.close()
