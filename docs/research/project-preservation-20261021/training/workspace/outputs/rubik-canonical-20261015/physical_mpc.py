from pathlib import Path
import os
os.environ['BRIDGE']='bridge-original-snapshot.dylib'
p=Path(__file__).resolve().parent
exec((p/'search_return.py').read_text().split('try:\n obj,Q=pickup')[0])
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
 canonical=goals.copy();turn(obj,Q,90);os.environ['DZ']='.008';air_regrip()

def regrip():
 global phase
 if not vertical:return air_regrip()
 phase+=1;g=goals.copy();move(g,.7,gr=[0,1]);g[0,2]+=.12;move(g,.8)
 g[0]=canonical[0];g[0,2]+=.12;move(g,1.5);g[0]=canonical[0];move(g,1);move(g,.8,gr=[1,1]);move(g,1);note('regrip',goal_error=float(np.max(abs(goals-canonical))))

def face_search(cycle):
 global phase,elapsed,goals,grips
 phase+=1;basePos=d.xpos[bodies].copy();baseCenter=basePos.mean(axis=0);baseR=R.from_quat(d.xquat[bodies][:,[1,2,3,0]]);coord=basePos[:,2] if vertical else -basePos[:,1];moving=np.argsort(coord)[-4:];holding=np.argsort(coord)[:4]
 def score():
  delta=R.from_quat(d.xquat[bodies][:,[1,2,3,0]])*baseR.inv();A=delta[moving].mean();B=delta[holding].mean();progress=float(np.rad2deg((B.inv()*A).as_rotvec()@axis));angle=float(np.rad2deg(B.magnitude()));pos=float(np.linalg.norm(d.xpos[bodies].mean(axis=0)-baseCenter));counts=[int(lib.rubik_contacts(s.p,h)).bit_count() for h in range(2)];deficit=max(0,3-counts[holdingHand])+max(0,2-counts[movingHand]);value=-abs(90-progress)-.4*angle-400*pos-8*deficit-60*(counts[holdingHand]==0 or counts[movingHand]==0)
  return value,progress,angle,pos,counts
 for step in range(24):
  snap=lib.snapshot(s.p);save=(elapsed,goals.copy(),grips.copy(),len(trace));best=None
  def restore():
   global elapsed,goals,grips
   lib.restore(s.p,snap);s.read();elapsed,g,gr,n=save;goals=g.copy();grips=gr.copy()
   for a in [trace,qs,vs,states,support,commands,actions]:del a[n:]
  variants=[]
  for x in [0,.035,.07]:
   for z in [-.03,0,.03]:variants.append((x,z,8,0,0,1))
  for dy in [-.015,.015]:variants.append((.035,0,8,dy,0,1))
  for ly in [-.008,.008]:variants.append((.035,0,8,0,ly,1))
  variants.extend([(.035,0,8,0,0,.7),(.035,0,-8,0,0,1),(.035,0,16,0,0,1)])
  for v in variants:
   restore();x,z,angle,dy,ly,grip=v;center=d.xpos[bodies].mean(axis=0)+([x,z,0] if vertical else [x,0,z]);rot=R.from_rotvec(axis*np.deg2rad(angle));g=goals.copy();h=movingHand;g[h,:3]=center+rot.apply(goals[h,:3]-center)+([dy,0,0] if vertical else [0,dy,0]);g[h,3:]=(rot*R.from_quat(goals[h,[4,5,6,3]])).as_quat()[[3,0,1,2]];g[holdingHand,2 if vertical else 1]+=ly
   if np.linalg.norm(g[holdingHand,:3]-canonical[holdingHand,:3])>.06:continue
   gr=[1,1];gr[holdingHand]=grip;move(g,.5,gr=gr);value=score()
   if best is None or value[0]>best[0][0]:best=(value,g.copy(),gr,v)
  restore();move(best[1],.5,gr=best[2]);lib.free_snapshot(snap)
  record=dict(cycle=cycle,step=step,time=elapsed,score=best[0],variant=best[3]);search_log.append(record);print(json.dumps(record),flush=True)
  if abs(best[0][1]-90)<1.5:
   move(goals,.75,gr=grips);settled=score();note('physical-settle',score=settled)
   if abs(settled[1]-90)<2.5:return True
  if q[cq+2]<.94:return False
 note('physical-final',score=score());return False
try:
 for cycle in range(int(os.environ.get('CYCLES','3'))):
  if not face_search(cycle):break
  regrip()
  if q[cq+2]<.94 or not lib.rubik_contacts(s.p,holdingHand):break
finally:
 run=os.environ.get('RUBIK_RUN','physical-mpc');np.savez_compressed(O/f'{run}.npz',trace=trace,qpos=qs,qvel=vs,userdata=states,support=support,commands=commands,actions=actions)
 (O/f'{run}.json').write_text(json.dumps(events,indent=2));(O/f'{run}-search.json').write_text(json.dumps(search_log,indent=2));s.close()
