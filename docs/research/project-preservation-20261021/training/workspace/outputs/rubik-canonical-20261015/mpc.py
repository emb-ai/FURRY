from pathlib import Path
import os,time
os.environ['BRIDGE']='bridge-snapshot.dylib'
p=Path(__file__).resolve().parent
exec((p/'search_return.py').read_text().split('try:\n obj,Q=pickup')[0])
lib.snapshot.argtypes=[ptr];lib.snapshot.restype=ptr;lib.restore.argtypes=[ptr,ptr];lib.free_snapshot.argtypes=[ptr]
obj,Q=pickup(True);canonical=goals.copy();turn(obj,Q,90);os.environ['DZ']='.008';air_regrip()
phase+=1;faceqa=int(m.joint('rubik_hinge_F').qposadr[0]);h0=q[faceqa];baseCenter=q[cq:cq+3].copy();baseR=R.from_quat(q[cq+3:cq+7][[1,2,3,0]]);search_log=[]
def score():
 progress=-np.rad2deg(q[faceqa]-h0);ang=(R.from_quat(q[cq+3:cq+7][[1,2,3,0]])*baseR.inv()).magnitude()*180/np.pi;pos=np.linalg.norm(q[cq:cq+3]-baseCenter);hands=[lib.rubik_contacts(s.p,h)!=0 for h in range(2)]
 return -abs(90-progress)-float(os.environ.get('ANGLE_COST','.4'))*ang-400*pos-40*(2-sum(hands)),progress,ang,pos
try:
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
  for grip in [.6,.7]:variants.append((.035,0,8,0,0,grip))
  variants.extend([(.035,0,-8,0,0,1),(.035,0,16,0,0,1),(.035,0,8,0,.008,.7)])
  for v in variants:
   restore();x,z,angle,dy,ly,grip=v;center=q[cq:cq+3]+[x,0,z];rot=R.from_rotvec([0,np.deg2rad(angle),0]);g=goals.copy();g[1,:3]=center+rot.apply(goals[1,:3]-center)+[0,dy,0];g[1,3:]=(rot*R.from_quat(goals[1,[4,5,6,3]])).as_quat()[[3,0,1,2]];g[0,1]+=ly
   if np.linalg.norm(g[0,:3]-canonical[0,:3])>.05:continue
   move(g,.5,gr=[grip,1]);value=score()
   if best is None or value[0]>best[0][0]:best=(value,g.copy(),grip,v)
  restore();move(best[1],.5,gr=[best[2],1]);lib.free_snapshot(snap)
  record=dict(step=step,time=elapsed,score=best[0],variant=best[3]);search_log.append(record);print(json.dumps(record),flush=True)
  if abs(best[0][1]-90)<2:move(goals,1,gr=grips);note('mpc-settle');break
  if q[cq+2]<.94:break
 note('mpc-final',score=score())
finally:
 run=os.environ.get('RUBIK_RUN','mpc');np.savez_compressed(O/f'{run}.npz',trace=trace,qpos=qs,qvel=vs,userdata=states,support=support,commands=commands,actions=actions)
 (O/f'{run}.json').write_text(json.dumps(events,indent=2));(O/f'{run}-search.json').write_text(json.dumps(search_log,indent=2));s.close()
