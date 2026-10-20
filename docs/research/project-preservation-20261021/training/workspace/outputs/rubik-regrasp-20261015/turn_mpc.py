from pathlib import Path
import os
p=Path(__file__).resolve().parent
source=(p/'turn_series.py').read_text();prefix,main=source.split('try:\n for cycle',1);exec(prefix)
lib.snapshot.argtypes=[ptr];lib.snapshot.restype=ptr;lib.restore.argtypes=[ptr,ptr];lib.free_snapshot.argtypes=[ptr]
searchLog=[]
def face_turn(direction=1):
 global phase,goals,grips,elapsed
 phase+=1;c,r=frame();axis=r.apply([0,1,0]);baseC=c.copy();base=R.from_quat(d.xquat[cubeids][:,[1,2,3,0]]);coords=r.inv().apply(d.xpos[cubeids]-c);moving=np.argsort(coords[:,1])[:4];holding=np.argsort(coords[:,1])[4:];startG=goals.copy()
 def score():
  delta=R.from_quat(d.xquat[cubeids][:,[1,2,3,0]])*base.inv();A=delta[moving].mean();B=delta[holding].mean();progress=float(np.rad2deg((B.inv()*A).as_rotvec()@axis));disp=float(np.rad2deg(np.r_[(A.inv()*delta[moving]).magnitude(),(B.inv()*delta[holding]).magnitude()]).max());common=float(np.rad2deg(B.magnitude()));drift=float(np.linalg.norm(d.xpos[cubeids].mean(axis=0)-baseC));counts=[int(lib.rubik_contacts(s.p,h)).bit_count() for h in range(2)];value=-abs(90*direction-progress)-.25*common-200*drift-4*disp-8*(max(0,3-counts[0])+max(0,2-counts[1]))-60*(min(counts)==0)
  return value,progress,disp,common,drift,counts,B
 for step in range(35):
  snap=lib.snapshot(s.p);save=(elapsed,goals.copy(),grips.copy(),len(trace));best=None
  def restore():
   global elapsed,goals,grips
   lib.restore(s.p,snap);s.read();elapsed,g,gr,n=save;goals=g.copy();grips=gr.copy()
   for v in [trace,qs,vs,states,support,commands,actions]:del v[n:]
  variants=[(x,z,8*direction,0) for x in [-.02,.02,.06] for z in [-.025,0,.025]]+[(.02,0,8*direction,y) for y in [-.008,.008]]+[(.02,0,16*direction,0),(.02,0,-4*direction,0)]
  for v in variants:
   restore();x,z,angle,dy=v;B=score()[-1];rr=B*r;currentAxis=B.apply(axis);pivot=d.xpos[cubeids].mean(axis=0)+rr.apply([x,0,z]);rot=R.from_rotvec(currentAxis*np.deg2rad(angle));g=goals.copy();g[1,:3]=pivot+rot.apply(goals[1,:3]-pivot)+rr.apply([0,dy,0]);g[1,3:]=(rot*R.from_quat(goals[1,[4,5,6,3]])).as_quat()[[3,0,1,2]];move(g,.4,gr=[1,1]);value=score()
   if best is None or value[0]>best[0][0]:best=(value[:-1],g.copy(),v)
  restore();move(best[1],.4,gr=[1,1]);lib.free_snapshot(snap);record=dict(time=elapsed,step=step,score=best[0],variant=best[2]);searchLog.append(record);print(json.dumps(record),flush=True)
  if abs(best[0][1]-90*direction)<2:
   move(goals,.7);value=score();note('mpc-settle',score=value[:-1])
   if abs(value[1]-90*direction)<3 and value[2]<5:return True
  if q[cq+2]<.94:return False
 note('mpc-failed',score=score()[:-1]);return False
try:exec('try:\n for cycle'+main)
finally:(O/f'{run}-search.json').write_text(json.dumps(searchLog,indent=2))
