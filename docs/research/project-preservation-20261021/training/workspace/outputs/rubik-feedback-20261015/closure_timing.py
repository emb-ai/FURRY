"""Choose when to close the original scalar Grip along a fixed wrist return."""
def close_fixed_home():
 global phase
 snap=lib.snapshot(s.p);start=goals.copy();startR=R.from_quat(start[:,[4,5,6,3]]);delta=R.from_quat(homeG[:,[4,5,6,3]])*startR.inv();reference=(d.xpos[cubeids].copy(),R.from_quat(d.xquat[cubeids][:,[1,2,3,0]]));best=None;trials=[]
 def sequence_timing(exponent,duration):
  count=round(duration*100)
  for k in range(count+100):
   u=min(1,(k+1)/count);u=u*u*(3-2*u);g=start.copy();g[:,:3]=start[:,:3]*(1-u)+homeG[:,:3]*u;g[:,3:]=(R.from_rotvec(delta.as_rotvec()*u)*startR).as_quat()[:,[3,0,1,2]];yield g,np.array([1.,u**exponent])
 try:
  for exponent,duration in [(1,2),(2,2),(3,2),(4,2),(2,3),(3,3),(4,3),(.5,2)]:
   lib.restore(s.p,snap);s.read();minimumZ=float(q[cq+2]);lost=False
   for g,gr in sequence_timing(exponent,duration):
    s.step(g,gr);minimumZ=min(minimumZ,float(q[cq+2]));lost|=not lib.rubik_contacts(s.p,0) and not lib.rubik_contacts(s.p,1)
   e=error_vector(reference);cost=float(e@e);met=posemetric();pr,rr=rigid_residual(reference);res=[float(np.linalg.norm(pr,axis=1).max()*1000),float(np.rad2deg(np.linalg.norm(rr,axis=1)).max())];valid=minimumZ>.95 and not lost and res[0]<.5 and res[1]<3
   rec=dict(exponent=exponent,duration=duration,cost=cost,metric=met,residual=res,valid=bool(valid));trials.append(rec);print('TIMING '+json.dumps(rec),flush=True)
   if valid and (best is None or cost<best[0]):best=(cost,exponent,duration,q.copy(),rec)
  lib.restore(s.p,snap);s.read()
  if best is None:note('closure-timing-failed');return False,trials
  phase+=1
  for g,gr in sequence_timing(best[1],best[2]):tick(g,gr)
  assert np.max(abs(q-best[3]))<1e-9
  note('timed-close',**best[4]);return True,trials
 finally:lib.free_snapshot(snap)
