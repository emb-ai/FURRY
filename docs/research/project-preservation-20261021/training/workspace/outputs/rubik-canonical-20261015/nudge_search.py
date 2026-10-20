from pathlib import Path
import os
os.environ['BRIDGE']='bridge-snapshot.dylib'
p=Path(__file__).resolve().parent
exec((p/'search_return.py').read_text().split('try:\n obj,Q=pickup')[0])
lib.snapshot.argtypes=[ptr];lib.snapshot.restype=ptr;lib.restore.argtypes=[ptr,ptr];lib.free_snapshot.argtypes=[ptr]
obj,Q=pickup(True);canonical=goals.copy();turn(obj,Q,90)
snap=lib.snapshot(s.p); saved=(elapsed,phase,goals.copy(),grips.copy(),len(trace),len(events))
def restore():
 global elapsed,phase,goals,grips
 lib.restore(s.p,snap);s.read();elapsed,phase,g,gr,n,ne=saved;goals=g.copy();grips=gr.copy()
 for a in [trace,qs,vs,states,support,commands,actions]:del a[n:]
 del events[ne:]
# Check controller/history/physics snapshot restoration before using branches.
for i in range(50):tick(goals,[1,1])
expected=np.array(qs[-50:]);expected_u=np.array(states[-50:]);restore()
for i in range(50):tick(goals,[1,1])
error=float(np.max(abs(expected-np.array(qs[-50:]))));assert error==0 and np.array_equal(expected_u,np.array(states[-50:])),error
(p/'snapshot-parity.json').write_text(json.dumps(dict(qpos_max_error=error,userdata_exact=True,steps=500)))
restore()
cases=[(0,0,.015,0),(0,0,.025,0),(0,.01,.025,0),(.01,.01,.025,0),(0,.02,.025,0),(-.01,0,.025,0),(0,.01,.025,.3),(0,.015,.01,.3)]
part=int(os.environ.get('PART','0')); cases=cases[4*part:4*part+4]
try:
 for n,(dx,dy,dz,open_grip) in enumerate(cases):
  restore();run=f'nudge-{part}-{n}';os.environ['DZ']='.008';air_regrip()
  phase+=1;g=goals.copy();move(g,.5,gr=[open_grip,1]);g[0,:3]=canonical[0,:3]+[dx,dy,dz];move(g,.7);move(g,.7,gr=[1,1]);move(g,.5);note('left-nudge',params=[dx,dy,dz,open_grip])
  turn(obj,Q,110)
  np.savez_compressed(O/f'{run}.npz',trace=trace,qpos=qs,qvel=vs,userdata=states,support=support,commands=commands,actions=actions)
  (O/f'{run}.json').write_text(json.dumps(events,indent=2))
finally:lib.free_snapshot(snap);s.close()
