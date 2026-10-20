from pathlib import Path
import os,sys
OUT=Path(__file__).resolve().parent;PREV=OUT.parent/'rubik-regrasp-20261015'
os.environ.update(BRIDGE='bridge-original-snapshot.dylib',RUBIK_RUN='search-fifth',ROLL='-45',REBASE='1',PIVOT_X='0',RECOVER='3')
__file__=str(PREV/'turn_series.py');exec((PREV/'turn_series.py').read_text().split('try:\n for cycle')[0]);O=OUT
z=dict(np.load(PREV/'four-cycles.npz'))
assert np.max(abs(np.array(qs)-z['qpos'][:len(qs)]))==0
for k in range(len(qs),len(z['qpos'])):
 phase=int(z['trace'][k,1]);c=z['commands'][k];tick(c[:14].reshape(2,7),c[-2:])
 if k==2089:
  homeC,homeR=frame();homeG=goals.copy();homeP=s.poses();homeF=fingers();homeLocal=homeP.copy()
  for h in range(2):homeLocal[h,:3]=homeR.inv().apply(homeP[h,:3]-homeC);homeLocal[h,3:]=(homeR.inv()*R.from_quat(homeP[h,[4,5,6,3]])).as_quat()[[3,0,1,2]]
assert np.max(abs(q-z['qpos'][-1]))==0
print('PREFIX EXACT',flush=True)
events=json.loads((PREV/'support_fourth_z.json').read_text());events=[e for e in events if e['time']<=117.011]
assert face_turn(1)
lib.snapshot.argtypes=[ptr];lib.snapshot.restype=ptr;lib.restore.argtypes=[ptr,ptr];lib.free_snapshot.argtypes=[ptr]
snapshot=lib.snapshot(s.p);saved=(elapsed,phase,goals.copy(),grips.copy(),len(trace),len(events));basePos=d.xpos[cubeids].copy();baseR=R.from_quat(d.xquat[cubeids][:,[1,2,3,0]])

def restore():
 global elapsed,phase,goals,grips
 lib.restore(s.p,snapshot);s.read();elapsed,phase,g,gr,n,ne=saved;goals=g.copy();grips=gr.copy()
 for v in [trace,qs,vs,states,support,commands,actions]:del v[n:]
 del events[ne:]

def residual():
 pos=d.xpos[cubeids];fit,_=R.align_vectors(pos-pos.mean(axis=0),basePos-basePos.mean(axis=0));dp=np.linalg.norm((pos-pos.mean(axis=0))-fit.apply(basePos-basePos.mean(axis=0)),axis=1).max();dr=np.rad2deg((R.from_quat(d.xquat[cubeids][:,[1,2,3,0]])*(fit*baseR).inv()).magnitude()).max();return float(dp*1000),float(dr)

oldstage=stage
# Freeze only the quaternion during finger closure; open/approach still follows cube.
def close_stage(duration,label,mode):
 global phase
 if mode=='normal':return oldstage(duration,False,0,True,label)
 phase+=1;start=goals.copy();old=grips.copy();targetOri=start[1,3:].copy()
 for k in range(round(duration*100)):
  u=(k+1)/(duration*100);u=u*u*(3-2*u);c,r=frame();delta=r*homeR.inv();g=homeG.copy();g[1,:3]=c+delta.apply(homeG[1,:3]-homeC)
  desired=(delta*R.from_quat(homeG[1,[4,5,6,3]]));current=R.from_quat(targetOri[[1,2,3,0]])
  if mode=='slow':
   error=(desired*current.inv()).as_rotvec();norm=np.linalg.norm(error);rot=R.from_rotvec(error*min(1,.0006/max(norm,1e-9)));targetOri=(rot*current).as_quat()[[3,0,1,2]]
  g[1,3:]=targetOri;tick(g,old*(1-u)+np.ones(2)*u)
 note(label,**posemetric())

def regrasp(dx,dz,mode,extra=0):
 global phase
 phase+=1;g=goals.copy();g[0,:3]+=[dx,0,dz];move(g,.4);move(g,.8,gr=[1,0]);c,r=frame();g[1,:3]+=r.apply([0,-offset,0]);move(g,.8)
 c,r=frame();delta=r*homeR.inv();g[1,:3]=c+delta.apply(homeG[1,:3]-homeC)+r.apply([0,-offset,0]);g[1,3:]=(delta*R.from_quat(homeG[1,[4,5,6,3]])).as_quat()[[3,0,1,2]];move(g,1)
 oldstage(1.,True,0,False,'approach');close_stage(1.,'close',mode);close_stage(1.,'settled',mode)
 for attempt in range(extra):
  if max(posemetric()['finger_rms_mm'])<8:break
  oldstage(.8,True,0,False,'open');oldstage(.8,True,offset,False,'retreat');oldstage(.3,True,offset,False,'outside');oldstage(1.,True,0,False,'approach');close_stage(1.,'close',mode);close_stage(1.,'settled',mode)

results=[];best=None
try:
 variants=[(x,z,mode) for mode in ['normal','freeze','slow'] for x,z in [(0,0),(0,.008),(0,.016),(.008,0),(-.008,0)]]
 for i,v in enumerate(variants):
  restore();regrasp(*v,extra=2);met=posemetric();res=residual();counts=[lib.rubik_contacts(s.p,h).bit_count() for h in range(2)];score=max(met['finger_rms_mm'])+2*res[1]+10*res[0]+100*(q[cq+2]<.94 or min(counts)==0)
  result=dict(variant=v,score=score,metric=met,residual=res,counts=counts,time=elapsed);results.append(result);print('RESULT '+json.dumps(result),flush=True)
  if best is None or score<best['score']:
   best=result;np.savez_compressed(O/'fifth-best.npz',trace=trace,qpos=qs,qvel=vs,userdata=states,support=support,commands=commands,actions=actions);(O/'fifth-best.json').write_text(json.dumps(events,indent=2))
finally:
 (O/'fifth-search.json').write_text(json.dumps(results,indent=2));lib.free_snapshot(snapshot);s.close()
