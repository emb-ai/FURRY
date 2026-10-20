"""Optimize the wrist/Grip closing phase before committing contact changes."""
from scipy.optimize import lsq_linear
z=dict(np.load(O/'predictive-stop-series.npz'))
for k in range(len(qs),14521):
 phase=int(z['trace'][k,1]);cc=z['commands'][k];tick(cc[:14].reshape(2,7),cc[-2:])
assert np.max(abs(q-z['qpos'][14520]))==0
exec((O/'fixed_reference.py').read_text(),globals());exec((O/'contact_mpc.py').read_text(),globals())
start=goals.copy();reference=(d.xpos[cubeids].copy(),R.from_quat(d.xquat[cubeids][:,[1,2,3,0]]));snap=lib.snapshot(s.p);scale=np.tile([.004,.004,.004,*([np.deg2rad(3)]*3)],2)
startR=R.from_quat(start[:,[4,5,6,3]]);targetR=R.from_quat(homeG[:,[4,5,6,3]])
def closure(x):
 target=homeG.copy()
 for h in range(2):
  target[h,:3]+=x[h*6:h*6+3]*scale[h*6:h*6+3]
  target[h,3:]=(R.from_rotvec(x[h*6+3:h*6+6]*scale[h*6+3:h*6+6])*targetR[h]).as_quat()[[3,0,1,2]]
 dr=R.from_quat(target[:,[4,5,6,3]])*startR.inv()
 for i in range(300):
  u=min(1,(i+1)/200);u=u*u*(3-2*u);g=start.copy();g[:,:3]=start[:,:3]*(1-u)+target[:,:3]*u;g[:,3:]=(R.from_rotvec(dr.as_rotvec()*u)*startR).as_quat()[:,[3,0,1,2]]
  yield g,np.array([1.,u])
def evaluate(x):
 lib.restore(s.p,snap);s.read()
 for g,gr in closure(x):s.step(g,gr)
 ee=error_vector(reference);ee=np.r_[ee,.10*x];return ee,q.copy(),posemetric()
try:
 x=np.zeros(12);e,bestQ,met=evaluate(x);bestCost=float(e@e);best=(x.copy(),bestQ.copy(),met);history=[]
 print('CLOSURE INITIAL '+json.dumps(dict(cost=bestCost,metric=met)),flush=True)
 for iteration in range(7):
  J=np.zeros((len(e),12));epsilon=.25
  for j in range(12):
   xx=x.copy();xx[j]+=epsilon;ee,_,_=evaluate(xx);J[:,j]=(ee-e)/epsilon
  sol=lsq_linear(np.vstack([J,.2*np.eye(12)]),np.r_[-e,np.zeros(12)],bounds=(np.maximum(-1.5,-4-x),np.minimum(1.5,4-x)),tol=1e-6,max_iter=100)
  trials=[];candidate=None
  for fraction in [1.,.5,.25]:
   xx=x+sol.x*fraction;ee,qq,mm=evaluate(xx);cost=float(ee@ee);trials.append(cost)
   if cost<bestCost and (candidate is None or cost<candidate[0]):candidate=(cost,xx,ee,qq,mm)
  if candidate is None:print('CLOSURE STALLED',bestCost,trials,flush=True);break
  bestCost,x,e,bestQ,met=candidate;best=(x.copy(),bestQ.copy(),met);rec=dict(iteration=iteration,cost=bestCost,parameters=x.tolist(),metric=met,trials=trials);history.append(rec);print('CLOSURE '+json.dumps(rec),flush=True)
  (O/'closure-sixth-progress.json').write_text(json.dumps(history,indent=2))
  if max(met['finger_rms_mm'])<5 and max(met['finger_orientation_rms_deg'])<4 and met['center_mm']<5 and met['rotation_deg']<5:break
 lib.restore(s.p,snap);s.read();phase+=1
 for g,gr in closure(best[0]):tick(g,gr)
 assert np.max(abs(q-best[1]))<1e-9
 pr,rr=rigid_residual(reference);note('planned-closure-end',residual=[float(np.linalg.norm(pr,axis=1).max()*1000),float(np.rad2deg(np.linalg.norm(rr,axis=1)).max())],parameters=best[0].tolist(),**posemetric());anchors['planned-sixth']=capture()
finally:
 lib.free_snapshot(snap)
 np.savez_compressed(O/'planned-closure-sixth.npz',trace=trace,qpos=qs,qvel=vs,userdata=states,support=support,commands=commands,actions=actions)
 (O/'planned-closure-sixth.json').write_text(json.dumps(events,indent=2))
