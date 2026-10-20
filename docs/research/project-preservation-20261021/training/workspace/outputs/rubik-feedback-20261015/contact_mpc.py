"""Local native contact model: 12 wrist DOF, original Grip only."""
from scipy.optimize import lsq_linear
SCALE=np.array([.004]*3+[np.deg2rad(2)]*3+[.006]*3+[np.deg2rad(3)]*3)
HORIZON=80;SETTLE=30

def rigid_residual(reference):
 bp,br=reference;pos=d.xpos[cubeids].copy();fit,_=R.align_vectors(pos-pos.mean(0),bp-bp.mean(0));pr=(pos-pos.mean(0))-fit.apply(bp-bp.mean(0));rr=(R.from_quat(d.xquat[cubeids][:,[1,2,3,0]])*(fit*br).inv()).as_rotvec()
 return pr,rr

def error_vector(reference):
 c,r=frame();f=fingers();pr,rr=rigid_residual(reference)
 handErrors=[]
 for h in range(2):
  error=f[h]-homeF[h];rms=np.sqrt(np.mean(np.sum(error**2,axis=1)));error*=max(0,1-.002/max(rms,1e-12));handErrors.append((error/(.006*np.sqrt(len(f[h])))).ravel())
 return np.r_[(c-homeC)/.004,(r*homeR.inv()).as_rotvec()/np.deg2rad(3),*handErrors,*[(e/(np.deg2rad(5)*np.sqrt(len(e)))).ravel() for e in finger_rotation_errors()],pr.ravel()/(.001*np.sqrt(8)),rr.ravel()/(np.deg2rad(3)*np.sqrt(8))]

def wrist_sequence(start,center,x):
 u=np.arange(1,HORIZON+1)/HORIZON;u=u*u*(3-2*u)
 for weight in np.r_[u,np.ones(SETTLE)]:
  v=x*SCALE*weight;common=R.from_rotvec(v[3:6]);g=start.copy()
  for h,sign in enumerate([-.5,.5]):
   g[h,:3]=center+common.apply(start[h,:3]-center)+v[:3]+sign*v[6:9]
   g[h,3:]=(R.from_rotvec(sign*v[9:])*common*R.from_quat(start[h,[4,5,6,3]])).as_quat()[[3,0,1,2]]
  yield g

def sequence(start,center,x,mode='closed'):
 if mode=='closed':
  for g in wrist_sequence(start,center,x):yield g,np.ones(2)
  return
 side=0 if mode=='left_relax' else 1
 relaxed=np.ones(2);relaxed[side]=.55
 for i in range(40):
  u=(i+1)/40;yield start.copy(),np.ones(2)*(1-u)+relaxed*u
 last=start.copy()
 for g in wrist_sequence(start,center,x):last=g;yield g,relaxed
 for i in range(40):
  u=(i+1)/40;yield last.copy(),relaxed*(1-u)+np.ones(2)*u
 for i in range(40):yield last.copy(),np.ones(2)

def predictive_return(iterations=45,label='contact-mpc'):
 global phase
 reference=(d.xpos[cubeids].copy(),R.from_quat(d.xquat[cubeids][:,[1,2,3,0]]));history=[];phase+=1;stall=0
 for iteration in range(iterations):
  if (O/'stop-mpc').exists():break
  initialError=error_vector(reference);initialCost=float(initialError@initialError);start=goals.copy();center,_=frame();snapshot=lib.snapshot(s.p)
  def evaluate(x,mode):
   lib.restore(s.p,snapshot);s.read()
   for g,gr in sequence(start,center,x,mode):s.step(g,gr)
   e=error_vector(reference);return e,q.copy()
  try:
   modeResults=[]
   for mode in ['closed','left_relax','right_relax']:
    ee,qq=evaluate(np.zeros(12),mode);modeResults.append((float(ee@ee),mode,ee,qq))
   _,mode,baseline,baselineQ=min(modeResults,key=lambda item:item[0]);J=np.zeros((len(baseline),12));epsilon=.25
   for j in range(12):
    x=np.zeros(12);x[j]=epsilon;e,_=evaluate(x,mode);J[:,j]=(e-baseline)/epsilon
   # Damped, bounded Gauss-Newton step; a physical rollout validates each candidate.
   damping=.15
   solution=lsq_linear(np.vstack([J,damping*np.eye(12)]),np.r_[-baseline,np.zeros(12)],bounds=(-np.ones(12),np.ones(12)),tol=1e-7,max_iter=100)
   best=(float(baseline@baseline),np.zeros(12),baselineQ)
   trials=[]
   for fraction in [1.,.5,.25]:
    x=solution.x*fraction;e,qq=evaluate(x,mode);cost=float(e@e);trials.append(cost)
    if cost<best[0]:best=(cost,x.copy(),qq.copy())
    if cost<initialCost*.9:break
   lib.restore(s.p,snapshot);s.read()
   if best[0]>=initialCost*.999:
    stall+=1
    if stall>=2:print('MPC STALLED',initialCost,best[0],flush=True);break
   else:stall=0
   for g,gr in sequence(start,center,best[1],mode):tick(g,gr)
   parity=float(np.max(abs(q-best[2])));assert parity<1e-9,parity
   met=posemetric();pr,rr=rigid_residual(reference);res=[float(np.linalg.norm(pr,axis=1).max()*1000),float(np.rad2deg(np.linalg.norm(rr,axis=1)).max())]
   rec=dict(mode=mode,mode_costs={a[1]:a[0] for a in modeResults},iteration=iteration,time=elapsed,cost_before=initialCost,cost_after=best[0],trials=trials,action=best[1].tolist(),metric=met,residual=res,prediction_max_q_error=parity)
   history.append(rec);print('MPC '+json.dumps(rec),flush=True)
   (O/'mpc-progress.json').write_text(json.dumps(dict(label=label,history=history),indent=2))
   if max(met['finger_rms_mm'])<5 and max(met['finger_orientation_rms_deg'])<4 and met['center_mm']<5 and met['rotation_deg']<5 and res[1]<3:
    note(label+'-accepted',**met);return True,history
   if q[cq+2]<.95 or res[0]>.5 or res[1]>4 or min(lib.rubik_contacts(s.p,h) for h in range(2))==0:break
  finally:lib.free_snapshot(snapshot)
 note(label+'-unfinished',**posemetric());return False,history
