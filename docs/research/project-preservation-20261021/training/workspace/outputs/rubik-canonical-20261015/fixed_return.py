"""Contact-only continuous trials. Only wrist goals + Grip; table support is allowed."""
from common import *
from scipy.spatial.transform import Rotation as R
s=Sim('rubik');s.step(n=2500);m,d,q=s.m,s.d,s.q;cq=int(m.joint('rubik_free').qposadr[0]);goals=s.poses();initial=goals.copy()
trace=[];qs=[];vs=[];states=[];support=[];commands=[];events=[];elapsed=0.;phase=0;grips=np.zeros(2);direct=False;homecmd=s.cmd.copy();actions=[];run=os.environ.get('RUBIK_RUN','table')
def tick(g,gr):
 global goals,grips,elapsed
 goals=g.copy();grips=np.array(gr);fail=s.step(None if direct else goals,gr);elapsed+=.01;actions.append(s.cmd.copy() if direct else s.applied.copy())
 st=s.state();trace.append([elapsed,phase,*q[cq:cq+7],*q[:3],lib.rubik_contacts(s.p,0),lib.rubik_contacts(s.p,1),*d.xpos[s.wr[0]],*d.xpos[s.wr[1]]]);qs.append(q.copy());states.append(st);support.append(lib.supports(s.p));commands.append(np.r_[goals.ravel(),grips]);v=np.zeros(m.nv);lib.velocity(s.p,v);vs.append(v)
 if fail:raise RuntimeError('native failure')
 return st

def move(target,duration=1.,gr=None):
 start=goals.copy();old=grips.copy();gr=old if gr is None else np.array(gr)
 for k in range(round(duration*100)):
  u=(k+1)/(duration*100);u=u*u*(3-2*u);g=start*(1-u)+target*u
  for h in range(2):
   b=target[h,3:].copy()
   if start[h,3:]@b<0:b=-b
   g[h,3:]=start[h,3:]*(1-u)+b*u;g[h,3:]/=np.linalg.norm(g[h,3:])
  tick(g,old*(1-u)+gr*u)

def note(kind,**kw):
 st=s.state();r=dict(kind=kind,time=elapsed,turns=int(st[2]),active=int(st[0]),face=int(st[3]),sign=int(st[4]),cube=q[cq:cq+7].tolist(),contacts=[lib.rubik_contacts(s.p,h) for h in range(2)],**kw);events.append(r);print(json.dumps(r),flush=True)

def pickup(first=False):
 global phase
 phase+=1;obj=q[cq:cq+3].copy();Q=R.from_quat(q[cq+3:cq+7][[1,2,3,0]]) if not first else R.identity()
 # Use nearest physical core orientation after seating on the table.
 grasp=np.array([obj+Q.apply([-.14,.0675,0]),obj+Q.apply([-.14,-.0675,0])]);ori=Q.as_quat()[[3,0,1,2]]
 g=goals.copy();g[:,:3]=initial[:,:3];g[:,1]=[.34,-.34];g[:,2]=np.maximum(g[:,2]+.12,obj[2]+.22);move(g,1.5,gr=[0,0])
 g[:,:3]=grasp;g[:,1]=[.34,-.34];g[:,2]+=.12;g[:,3:]=ori;move(g,1.5)
 g[:,:3]=grasp+Q.apply([[0,.055,0],[0,-.055,0]])+[0,0,.12];move(g,1)
 g[:,:3]-=[0,0,.12];move(g,1)
 g[:,:3]=grasp;move(g,1.5)
 move(g,1,gr=[1,1]);g[:,:3]+=[0,0,.14];move(g,1.5);move(g,2)
 note('pickup');return obj,Q

def turn(obj,Q,angle=90):
 global phase
 phase+=1;start=goals.copy();st0=s.state();pivot=obj+[0,0,.14];done=False

 for k in range(700):
  a=angle*min(k/400.,1);rot=R.from_rotvec(Q.apply([0,np.deg2rad(a),0]));g=start.copy()
  if not done:
   g[1,:3]=pivot+rot.apply(start[1,:3]-pivot);g[1,3:]=(rot*R.from_quat(start[1,[4,5,6,3]])).as_quat()[[3,0,1,2]]
  else:g=goals.copy()
  st=tick(g,[1,1])
  if st[2]>st0[2]:
   f=int(st[0]);qa=int(m.joint('rubik_hinge_'+['R','L','B','F','U','D'][f]).qposadr[0]);err=abs(q[qa]-st[1])*180/np.pi
   if st[4]*(q[qa]-st[1])>=-np.deg2rad(1):done=True
 note('turn',delta=int(st[2]-st0[2]),done=done)
 return done

def putdown():
 global phase
 phase+=1;g=goals.copy();lower=max(0,q[cq+2]-(.825+.085/2+.008));g[:,:3]-=[0,0,lower];move(g,2)
 move(g,.7,gr=[0,0]);move(g,.5)
 g=goals.copy();g[0,1]+=.14;g[1,1]-=.14;move(g,1.5)
 g[:,2]+=.15;move(g,1);move(g,1)
 note('putdown')
 global direct
 phase+=1;direct=True;startcmd=homecmd.copy();startcmd[6:]=q[s.qa]
 for k in range(300):
  u=(k+1)/300;u=u*u*(3-2*u);s.cmd=startcmd*(1-u)+homecmd*u;s.cmd=s.cmd.astype(np.float32);tick(goals,[0,0])
 direct=False;lib.reset_arms.argtypes=[ptr];lib.reset_arms(s.p);goals[:]=s.poses();note('home')

def air_regrip():
 global phase
 phase+=1;g=goals.copy();move(g,.5,gr=[1,0]);g[1,1]-=float(os.environ.get('RETREAT','.12'));move(g,1)
 g[1,:3]=canonical[1,:3]+[0,-float(os.environ.get('RETREAT','.12')),0];g[1,3:]=canonical[1,3:];move(g,2)
 g[1]=canonical[1];move(g,1)
 move(g,1,gr=[1,float(os.environ.get('RIGHT_GRIP','1'))]);move(g,1);note('regrip',goal_error=float(np.max(abs(goals-canonical))))
try:
 obj,Q=pickup(True);canonical=goals.copy()
 for index in range(int(os.environ.get('MOVES','5'))):
  if not turn(obj,Q,angle=float(os.environ.get('ANGLE','100')) if index else 90):break
  air_regrip()
finally:
 np.savez_compressed(O/f'{run}.npz',trace=trace,qpos=qs,qvel=vs,userdata=states,support=support,commands=commands,actions=actions)
 (O/f'{run}.json').write_text(json.dumps(events,indent=2));s.close()
