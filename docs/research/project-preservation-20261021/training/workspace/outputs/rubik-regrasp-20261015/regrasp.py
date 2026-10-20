from pathlib import Path
import sys,os
os.environ.setdefault('BRIDGE','bridge-original-snapshot.dylib')
BASE=Path(__file__).resolve().parent.parent/'rubik-canonical-20261015'
sys.path.insert(0,str(BASE))
source=(BASE/'search_return.py').read_text().split('try:\n obj,Q=pickup')[0]
source=source.replace('[-.14,.0675,0]',"[-float(os.environ.get('GX','.14')),float(os.environ.get('GY','.0675')),0]").replace('[-.14,-.0675,0]',"[-float(os.environ.get('GX','.14')),-float(os.environ.get('GY','.0675')),0]")
exec(source)
O=Path(__file__).resolve().parent
cubeids=[m.body('rubik_cubie_'+str(i)).id for i in range(8)]
handids=[[b for b in range(m.nbody) if (m.body(b).name or '').startswith(side+'_hand_')] for side in ['left','right']]
def frame():return d.xpos[cubeids].mean(axis=0).copy(),R.from_quat(q[cq+3:cq+7][[1,2,3,0]])
def fingers():
 c,r=frame();return [r.inv().apply(d.xpos[ids]-c) for ids in handids]
def posemetric():
 c,r=frame();f=fingers();return dict(center_mm=float(np.linalg.norm(c-homeC)*1000),rotation_deg=float(np.rad2deg((r*homeR.inv()).magnitude())),finger_rms_mm=[float(np.sqrt(np.mean(np.sum((f[h]-homeF[h])**2,axis=1)))*1000) for h in range(2)],finger_max_mm=[float(np.linalg.norm(f[h]-homeF[h],axis=1).max()*1000) for h in range(2)],wrist_mm=[float(np.linalg.norm(r.inv().apply(s.poses()[h,:3]-c)-homeLocal[h,:3])*1000) for h in range(2)])
fingerTargets=np.full(14,np.nan);fingerCommands=[]
if hasattr(lib,'fingers'):
 lib.fingers.argtypes=[ptr,dp]
 originalTick=tick
 def tick(g,gr):
  lib.fingers(s.p,fingerTargets);fingerCommands.append(fingerTargets.copy());return originalTick(g,gr)
obj,Q=pickup(True)
if float(os.environ.get('ROLL','0')):
 start=goals.copy();center,_=frame()
 for k in range(400):
  u=(k+1)/400;u=u*u*(3-2*u);rot=R.from_rotvec([np.deg2rad(float(os.environ['ROLL']))*u,0,0]);g=start.copy()
  for h in range(2):g[h,:3]=center+rot.apply(start[h,:3]-center);g[h,3:]=(rot*R.from_quat(start[h,[4,5,6,3]])).as_quat()[[3,0,1,2]]
  tick(g,[1,1])
 move(g,1);note('oriented')
homeC,homeR=frame();homeG=goals.copy();homeP=s.poses();homeF=fingers();homeLocal=homeP.copy()
for h in range(2):homeLocal[h,:3]=homeR.inv().apply(homeP[h,:3]-homeC);homeLocal[h,3:]=(homeR.inv()*R.from_quat(homeP[h,[4,5,6,3]])).as_quat()[[3,0,1,2]]
holdGain=float(os.environ.get('HOLD_GAIN','0'));track=int(os.environ.get('TRACK','1'));offset=float(os.environ.get('RETREAT','.10'));closeGrip=float(os.environ.get('GRIP','1'));servo=float(os.environ.get('SERVO','0'));which=int(os.environ.get('HAND','1'));corr=np.zeros(6);rightcorr=np.zeros(6)

if hasattr(lib,'fingers'):
 aids=[a for a in range(m.nu) if 'hand_' in m.actuator(a).name]
 for k,a in enumerate(aids):
  if 'PRELOAD' in os.environ and m.actuator(a).name.startswith(('right' if which==0 else 'left')+'_hand_thumb_2'):
   fingerTargets[k]=q[m.jnt_qposadr[m.actuator_trnid[a,0]]]+(1 if which==1 else -1)*float(os.environ.get('PRELOAD','.1'))

def stage(duration,opening,outward,closing,label):
 global goals,grips,phase,corr,rightcorr
 phase+=1;start=goals.copy();old=grips.copy();h=which;held=1-h
 adjust=np.zeros(6)
 startOut=0 if label=='retreat' else offset if label=='approach' else outward
 for k in range(round(duration*100)):
  u=(k+1)/(duration*100);u=u*u*(3-2*u);c,r=frame();delta=r*homeR.inv();g=homeG.copy()
  if holdGain:
   err=homeC-c;rot=(homeR*r.inv()).as_rotvec();corr[:3]+=np.clip(err,-.02,.02)*holdGain*.01;corr[3:]+=np.clip(rot,-.3,.3)*holdGain*.01;corr[:3]=np.clip(corr[:3],-.06,.06);corr[3:]=np.clip(corr[3:],-.8,.8)
   g[held,:3]+=corr[:3];g[held,3:]=(R.from_rotvec(corr[3:])*R.from_quat(homeG[held,[4,5,6,3]])).as_quat()[[3,0,1,2]]
  if track:
   g[h,:3]=c+delta.apply(homeG[h,:3]-homeC);g[h,3:]=(delta*R.from_quat(homeG[h,[4,5,6,3]])).as_quat()[[3,0,1,2]]
  if track and 'BLEND' in os.environ:
   b=float(os.environ['BLEND']);g[h,:3]=(1-b)*g[h,:3]+b*homeG[h,:3];rblend=R.from_rotvec((R.from_quat(homeG[h,[4,5,6,3]])*R.from_quat(g[h,[4,5,6,3]]).inv()).as_rotvec()*b);g[h,3:]=(rblend*R.from_quat(g[h,[4,5,6,3]])).as_quat()[[3,0,1,2]]
  out=startOut*(1-u)+outward*u if label in ['retreat','approach'] else outward
  v=np.array([0,1 if h==0 else -1,0])*out
  g[h,:3]+=r.apply(v+np.array([float(os.environ.get('DX','0')),0,float(os.environ.get('DZ','0'))])) if track else v
  if servo:
   target=c+r.apply(homeLocal[h,:3]+v);actual=s.poses()[h,:3];desiredR=r*R.from_quat(homeLocal[h,[4,5,6,3]]);errR=(desiredR*R.from_quat(s.poses()[h,[4,5,6,3]]).inv()).as_rotvec()
   rightcorr[:3]+=np.clip(target-actual,-.03,.03)*servo*.01;rightcorr[3:]+=np.clip(errR,-.3,.3)*servo*.01;rightcorr[:3]=np.clip(rightcorr[:3],-.03,.03);rightcorr[3:]=np.clip(rightcorr[3:],-.4,.4)
   g[h,:3]+=rightcorr[:3];g[h,3:]=(R.from_rotvec(rightcorr[3:])*R.from_quat(g[h,[4,5,6,3]])).as_quat()[[3,0,1,2]]
  if label=='settled' and float(os.environ.get('ADJUST','0')):
   gain=float(os.environ['ADJUST']);wp=s.poses()[held];wantR=R.from_quat(wp[[4,5,6,3]])*R.from_quat(homeLocal[held,[4,5,6,3]]).inv();wantC=wp[:3]-wantR.apply(homeLocal[held,:3]);adjust[:3]+=np.clip(wantC-c,-.01,.01)*gain*.01;adjust[3:]+=np.clip((wantR*r.inv()).as_rotvec(),-.08,.08)*gain*.01;adjust[:3]=np.clip(adjust[:3],-.025,.025);adjust[3:]=np.clip(adjust[3:],-.15,.15)
   g[h,:3]+=adjust[:3];g[h,3:]=(R.from_rotvec(adjust[3:])*R.from_quat(g[h,[4,5,6,3]])).as_quat()[[3,0,1,2]]
  if int(os.environ.get('FREEZE_ROT','0')) and elapsed>=float(os.environ.get('FREEZE_AFTER','0')) and label in ['open','retreat','close','settled']:
   g[h,3:]=start[h,3:]
  gr=np.ones(2)*closeGrip;gr[h]=old[h]*(1-u)+(0 if opening else closeGrip)*u
  if hasattr(lib,'fingers') and os.environ.get('FINGER_MODE'):
   for j,a in enumerate(aids):
    n=m.actuator(a).name
    if not n.startswith(('left' if h==0 else 'right')+'_'):continue
    sign=1 if h==0 else -1;vgr=gr[h]
    distal=('thumb_2' in n or ('thumb' not in n and '_1_' in n))
    if label=='close':
     delay=float(os.environ.get('DELAY','.4'))
     vgr=np.clip((u-delay)/(1-delay),0,1) if distal else min(u/.6,1)
    if 'thumb_0' in n:target=0
    elif 'thumb' in n:target=sign*(1.74 if distal else 1)*vgr
    else:target=-sign*(1.74 if distal else float(os.environ.get('PROX','1.57')))*vgr
    fingerTargets[j]=target
  tick(g,gr)
 note(label,**posemetric())
try:
 for cycle in range(int(os.environ.get('CYCLES','5'))):
  stage(.8,True,0,False,'open');stage(.8,True,offset,False,'retreat');stage(.3,True,offset,False,'outside');stage(1.,True,0,False,'approach');stage(1.,False,0,True,'close');stage(1.,False,0,True,'settled')
  if q[cq+2]<.94 or not lib.rubik_contacts(s.p,1-which):break
  if cycle==0 and int(os.environ.get('REBASE','0')):
   homeC,homeR=frame();homeG=goals.copy();homeP=s.poses();homeF=fingers();homeLocal=homeP.copy()
   for h in range(2):homeLocal[h,:3]=homeR.inv().apply(homeP[h,:3]-homeC);homeLocal[h,3:]=(homeR.inv()*R.from_quat(homeP[h,[4,5,6,3]])).as_quat()[[3,0,1,2]]
   note('canonical-reset',**posemetric())
finally:
 np.savez_compressed(O/f'{run}.npz',trace=trace,qpos=qs,qvel=vs,userdata=states,support=support,commands=commands,actions=actions,finger_commands=fingerCommands)
 (O/f'{run}.json').write_text(json.dumps(events,indent=2));s.close()
