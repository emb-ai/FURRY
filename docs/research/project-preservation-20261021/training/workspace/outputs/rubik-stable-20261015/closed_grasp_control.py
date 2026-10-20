phase+=1;stable=0;servoLog=[]
for k in range(3000):
 c,r=frame();actual=s.poses();g=goals.copy()
 for h in range(2):
  desiredPos=c+r.apply(homeLocal[h,:3]);desiredRot=r*R.from_quat(homeLocal[h,[4,5,6,3]]);error=desiredPos-actual[h,:3];velocity=.4*error;norm=np.linalg.norm(velocity);velocity*=min(1,.003/max(norm,1e-12));g[h,:3]+=.01*velocity
  errorRot=(desiredRot*R.from_quat(actual[h,[4,5,6,3]]).inv()).as_rotvec();omega=.25*errorRot;norm=np.linalg.norm(omega);omega*=min(1,np.deg2rad(2)/max(norm,1e-12));g[h,3:]=(R.from_rotvec(.01*omega)*R.from_quat(g[h,[4,5,6,3]])).as_quat()[[3,0,1,2]]
 tick(g,[1,1])
 if k%100==0:
  met=posemetric();res=residual();servoLog.append(dict(time=elapsed,metric=met,residual=res));print('CLOSED SERVO '+json.dumps(servoLog[-1]),flush=True)
  if q[cq+2]<.95 or res[1]>5:break
 if max(posemetric()['finger_rms_mm'])<8:stable+=1
 else:stable=0
 if stable>=100:break
note('closed-servo-return',residual=residual(),**posemetric())
