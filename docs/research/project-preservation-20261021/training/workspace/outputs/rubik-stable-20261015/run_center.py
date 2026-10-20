tick=originalTick
z=dict(np.load(O/'wrist-closed-servo.npz'))
for k in range(len(qs),len(z['qpos'])):
 phase=int(z['trace'][k,1]);cc=z['commands'][k];tick(cc[:14].reshape(2,7),cc[-2:])
assert np.max(abs(q-z['qpos'][-1]))==0
phase+=1;centeringLog=[]
for k in range(2000):
 c,r=frame();v=.5*(homeC-c);v*=min(1,.01/max(np.linalg.norm(v),1e-12));omega=.3*(homeR*r.inv()).as_rotvec();omega*=min(1,np.deg2rad(5)/max(np.linalg.norm(omega),1e-12));rot=R.from_rotvec(.01*omega);g=goals.copy()
 for h in range(2):g[h,:3]=c+rot.apply(g[h,:3]-c)+.01*v;g[h,3:]=(rot*R.from_quat(g[h,[4,5,6,3]])).as_quat()[[3,0,1,2]]
 tick(g,[1,1])
 if k%100==0:
  met=posemetric();res=residual();centeringLog.append(dict(time=elapsed,metric=met,residual=res));print('CENTER '+json.dumps(centeringLog[-1]),flush=True)
  if q[cq+2]<.95 or res[1]>5:break
 if posemetric()['center_mm']<3 and posemetric()['rotation_deg']<2:break
note('centered-return',residual=residual(),**posemetric())
np.savez_compressed(O/'wrist-centered.npz',trace=trace,qpos=qs,qvel=vs,userdata=states,support=support,commands=commands,actions=actions)
(O/'wrist-centered.json').write_text(json.dumps(events,indent=2));(O/'wrist-centered-control.json').write_text(json.dumps(dict(individual_finger_commands=False,residual=residual(),metric=posemetric(),progress=centeringLog),indent=2))
