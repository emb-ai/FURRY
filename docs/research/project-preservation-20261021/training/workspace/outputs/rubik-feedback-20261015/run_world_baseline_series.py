z=dict(np.load(O/'coupled-series.npz'))
for k in range(len(qs),12890):
 phase=int(z['trace'][k,1]);cc=z['commands'][k];tick(cc[:14].reshape(2,7),cc[-2:])
assert np.max(abs(q-z['qpos'][12889]))==0
exec((O/'fixed_reference.py').read_text(),globals());exec((O/'adaptive_turn_plan.py').read_text(),globals())
start=goals.copy();startR=R.from_quat(start[:,[4,5,6,3]]);delta=R.from_quat(homeG[:,[4,5,6,3]])*startR.inv();phase+=1;allTrials=[]
try:
 for k in range(300):
  u=min(1,(k+1)/200);u=u*u*(3-2*u);g=start.copy();g[:,:3]=start[:,:3]*(1-u)+homeG[:,:3]*u;g[:,3:]=(R.from_rotvec(delta.as_rotvec()*u)*startR).as_quat()[:,[3,0,1,2]];tick(g,[1,u])
 note('world-baseline-fifth',**posemetric())
 for cycle in range(6,11):
  good,trials=planned_turn(1);allTrials.append(dict(cycle=cycle,trials=trials));(O/'world-baseline-trials.json').write_text(json.dumps(allTrials,indent=2))
  if not good:break
  afterTurn=(d.xpos[cubeids].copy(),R.from_quat(d.xquat[cubeids][:,[1,2,3,0]]))
  for attempt in range(3):
   right_released=False;exec((O/'height_aware_return.py').read_text(),globals());pr,rr=rigid_residual(afterTurn);res=[float(np.linalg.norm(pr,axis=1).max()*1000),float(np.rad2deg(np.linalg.norm(rr,axis=1)).max())];met=posemetric();note('world-baseline-return',cycle=cycle,attempt=attempt,residual=res,**met)
   accepted=distance==0 and res[0]<.5 and res[1]<3 and met['center_mm']<8 and met['rotation_deg']<6 and max(met['finger_rms_mm'])<8 and max(met['finger_orientation_rms_deg'])<6
   if accepted or distance>0 or res[0]>.5 or res[1]>3:break
  if not accepted:break
  note('world-baseline-cycle-complete',cycle=cycle,**met)
finally:
 np.savez_compressed(O/'world-baseline-series.npz',trace=trace,qpos=qs,qvel=vs,userdata=states,support=support,commands=commands,actions=actions)
 (O/'world-baseline-series.json').write_text(json.dumps(events,indent=2))
