exec((O/'closure_timing_strict.py').read_text(),globals())
try:
 phase+=1;g=goals.copy();holder=g[0].copy();move(g,.8,gr=[1,0]);outward=turnStartFrame.apply([0,-.065,0]);g[1,:3]+=outward;move(g,1.)
 for cmd in savedTurnCommands[::-1]:
  g=cmd[:14].reshape(2,7).copy();g[0]=holder;g[1,:3]+=outward;tick(g,[1,0])
 note('reverse-unwind',actual_wrist_goal_error_mm=(np.linalg.norm(s.poses()[:,:3]-goals[:,:3],axis=1)*1000).tolist(),**posemetric())
 if np.linalg.norm(s.poses()[1,:3]-goals[1,:3])<.04:
  right_released=True;approachSource=(O/'height_aware_return.py').read_text();exec(approachSource[:approachSource.index('if distance==0:')],globals())
  if distance<=.005:
   good,timing=close_fixed_home();note('reverse-unwind-return',closure_valid=bool(good),**posemetric());(O/'reverse-unwind-timing.json').write_text(json.dumps(timing,indent=2));anchors['reverse-unwind-return']=capture()
finally:
 np.savez_compressed(O/'other-face-reverse-unwind.npz',trace=trace,qpos=qs,qvel=vs,userdata=states,support=support,commands=commands,actions=actions)
 (O/'other-face-reverse-unwind.json').write_text(json.dumps(events,indent=2))
