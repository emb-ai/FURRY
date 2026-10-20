exec((O/'closure_timing_strict.py').read_text(),globals())
try:
 # Withdraw down/outward before unwinding: the tilted cube-normal path raised
 # the wrist into a shoulder-limit IK branch in the rejected return.
 phase+=1;g=goals.copy();move(g,.8,gr=[1,0]);g[1,:3]+=[-.025,-.11,-.08];move(g,1.5)
 c,r=frame();delta=r*homeR.inv();g[1,:3]=c+delta.apply(homeG[1,:3]-homeC)+r.apply([0,-offset,0]);move(g,1.5)
 g[1,3:]=(delta*R.from_quat(homeG[1,[4,5,6,3]])).as_quat()[[3,0,1,2]];move(g,3.)
 note('unwound-outside',actual_wrist_goal_error_mm=(np.linalg.norm(s.poses()[:,:3]-goals[:,:3],axis=1)*1000).tolist())
 right_released=True;approachSource=(O/'height_aware_return.py').read_text();exec(approachSource[:approachSource.index('if distance==0:')],globals())
 if distance<=.005:
  good,timing=close_fixed_home();(O/'other-face-unwind-close-trials.json').write_text(json.dumps(timing,indent=2))
  note('other-face-unwind-return',closure_valid=bool(good),**posemetric());anchors['other-face-unwind-return']=capture()
 else:note('other-face-unwind-return-approach-failed',distance=distance)
finally:
 np.savez_compressed(O/'other-face-unwind-return.npz',trace=trace,qpos=qs,qvel=vs,userdata=states,support=support,commands=commands,actions=actions)
 (O/'other-face-unwind-return.json').write_text(json.dumps(events,indent=2))
