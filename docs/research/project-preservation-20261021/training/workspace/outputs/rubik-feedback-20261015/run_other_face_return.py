exec((O/'closure_timing.py').read_text(),globals())
try:
 right_released=False;approachSource=(O/'height_aware_return.py').read_text();exec(approachSource[:approachSource.index('if distance==0:')],globals())
 if distance<=.005:
  good,timing=close_fixed_home();(O/'other-face-close-trials.json').write_text(json.dumps(timing,indent=2))
  note('other-face-return',closure_valid=bool(good),**posemetric());anchors['other-face-return']=capture()
 else:note('other-face-return-approach-failed',distance=distance)
finally:
 np.savez_compressed(O/'other-face-return.npz',trace=trace,qpos=qs,qvel=vs,userdata=states,support=support,commands=commands,actions=actions)
 (O/'other-face-return.json').write_text(json.dumps(events,indent=2))
