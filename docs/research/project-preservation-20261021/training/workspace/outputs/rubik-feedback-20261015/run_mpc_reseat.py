exec((O/'closure_timing_strict.py').read_text(),globals())
try:
 right_released=False;approachSource=(O/'height_aware_return.py').read_text();exec(approachSource[:approachSource.index('if distance==0:')],globals())
 if distance<=.005:
  good,timing=close_fixed_home();met=posemetric();note('whole-regrasp-reseat',closure_valid=bool(good),**met)
  (O/'whole-regrasp-reseat-timing.json').write_text(json.dumps(timing,indent=2))
  anchors['whole-regrasp-reseat']=capture()
  if good and max(met['finger_rms_mm'])<10 and met['center_mm']<10 and met['rotation_deg']<8:
   good,trials=planned_turn(1);note('reseat-other-face-turn',completed=bool(good));(O/'reseat-other-face-turn-trials.json').write_text(json.dumps(trials,indent=2))
   if good:anchors['reseat-other-face-turned']=capture()
finally:
 np.savez_compressed(O/'other-face-reseat.npz',trace=trace,qpos=qs,qvel=vs,userdata=states,support=support,commands=commands,actions=actions)
 (O/'other-face-reseat.json').write_text(json.dumps(events,indent=2))
