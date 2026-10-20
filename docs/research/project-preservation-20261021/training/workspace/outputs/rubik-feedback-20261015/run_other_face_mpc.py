exec((O/'fixed_reference.py').read_text(),globals())
exec((O/'contact_mpc.py').read_text().replace("O/'stop-mpc'","O/'stop-other-face-mpc'"),globals())
exec((O/'compiled_turn_plan.py').read_text(),globals())
try:
 good,history=predictive_return(iterations=14,label='whole-regrasp-mpc')
 (O/'whole-regrasp-mpc-control.json').write_text(json.dumps(history,indent=2))
 anchors['whole-regrasp-mpc']=capture();met=posemetric()
 if met['center_mm']<10 and met['rotation_deg']<8 and max(met['finger_rms_mm'])<12:
  good,trials=planned_turn(1)
  note('mpc-other-face-turn',completed=bool(good));(O/'mpc-other-face-turn-trials.json').write_text(json.dumps(trials,indent=2))
  if good:anchors['mpc-other-face-turned']=capture()
finally:
 np.savez_compressed(O/'other-face-mpc.npz',trace=trace,qpos=qs,qvel=vs,userdata=states,support=support,commands=commands,actions=actions)
 (O/'other-face-mpc.json').write_text(json.dumps(events,indent=2))
