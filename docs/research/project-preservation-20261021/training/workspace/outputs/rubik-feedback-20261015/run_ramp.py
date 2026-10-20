exec((O/'fixed_reference.py').read_text(),globals());exec((O/'ramp_turn.py').read_text(),globals())
try:
 good=face_turn(1);note('ramp-result',valid=bool(good),**posemetric())
 if good:
  right_released=False;exec((O/'world_return_control.py').read_text(),globals())
finally:
 np.savez_compressed(O/'ramp-sixth.npz',trace=trace,qpos=qs,qvel=vs,userdata=states,support=support,commands=commands,actions=actions)
 (O/'ramp-sixth.json').write_text(json.dumps(events,indent=2))
