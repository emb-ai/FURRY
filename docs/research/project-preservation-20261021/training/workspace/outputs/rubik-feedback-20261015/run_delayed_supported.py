exec((O/'fixed_reference.py').read_text(),globals());exec((O/'delayed_supported_turn.py').read_text(),globals())
try:
 good=face_turn(1)
 if good:
  right_released=False;exec((O/'coupled_return_control.py').read_text(),globals())
 note('delayed-support-result',turn_valid=bool(good),**posemetric())
finally:
 np.savez_compressed(O/'delayed-support.npz',trace=trace,qpos=qs,qvel=vs,userdata=states,support=support,commands=commands,actions=actions)
 (O/'delayed-support.json').write_text(json.dumps(events,indent=2))
