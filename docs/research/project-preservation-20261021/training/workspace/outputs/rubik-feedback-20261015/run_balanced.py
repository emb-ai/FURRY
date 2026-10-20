exec((O/'fixed_reference.py').read_text(),globals());exec((O/'balanced_turn.py').read_text(),globals())
try:
 good=face_turn(1);anchors['balanced-turn']=capture()
 note('balanced-turn-result',valid=bool(good),**posemetric())
 if good:
  right_released=False;exec((O/'coupled_return_control.py').read_text(),globals());anchors['balanced-return']=capture()
finally:
 np.savez_compressed(O/'balanced.npz',trace=trace,qpos=qs,qvel=vs,userdata=states,support=support,commands=commands,actions=actions)
 (O/'balanced.json').write_text(json.dumps(events,indent=2))
