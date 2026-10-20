exec((O/'contact_mpc.py').read_text().replace("O/'stop-mpc'","O/'stop-after-other-mpc'"),globals())
try:
 good,history=predictive_return(iterations=6,label='after-other-face-mpc');(O/'after-other-face-mpc-control.json').write_text(json.dumps(history,indent=2));met=posemetric();anchors['after-other-face-mpc']=capture()
 if met['center_mm']<15 and met['rotation_deg']<8 and max(met['finger_rms_mm'])<20:
  turnStartIndex=len(commands);turnStartCenter,turnStartFrame=frame();good,trials=planned_turn(1)
  note('second-other-face-turn',completed=bool(good));(O/'second-other-face-turn-trials.json').write_text(json.dumps(trials,indent=2))
  if good:
   savedTurnCommands=np.asarray(commands[turnStartIndex:]).copy();anchors['second-other-face-turned']=capture()
finally:
 np.savez_compressed(O/'second-other-face.npz',trace=trace,qpos=qs,qvel=vs,userdata=states,support=support,commands=commands,actions=actions)
 (O/'second-other-face.json').write_text(json.dumps(events,indent=2))
