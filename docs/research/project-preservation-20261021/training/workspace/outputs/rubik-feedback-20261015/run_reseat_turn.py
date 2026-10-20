try:
 turnStartIndex=len(commands);turnStartCenter,turnStartFrame=frame();good,trials=planned_turn(1)
 note('reseated-other-face-turn',completed=bool(good));(O/'reseated-other-face-turn-trials.json').write_text(json.dumps(trials,indent=2))
 if good:
  savedTurnCommands=np.asarray(commands[turnStartIndex:]).copy();anchors['reseated-other-face-turned']=capture()
finally:
 np.savez_compressed(O/'reseated-other-face.npz',trace=trace,qpos=qs,qvel=vs,userdata=states,support=support,commands=commands,actions=actions)
 (O/'reseated-other-face.json').write_text(json.dumps(events,indent=2))
