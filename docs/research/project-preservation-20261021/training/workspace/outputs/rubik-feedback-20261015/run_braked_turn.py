exec((O/'fixed_reference.py').read_text(),globals());exec((O/'braked_turn_plan.py').read_text(),globals())
trialHistory=[]
try:
 for cycle in range(6,13):
  good,trials=planned_turn(1);trialHistory.append(dict(cycle=cycle,trials=trials))
  (O/'braked-turn-progress.json').write_text(json.dumps(trialHistory,indent=2))
  if not good:break
  afterTurn=(d.xpos[cubeids].copy(),R.from_quat(d.xquat[cubeids][:,[1,2,3,0]]));right_released=False
  exec((O/'world_return_control.py').read_text(),globals())
  pr,rr=rigid_residual(afterTurn);res=[float(np.linalg.norm(pr,axis=1).max()*1000),float(np.rad2deg(np.linalg.norm(rr,axis=1)).max())];met=posemetric();note('contact-cycle-end',cycle=cycle,residual=res,**met)
  if distance>0 or res[0]>.5 or res[1]>3 or met['center_mm']>10 or max(met['finger_rms_mm'])>8 or max(met['finger_orientation_rms_deg'])>6:break
finally:
 np.savez_compressed(O/'braked-turn-series.npz',trace=trace,qpos=qs,qvel=vs,userdata=states,support=support,commands=commands,actions=actions)
 (O/'braked-turn-series.json').write_text(json.dumps(events,indent=2));(O/'braked-turn-plan.json').write_text(json.dumps(trialHistory,indent=2))
