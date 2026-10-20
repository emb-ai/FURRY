exec((O/'fixed_reference.py').read_text(),globals())
source=(PREV/'turn_series.py').read_text();exec(source[source.index('def face_turn('):source.index('def return_after_turn(')],globals())
plannedResults=[]
try:
 for cycle in range(6,13):
  note('planned-cycle-start',cycle=cycle,**posemetric())
  if not face_turn(1):break
  afterTurn=(d.xpos[cubeids].copy(),R.from_quat(d.xquat[cubeids][:,[1,2,3,0]]));right_released=False
  exec((O/'world_return_control.py').read_text(),globals())
  pr,rr=rigid_residual(afterTurn);res=[float(np.linalg.norm(pr,axis=1).max()*1000),float(np.rad2deg(np.linalg.norm(rr,axis=1)).max())];met=posemetric()
  rec=dict(cycle=cycle,time=elapsed,residual=res,metric=met);plannedResults.append(rec);note('planned-cycle-return',cycle=cycle,residual=res,**met)
  if distance>0 or res[0]>.5 or res[1]>3 or met['center_mm']>10 or max(met['finger_rms_mm'])>8 or max(met['finger_orientation_rms_deg'])>6:break
finally:
 np.savez_compressed(O/'planned-series.npz',trace=trace,qpos=qs,qvel=vs,userdata=states,support=support,commands=commands,actions=actions)
 (O/'planned-series.json').write_text(json.dumps(events,indent=2));(O/'planned-series-control.json').write_text(json.dumps(plannedResults,indent=2))
