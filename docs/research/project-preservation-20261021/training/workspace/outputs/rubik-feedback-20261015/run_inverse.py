exec((O/'fixed_reference.py').read_text(),globals())
source=(PREV/'turn_series.py').read_text();exec(source[source.index('def face_turn('):source.index('try:\n for cycle')],globals())
try:
 for cycle,direction in enumerate([-1,1,-1,-1,1,1],5):
  note('inverse-series-start',cycle=cycle,direction=direction,**posemetric())
  if not face_turn(direction):break
  return_after_turn()
  for attempt in range(3):
   if max(posemetric()['finger_rms_mm'])<8:break
   stage(.8,True,0,False,'open');stage(.8,True,offset,False,'retreat');stage(.3,True,offset,False,'outside');stage(1.,True,0,False,'approach');stage(1.,False,0,True,'close');stage(1.,False,0,True,'settled')
  note('inverse-series-return',cycle=cycle,**posemetric())
  if q[cq+2]<.95 or max(posemetric()['finger_rms_mm'])>12:break
finally:
 np.savez_compressed(O/'inverse-series.npz',trace=trace,qpos=qs,qvel=vs,userdata=states,support=support,commands=commands,actions=actions)
 (O/'inverse-series.json').write_text(json.dumps(events,indent=2))
