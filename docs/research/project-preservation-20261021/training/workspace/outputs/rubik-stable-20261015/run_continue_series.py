tick=originalTick
z=dict(np.load(O/'five-cycles.npz'))
for k in range(len(qs),len(z['qpos'])):
 phase=int(z['trace'][k,1]);cc=z['commands'][k];tick(cc[:14].reshape(2,7),cc[-2:])
assert np.max(abs(q-z['qpos'][-1]))==0
seriesResults=[]
try:
 for cycle in range(6,13):
  note('begin-controlled-cycle',cycle=cycle,**posemetric())
  if not face_turn(1):break
  basePos=d.xpos[cubeids].copy();baseR=R.from_quat(d.xquat[cubeids][:,[1,2,3,0]])
  exec((O/'wrist_return_control.py').read_text(),globals())
  exec((O/'closed_grasp_control.py').read_text(),globals())
  met=posemetric();res=residual();ok=max(met['finger_rms_mm'])<8 and min(lib.rubik_contacts(s.p,h) for h in range(2))>0 and res[0]<.5 and res[1]<3 and q[cq+2]>.95
  note('controlled-cycle-result',cycle=cycle,accepted=bool(ok),residual=res,**met);seriesResults.append(dict(cycle=cycle,time=elapsed,accepted=bool(ok),residual=res,metric=met))
  if not ok:break
finally:
 np.savez_compressed(O/'controlled-series.npz',trace=trace,qpos=qs,qvel=vs,userdata=states,support=support,commands=commands,actions=actions)
 (O/'controlled-series.json').write_text(json.dumps(events,indent=2));(O/'controlled-series-results.json').write_text(json.dumps(seriesResults,indent=2))
