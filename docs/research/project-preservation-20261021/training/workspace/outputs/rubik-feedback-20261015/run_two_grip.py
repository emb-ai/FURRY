z=dict(np.load(O/'world-baseline-series.npz'))
for k in range(len(qs),len(z['qpos'])):
 phase=int(z['trace'][k,1]);cc=z['commands'][k];tick(cc[:14].reshape(2,7),cc[-2:])
assert np.max(abs(q-z['qpos'][-1]))==0
exec((O/'fixed_reference.py').read_text(),globals());anchors['baseline-six']=capture();exec((O/'two_grip_turn_plan.py').read_text(),globals());hist=[]
try:
 for cycle in range(7,11):
  good,trials=planned_turn(1);hist.append(dict(cycle=cycle,trials=trials));(O/'two-grip-progress.json').write_text(json.dumps(hist,indent=2))
  if not good:break
  before=(d.xpos[cubeids].copy(),R.from_quat(d.xquat[cubeids][:,[1,2,3,0]]));right_released=False;exec((O/'height_aware_return.py').read_text(),globals());pr,rr=rigid_residual(before);res=[float(np.linalg.norm(pr,axis=1).max()*1000),float(np.rad2deg(np.linalg.norm(rr,axis=1)).max())];met=posemetric();note('two-grip-cycle-end',cycle=cycle,residual=res,**met)
  if distance>0 or res[0]>.5 or res[1]>3 or met['center_mm']>8 or met['rotation_deg']>6 or max(met['finger_rms_mm'])>8 or max(met['finger_orientation_rms_deg'])>6:break
finally:
 np.savez_compressed(O/'two-grip-series.npz',trace=trace,qpos=qs,qvel=vs,userdata=states,support=support,commands=commands,actions=actions)
 (O/'two-grip-series.json').write_text(json.dumps(events,indent=2))
