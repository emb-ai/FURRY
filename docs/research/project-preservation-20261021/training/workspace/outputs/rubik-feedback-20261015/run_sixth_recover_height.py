z=dict(np.load(O/'planned-closure-sixth.npz'))
for k in range(len(qs),len(z['qpos'])):
 phase=int(z['trace'][k,1]);cc=z['commands'][k];tick(cc[:14].reshape(2,7),cc[-2:])
assert np.max(abs(q-z['qpos'][-1]))==0
exec((O/'fixed_reference.py').read_text(),globals());before=(d.xpos[cubeids].copy(),R.from_quat(d.xquat[cubeids][:,[1,2,3,0]]))
try:
 for attempt in range(3):
  right_released=False;exec((O/'height_aware_return.py').read_text(),globals());pr,rr=rigid_residual(before);res=[float(np.linalg.norm(pr,axis=1).max()*1000),float(np.rad2deg(np.linalg.norm(rr,axis=1)).max())];met=posemetric();note('sixth-recovery',attempt=attempt,residual=res,**met)
  if distance>0 or res[0]>.5 or res[1]>3:break
  if max(met['finger_rms_mm'])<8 and max(met['finger_orientation_rms_deg'])<6 and met['center_mm']<8 and met['rotation_deg']<6:
   note('sixth-restored',**met);anchors['sixth-restored']=capture();break
finally:
 np.savez_compressed(O/'sixth-recover-height.npz',trace=trace,qpos=qs,qvel=vs,userdata=states,support=support,commands=commands,actions=actions)
 (O/'sixth-recover-height.json').write_text(json.dumps(events,indent=2))
