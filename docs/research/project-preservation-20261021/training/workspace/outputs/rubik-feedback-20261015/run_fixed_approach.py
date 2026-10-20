z=dict(np.load(O/'predictive-stop-series.npz'))
for k in range(len(qs),13661):
 phase=int(z['trace'][k,1]);cc=z['commands'][k];tick(cc[:14].reshape(2,7),cc[-2:])
assert np.max(abs(q-z['qpos'][13660]))==0
exec((O/'fixed_reference.py').read_text(),globals());before=(d.xpos[cubeids].copy(),R.from_quat(d.xquat[cubeids][:,[1,2,3,0]]));right_released=False
try:
 exec((O/'fixed_approach_return.py').read_text(),globals());pr,rr=rigid_residual(before);note('fixed-approach-end',residual=[float(np.linalg.norm(pr,axis=1).max()*1000),float(np.rad2deg(np.linalg.norm(rr,axis=1)).max())],**posemetric())
finally:
 np.savez_compressed(O/'fixed-approach-sixth.npz',trace=trace,qpos=qs,qvel=vs,userdata=states,support=support,commands=commands,actions=actions)
 (O/'fixed-approach-sixth.json').write_text(json.dumps(events,indent=2))
