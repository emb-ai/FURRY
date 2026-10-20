z=dict(np.load(O/'two-grip-series.npz'))
for k in range(len(qs),17798):
 phase=int(z['trace'][k,1]);cc=z['commands'][k];tick(cc[:14].reshape(2,7),cc[-2:])
assert np.max(abs(q-z['qpos'][17797]))==0
exec((O/'fixed_reference.py').read_text(),globals());exec((O/'contact_mpc.py').read_text(),globals());exec((O/'closure_timing.py').read_text(),globals());anchors['eight-preclose']=capture()
try:
 good,trials=close_fixed_home();anchors['eight-timed']=capture();(O/'eighth-timing-trials.json').write_text(json.dumps(trials,indent=2))
finally:
 np.savez_compressed(O/'eighth-timing.npz',trace=trace,qpos=qs,qvel=vs,userdata=states,support=support,commands=commands,actions=actions)
 (O/'eighth-timing.json').write_text(json.dumps(events,indent=2))
