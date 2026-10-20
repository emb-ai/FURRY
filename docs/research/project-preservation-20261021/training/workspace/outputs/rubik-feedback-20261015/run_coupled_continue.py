(O/'stop-mpc').unlink(missing_ok=True)
z=dict(np.load(O/'coupled-series.npz'))
for k in range(len(qs),len(z['qpos'])):
 phase=int(z['trace'][k,1]);cc=z['commands'][k];tick(cc[:14].reshape(2,7),cc[-2:])
 if k==13189:anchors['coupled-close']=capture()
assert np.max(abs(q-z['qpos'][-1]))==0
anchors['coupled-mpc']=capture()
exec((O/'fixed_reference.py').read_text(),globals());exec((O/'contact_mpc.py').read_text(),globals())
try:
 accepted,history=predictive_return(iterations=22,label='coupled-continue')
 anchors['coupled-end']=capture()
 if accepted:
  note('start-sixth')
  face_turn(1)
finally:
 np.savez_compressed(O/'coupled-continue.npz',trace=trace,qpos=qs,qvel=vs,userdata=states,support=support,commands=commands,actions=actions)
 (O/'coupled-continue.json').write_text(json.dumps(events,indent=2))
