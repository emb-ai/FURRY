(O/'stop-mpc').unlink(missing_ok=True)
z=dict(np.load(O/'reference-series.npz'))
for k in range(len(qs),len(z['qpos'])):
 phase=int(z['trace'][k,1]);cc=z['commands'][k];tick(cc[:14].reshape(2,7),cc[-2:])
assert np.max(abs(q-z['qpos'][-1]))==0
exec((O/'fixed_reference.py').read_text(),globals());exec((O/'contact_mpc.py').read_text(),globals())
try:
 accepted,mpcHistory=predictive_return(iterations=25,label='hybrid-five')
 anchors['hybrid5']=capture()
finally:
 np.savez_compressed(O/'hybrid-five.npz',trace=trace,qpos=qs,qvel=vs,userdata=states,support=support,commands=commands,actions=actions)
 (O/'hybrid-five.json').write_text(json.dumps(events,indent=2))
 if 'mpcHistory' in globals():(O/'hybrid-five-control.json').write_text(json.dumps(dict(accepted=accepted,reference_time=referenceTime,history=mpcHistory,metric=posemetric()),indent=2))
