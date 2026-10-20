z=dict(np.load(O/'mpc-five-v1.npz'))
for k in range(len(qs),len(z['qpos'])):
 phase=int(z['trace'][k,1]);cc=z['commands'][k];tick(cc[:14].reshape(2,7),cc[-2:])
assert np.max(abs(q-z['qpos'][-1]))==0
exec((O/'contact_mpc.py').read_text(),globals())
try:
 accepted,mpcHistory=predictive_return(iterations=20,label='refined-five')
 if accepted:anchors['centered5']=capture()
finally:
 np.savez_compressed(O/'mpc-five.npz',trace=trace,qpos=qs,qvel=vs,userdata=states,support=support,commands=commands,actions=actions)
 (O/'mpc-five.json').write_text(json.dumps(events,indent=2))
 if 'mpcHistory' in globals():(O/'mpc-five-control.json').write_text(json.dumps(dict(accepted=accepted,history=mpcHistory,metric=posemetric()),indent=2))
