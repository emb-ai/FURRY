z=dict(np.load(O/'coupled-continue.npz'))
for k in range(len(qs),len(z['qpos'])):
 phase=int(z['trace'][k,1]);cc=z['commands'][k];tick(cc[:14].reshape(2,7),cc[-2:])
assert np.max(abs(q-z['qpos'][-1]))==0
exec((O/'fixed_reference.py').read_text(),globals())
source=(PREV/'turn_series.py').read_text();exec(source[source.index('def face_turn('):source.index('def return_after_turn(')],globals())
try:
 note('sixth-coupled-start',**posemetric());good=face_turn(1);note('sixth-coupled-result',valid=bool(good),**posemetric())
 if good:
  right_released=False;exec((O/'coupled_return_control.py').read_text(),globals())
finally:
 np.savez_compressed(O/'sixth-coupled.npz',trace=trace,qpos=qs,qvel=vs,userdata=states,support=support,commands=commands,actions=actions)
 (O/'sixth-coupled.json').write_text(json.dumps(events,indent=2))
