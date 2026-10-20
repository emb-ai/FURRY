z=dict(np.load(O/'coupled-continue.npz'))
for k in range(len(qs),len(z['qpos'])):
 phase=int(z['trace'][k,1]);cc=z['commands'][k];tick(cc[:14].reshape(2,7),cc[-2:])
assert np.max(abs(q-z['qpos'][-1]))==0
exec((O/'fixed_reference.py').read_text(),globals())
source=(PREV/'turn_series.py').read_text();exec(source[source.index('def face_turn('):source.index('def return_after_turn(')],globals())
try:
 before=(d.xpos[cubeids].copy(),R.from_quat(d.xquat[cubeids][:,[1,2,3,0]]));phase+=1
 move(homeG,5.,gr=[1,1]);move(goals,3.);pr,rr=rigid_residual(before)
 note('goal-reset',residual=[float(np.linalg.norm(pr,axis=1).max()*1000),float(np.rad2deg(np.linalg.norm(rr,axis=1)).max())],**posemetric());anchors['goal-reset']=capture()
 if q[cq+2]>.95:face_turn(1)
finally:
 np.savez_compressed(O/'goal-reset.npz',trace=trace,qpos=qs,qvel=vs,userdata=states,support=support,commands=commands,actions=actions)
 (O/'goal-reset.json').write_text(json.dumps(events,indent=2))
