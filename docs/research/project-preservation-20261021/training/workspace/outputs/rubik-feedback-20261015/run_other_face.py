exec((O/'fixed_reference.py').read_text(),globals())
zother=dict(np.load(O/'whole-regrasp.npz'))
for k in range(len(qs),len(zother['qpos'])):
 phase=int(zother['trace'][k,1]);cc=zother['commands'][k];tick(cc[:14].reshape(2,7),cc[-2:])
assert np.max(abs(q-zother['qpos'][-1]))==0
anchors['whole-regrasp']=capture()
exec((O/'compiled_turn_plan.py').read_text(),globals())
c,r=frame();bp=d.xpos[cubeids].copy();br=R.from_quat(d.xquat[cubeids][:,[1,2,3,0]]);moving=np.argsort(r.inv().apply(bp-c)[:,1])[:4];turn=R.from_rotvec(r.apply([0,np.pi/2,0]));expected=bp.copy();expected[moving]=c+turn.apply(bp[moving]-c);expectedR=br.as_quat();expectedR[moving]=(turn*br[moving]).as_quat();expectedR=R.from_quat(expectedR)
note('different-face-start',moving_corner_ids=moving.tolist(),previous_face_corner_ids=[0,4,6,7],different_face=bool(len(set(moving)&{0,4,6,7})==2))
try:
 good,trials=planned_turn(1)
 (O/'other-face-trials.json').write_text(json.dumps(trials,indent=2))
 note('different-face-turn-result',completed=bool(good))
 if good:anchors['other-face-turned']=capture()
finally:
 np.savez_compressed(O/'other-face.npz',trace=trace,qpos=qs,qvel=vs,userdata=states,support=support,commands=commands,actions=actions)
 (O/'other-face.json').write_text(json.dumps(events,indent=2))
