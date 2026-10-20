zrev=dict(np.load(O/'other-face-reverse-unwind.npz'))
for k in range(len(qs),len(zrev['qpos'])):
 phase=int(zrev['trace'][k,1]);cc=zrev['commands'][k];tick(cc[:14].reshape(2,7),cc[-2:])
assert np.max(abs(q-zrev['qpos'][-1]))==0
anchors['reverse-preclose']=capture()
try:
 good,timing=close_fixed_home();met=posemetric();note('reverse-close-return',closure_valid=bool(good),canonical_return=bool(good and met['center_mm']<8 and met['rotation_deg']<6 and max(met['finger_rms_mm'])<8 and max(met['finger_orientation_rms_deg'])<6),**met)
 (O/'reverse-close-timing.json').write_text(json.dumps(timing,indent=2));anchors['reverse-closed']=capture()
finally:
 np.savez_compressed(O/'other-face-reverse-closed.npz',trace=trace,qpos=qs,qvel=vs,userdata=states,support=support,commands=commands,actions=actions)
 (O/'other-face-reverse-closed.json').write_text(json.dumps(events,indent=2))
