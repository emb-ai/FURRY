exec((O/'fixed_reference.py').read_text(),globals())
exec((O/'whole_regrasp_control.py').read_text(),globals())
wholeReference=(d.xpos[cubeids].copy(),R.from_quat(d.xquat[cubeids][:,[1,2,3,0]]))
try:
 for h in [1,0]:
  ok=regrasp_hand(h)
  delta=R.from_quat(d.xquat[cubeids][:,[1,2,3,0]])*wholeReference[1].inv();common=delta.mean();pr,rr=rigid_residual(wholeReference)
  note('whole-regrasp',hand=h,approach_completed=bool(ok),rotation_drift_deg=float(np.rad2deg(common.magnitude())),rigid_residual=[float(np.linalg.norm(pr,axis=1).max()*1000),float(np.rad2deg(np.linalg.norm(rr,axis=1)).max())],**posemetric())
  if not ok or q[cq+2]<.95:break
finally:
 np.savez_compressed(O/'whole-regrasp.npz',trace=trace,qpos=qs,qvel=vs,userdata=states,support=support,commands=commands,actions=actions)
 (O/'whole-regrasp.json').write_text(json.dumps(events,indent=2))
