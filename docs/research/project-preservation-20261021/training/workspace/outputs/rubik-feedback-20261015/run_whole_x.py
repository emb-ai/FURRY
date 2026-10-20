exec((O/'fixed_reference.py').read_text(),globals())
basePos=d.xpos[cubeids].copy();baseRot=R.from_quat(d.xquat[cubeids][:,[1,2,3,0]]);baseCenter,baseGrid=frame();axis=baseGrid.apply([1,0,0]);desired=R.from_rotvec(axis*np.pi/2);start=goals.copy();phase+=1
try:
 for k in range(500):
  u=(k+1)/500;u=u*u*(3-2*u);rot=R.from_rotvec(axis*np.pi/2*u);g=start.copy()
  for h in range(2):g[h,:3]=baseCenter+rot.apply(start[h,:3]-baseCenter);g[h,3:]=(rot*R.from_quat(start[h,[4,5,6,3]])).as_quat()[[3,0,1,2]]
  tick(g,[1,1])
  if k%100==0:
   delta=R.from_quat(d.xquat[cubeids][:,[1,2,3,0]])*baseRot.inv();common=delta.mean();note('whole-x-progress',angle_deg=float(np.rad2deg(common.as_rotvec()@axis)),rigid_dispersion_deg=float(np.rad2deg((common.inv()*delta).magnitude()).max()))
  if q[cq+2]<.95 or not (lib.rubik_contacts(s.p,0) or lib.rubik_contacts(s.p,1)):break
 move(goals,1.)
 delta=R.from_quat(d.xquat[cubeids][:,[1,2,3,0]])*baseRot.inv();common=delta.mean();pr,rr=rigid_residual((basePos,baseRot));res=[float(np.linalg.norm(pr,axis=1).max()*1000),float(np.rad2deg(np.linalg.norm(rr,axis=1)).max())];rotationError=float(np.rad2deg((desired*common.inv()).magnitude()))
 note('whole-x-end',requested_deg=90,physical_rotation_deg=float(np.rad2deg(common.magnitude())),rotation_error_deg=rotationError,rigid_residual=res,needs_both_hands_regrasp=True,**posemetric());anchors['whole-x']=capture()
finally:
 np.savez_compressed(O/'whole-x.npz',trace=trace,qpos=qs,qvel=vs,userdata=states,support=support,commands=commands,actions=actions)
 (O/'whole-x.json').write_text(json.dumps(events,indent=2))
