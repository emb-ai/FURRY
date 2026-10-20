"""Continue from the verified prefix; accepted actions only, no resets between moves."""
(O/'stop-mpc').unlink(missing_ok=True)
z=dict(np.load(O/'mpc-five.npz'))
for k in range(len(qs),len(z['qpos'])):
 phase=int(z['trace'][k,1]);cc=z['commands'][k];tick(cc[:14].reshape(2,7),cc[-2:])
assert np.max(abs(q-z['qpos'][-1]))==0
assert max(posemetric()['finger_rms_mm'])<8 and posemetric()['center_mm']<5 and posemetric()['rotation_deg']<5
exec((O/'contact_mpc.py').read_text(),globals())
clib=ctypes.CDLL(str(O.parent/'rubik-sequences-20261015/bridge-contact-observer.dylib'));clib.cube_contacts.argtypes=[ptr,dp,ctypes.c_int];clib.cube_contacts.restype=ctypes.c_int
assert not hasattr(clib,'fingers')
handGeoms=[i for i in range(m.ngeom) if (m.geom_contype[i] or m.geom_conaffinity[i]) and ((m.body(m.geom_bodyid[i]).name or '').startswith('right_hand_') or m.body(m.geom_bodyid[i]).name=='right_wrist_yaw_link')]
cubeGeoms=[i for i in range(m.ngeom) if m.geom_bodyid[i] in cubeids and (m.geom_contype[i] or m.geom_conaffinity[i])];plan=mujoco.MjData(m)
def contact_force():
 buf=np.zeros((256,17));n=clib.cube_contacts(s.p,buf.ravel(),256);b=buf[:n];return b,b[b[:,0]==1,11:14].sum(0) if n else np.zeros(3)
exec((O/'supported_turn.py').read_text(),globals())
seriesResults=[]
def save_series():
 np.savez_compressed(O/'supported-series.npz',trace=trace,qpos=qs,qvel=vs,userdata=states,support=support,commands=commands,actions=actions)
 (O/'supported-series.json').write_text(json.dumps(events,indent=2));(O/'supported-series-control.json').write_text(json.dumps(seriesResults,indent=2))
try:
 for cycle in range(6,11):
  note('begin-cycle',cycle=cycle,**posemetric())
  if not face_turn(1):break
  afterTurn=(d.xpos[cubeids].copy(),R.from_quat(d.xquat[cubeids][:,[1,2,3,0]]))
  exec((O/'wrist_return_control.py').read_text(),globals())
  pr,rr=rigid_residual(afterTurn)
  if distance>0 or np.rad2deg(np.linalg.norm(rr,axis=1)).max()>4:note('regrasp-failed');break
  accepted,history=predictive_return(iterations=35,label='cycle-'+str(cycle))
  pr,rr=rigid_residual(afterTurn);res=[float(np.linalg.norm(pr,axis=1).max()*1000),float(np.rad2deg(np.linalg.norm(rr,axis=1)).max())]
  accepted=accepted and res[0]<.5 and res[1]<3
  seriesResults.append(dict(cycle=cycle,time=elapsed,accepted=bool(accepted),metric=posemetric(),residual=res,history=history));note('cycle-result',cycle=cycle,accepted=bool(accepted),residual=res,**posemetric());save_series()
  if not accepted:break
finally:save_series()
