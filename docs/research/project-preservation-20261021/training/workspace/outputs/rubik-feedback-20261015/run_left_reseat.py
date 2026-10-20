"""Restore the holding grasp with a low scalar Grip and wrist feedback."""
reference=(d.xpos[cubeids].copy(),R.from_quat(d.xquat[cubeids][:,[1,2,3,0]]));phase+=1
move(goals,.5,gr=[.2,1]);note('left-reseat-relaxed',**posemetric())
baseZ=q[cq+2];logs=[]
for k in range(600):
 c,r=frame();actual=s.poses()[0];g=goals.copy();want=c+r.apply(homeLocal[0,:3]);error=want-actual[:3];v=1.2*error;v*=min(1,.012/max(np.linalg.norm(v),1e-12));g[0,:3]+=.01*v
 desired=r*R.from_quat(homeLocal[0,[4,5,6,3]]);err=(desired*R.from_quat(actual[[4,5,6,3]]).inv()).as_rotvec();omega=.8*err;omega*=min(1,np.deg2rad(8)/max(np.linalg.norm(omega),1e-12));g[0,3:]=(R.from_rotvec(.01*omega)*R.from_quat(g[0,[4,5,6,3]])).as_quat()[[3,0,1,2]]
 tick(g,[.2,1])
 if k%100==0:note('left-reseat-progress',wrist_error_mm=float(np.linalg.norm(error)*1000),angle_error_deg=float(np.rad2deg(np.linalg.norm(err))),**posemetric())
 if q[cq+2]<.95:break
 if np.linalg.norm(error)<.003 and np.linalg.norm(err)<np.deg2rad(3):break
move(goals,1.,gr=[1,1]);move(goals,1.)
pr,rr=rigid_residual(reference);res=[float(np.linalg.norm(pr,axis=1).max()*1000),float(np.rad2deg(np.linalg.norm(rr,axis=1)).max())]
note('left-reseated',residual=res,**posemetric());anchors['reseated5']=capture()
np.savez_compressed(O/'left-reseated-five.npz',trace=trace,qpos=qs,qvel=vs,userdata=states,support=support,commands=commands,actions=actions)
(O/'left-reseated-five.json').write_text(json.dumps(events,indent=2))
