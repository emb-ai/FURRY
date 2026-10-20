"""One fixed reference: the settled grasp immediately before the first successful turn."""
referenceTime=40.5
ref=dict(np.load(O.parent/'rubik-regrasp-20261015/four-cycles.npz'));index=round(referenceTime*100)-1;referenceData=mujoco.MjData(m);referenceData.qpos[:]=ref['qpos'][index];mujoco.mj_kinematics(m,referenceData)
choices=R.from_quat(referenceData.qpos[cq+3:cq+7][[1,2,3,0]])*cubeGroup;referenceRotation=choices[np.argmin((homeR.inv()*choices).magnitude())]
homeC=referenceData.xpos[cubeids].mean(0);homeR=referenceRotation;homeG=ref['commands'][index,:14].reshape(2,7).copy();homeP=np.array([np.r_[referenceData.xpos[b],referenceData.xquat[b]] for b in s.wr]);homeF=[homeR.inv().apply(referenceData.xpos[ids]-homeC) for ids in handids];homeFrot=[homeR.inv()*R.from_quat(referenceData.xquat[ids][:,[1,2,3,0]]) for ids in handids];homeLocal=homeP.copy()
for h in range(2):homeLocal[h,:3]=homeR.inv().apply(homeP[h,:3]-homeC);homeLocal[h,3:]=(homeR.inv()*R.from_quat(homeP[h,[4,5,6,3]])).as_quat()[[3,0,1,2]]
legacyPosemetric=globals().get('legacyPosemetric',posemetric)
def finger_rotation_errors():
 c,r=frame();return [(r.inv()*R.from_quat(d.xquat[handids[h]][:,[1,2,3,0]])*homeFrot[h].inv()).as_rotvec() for h in range(2)]
def posemetric():
 metric=legacyPosemetric();ee=finger_rotation_errors();metric['finger_orientation_rms_deg']=[float(np.rad2deg(np.sqrt(np.mean(np.sum(e*e,axis=1))))) for e in ee];metric['finger_orientation_max_deg']=[float(np.rad2deg(np.linalg.norm(e,axis=1)).max()) for e in ee];return metric
note('fixed-reference',reference_time=referenceTime,**posemetric())
