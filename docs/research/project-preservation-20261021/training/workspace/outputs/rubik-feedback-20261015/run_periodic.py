"""Check whether the existing four-turn command block forms a stable periodic cycle."""
exec((O/'fixed_reference.py').read_text(),globals())
source=dict(np.load(PREV/'four-cycles.npz'));bounds=[40.5,59.32,78.17,93.02,117.01];periodicResults=[]
try:
 done=False
 for repetition in range(4):
  for part in range(4):
   phase+=1;before=d.xpos[cubeids].copy();beforeR=R.from_quat(d.xquat[cubeids][:,[1,2,3,0]]);c,r=frame();moving=np.argsort(r.inv().apply(before-c)[:,1])[:4];turn=R.from_rotvec(r.apply([0,np.pi/2,0]));expected=before.copy();expected[moving]=c+turn.apply(before[moving]-c);er=beforeR.as_quat();er[moving]=(turn*beforeR[moving]).as_quat();er=R.from_quat(er)
   for k in range(round(bounds[part]*100),round(bounds[part+1]*100)):
    cc=source['commands'][k];tick(cc[:14].reshape(2,7),cc[-2:])
    if q[cq+2]<.94:done=True;break
   pos=d.xpos[cubeids];fit,_=R.align_vectors(pos-pos.mean(0),expected-expected.mean(0));pr=np.linalg.norm((pos-pos.mean(0))-fit.apply(expected-expected.mean(0)),axis=1);rr=np.rad2deg((R.from_quat(d.xquat[cubeids][:,[1,2,3,0]])*(fit*er).inv()).magnitude());res=[float(pr.max()*1000),float(rr.max())];met=posemetric();ok=not done and res[0]<.5 and res[1]<3
   rec=dict(repetition=repetition,cycle=5+4*repetition+part,time=elapsed,quarter_turn_valid=bool(ok),residual=res,metric=met);periodicResults.append(rec);note('periodic-result',**{k:v for k,v in rec.items() if k!='time'})
   if not ok or met['center_mm']>40 or max(met['finger_rms_mm'])>25:done=True;break
  if done:break
finally:
 np.savez_compressed(O/'periodic.npz',trace=trace,qpos=qs,qvel=vs,userdata=states,support=support,commands=commands,actions=actions)
 (O/'periodic.json').write_text(json.dumps(events,indent=2));(O/'periodic-control.json').write_text(json.dumps(periodicResults,indent=2))
