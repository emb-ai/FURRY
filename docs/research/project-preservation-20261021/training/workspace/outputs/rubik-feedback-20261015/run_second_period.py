exec((O/'fixed_reference.py').read_text(),globals())
source=dict(np.load(PREV/'four-cycles.npz'));periodResults=[]
try:
 for cycle in range(5,13):
  phase+=1;move(source['commands'][5931,:14].reshape(2,7),2.,gr=[1,1]);move(goals,1.)
  before=d.xpos[cubeids].copy();beforeR=R.from_quat(d.xquat[cubeids][:,[1,2,3,0]]);c,r=frame();moving=np.argsort(r.inv().apply(before-c)[:,1])[:4];turn=R.from_rotvec(r.apply([0,np.pi/2,0]));expected=before.copy();expected[moving]=c+turn.apply(before[moving]-c);er=beforeR.as_quat();er[moving]=(turn*beforeR[moving]).as_quat();er=R.from_quat(er)
  for k in range(5932,7817):
   cc=source['commands'][k];tick(cc[:14].reshape(2,7),cc[-2:])
   if q[cq+2]<.94:break
  pos=d.xpos[cubeids];fit,_=R.align_vectors(pos-pos.mean(0),expected-expected.mean(0));pr=np.linalg.norm((pos-pos.mean(0))-fit.apply(expected-expected.mean(0)),axis=1);rr=np.rad2deg((R.from_quat(d.xquat[cubeids][:,[1,2,3,0]])*(fit*er).inv()).magnitude());res=[float(pr.max()*1000),float(rr.max())];met=posemetric();ok=q[cq+2]>.94 and res[0]<.5 and res[1]<3
  record=dict(cycle=cycle,time=elapsed,valid=bool(ok),residual=res,metric=met);periodResults.append(record);note('second-period-result',**{k:v for k,v in record.items() if k!='time'})
  if not ok or met['center_mm']>30 or max(met['finger_rms_mm'])>15:break
finally:
 np.savez_compressed(O/'second-period.npz',trace=trace,qpos=qs,qvel=vs,userdata=states,support=support,commands=commands,actions=actions)
 (O/'second-period.json').write_text(json.dumps(events,indent=2));(O/'second-period-control.json').write_text(json.dumps(periodResults,indent=2))
