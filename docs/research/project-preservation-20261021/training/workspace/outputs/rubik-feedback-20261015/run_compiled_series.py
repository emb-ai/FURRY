exec((O/'fixed_reference.py').read_text(),globals());exec((O/'compiled_turn_plan.py').read_text(),globals());exec((O/'closure_timing.py').read_text(),globals())
logical=[('R',1),("R'",3),('R',1)];cycle=8;compiledHistory=[];continueRun=True
try:
 for logicalIndex,(command,count) in enumerate(logical):
  note('logical-command-start',command=command,quarter_turns=count)
  for part in range(count):
   cycle+=1;note('compiled-quarter-start',cycle=cycle,logical_command=command,part=part+1)
   bp=d.xpos[cubeids].copy();br=R.from_quat(d.xquat[cubeids][:,[1,2,3,0]]);bc,bR=frame();moving=np.argsort(bR.inv().apply(bp-bc)[:,1])[:4];turn=R.from_rotvec(bR.apply([0,np.pi/2,0]));expected=bp.copy();expected[moving]=bc+turn.apply(bp[moving]-bc);expectedR=br.as_quat();expectedR[moving]=(turn*br[moving]).as_quat();expectedR=R.from_quat(expectedR)
   good,trials=planned_turn(1)
   if not good:continueRun=False;break
   right_released=False;approachSource=(O/'height_aware_return.py').read_text();exec(approachSource[:approachSource.index('if distance==0:')],globals())
   if distance>0:continueRun=False;break
   good,timing=close_fixed_home();met=posemetric();pos=d.xpos[cubeids];fit,_=R.align_vectors(pos-pos.mean(0),expected-expected.mean(0));pr=(pos-pos.mean(0))-fit.apply(expected-expected.mean(0));rr=(R.from_quat(d.xquat[cubeids][:,[1,2,3,0]])*(fit*expectedR).inv()).magnitude();res=[float(np.linalg.norm(pr,axis=1).max()*1000),float(np.rad2deg(rr).max())]
   accepted=good and res[0]<.5 and res[1]<3 and met['center_mm']<8 and met['rotation_deg']<6 and max(met['finger_rms_mm'])<8 and max(met['finger_orientation_rms_deg'])<6
   rec=dict(cycle=cycle,time=elapsed,logical_command=command,part=part+1,accepted=bool(accepted),metric=met,quarter_residual=res,turn_trials=trials,timing_trials=timing);compiledHistory.append(rec);note('compiled-quarter-end',cycle=cycle,accepted=bool(accepted),quarter_residual=res,**met)
   (O/'compiled-progress.json').write_text(json.dumps(compiledHistory,indent=2))
   if not accepted:continueRun=False;break
   anchors['compiled-last']=capture()
  if not continueRun:break
  note('logical-command-complete',command=command)
finally:
 np.savez_compressed(O/'compiled-series.npz',trace=trace,qpos=qs,qvel=vs,userdata=states,support=support,commands=commands,actions=actions)
 (O/'compiled-series.json').write_text(json.dumps(events,indent=2));(O/'compiled-series-control.json').write_text(json.dumps(compiledHistory,indent=2))
