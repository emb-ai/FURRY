from pathlib import Path
import sys,json
import numpy as np,mujoco
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
APP=Path(__file__).resolve().parents[2];sys.path.insert(0,str(APP));sys.path.insert(0,str(APP/'experiments/homie_sonic'))
from evaluate import Metrics,reference_at,summarize
out=Path(sys.argv[1]);source=[np.load(out/f'{x}.npz') for x in ['off','on']];a=source[1]['source'];raw=a[:,86:184].reshape(-1,14,7);patched=a[:,184:282].reshape(-1,14,7)
assert np.array_equal(source[0]['source'][:,86:184],a[:,86:184]);t=np.arange(len(a))*.01
phase=a[:,2].astype(int);lift=a[:,3:5];active=phase>=0;ids=np.where(active)[0];side=phase[active];core=lift.max(axis=1)>.01
starts=np.where(active&np.r_[True,(phase[1:]!=phase[:-1])])[0]
length=lambda q,i,j:np.linalg.norm(q[:,i,:3]-q[:,j,:3],axis=1)
bones=[(2,4),(4,6),(3,5),(5,7)]
assert np.max([np.abs(length(raw,i,j)-length(patched,i,j)).max() for i,j in bones])<1e-8
upper=[0,1,8,9,10,11,12,13];upper_diff=np.linalg.norm(raw[:,upper,:3]-patched[:,upper,:3],axis=2)
summary={'duration_s':33.5,'episode':'episode-1791652593739690','source':dict(estimated_swing_segments=len(starts),swing_fraction=float(active.mean()),lifted_fraction=float((lift.max(axis=1)>.001).mean()),max_added_height_cm=float(lift.max()*100),mean_added_height_during_detected_swing_cm=float(lift.sum(axis=1)[active].mean()*100),bone_length_max_error_m=float(np.max([np.abs(length(raw,i,j)-length(patched,i,j)).max() for i,j in bones])),upper_position_max_change_m=float(upper_diff.max()),ik_limited_frames=int(a[:,7].sum()))}
stat=lambda v:dict(mean=float(np.mean(v)),median=float(np.median(v)),p10=float(np.percentile(v,10)),p95=float(np.percentile(v,95)))
for name,q in [('off',raw),('on',patched)]:summary['source'][name+'_swing_ankle_above_support_cm']=stat((q[ids,6+side,1]-q[ids,7-side,1])*100)
m=mujoco.MjModel.from_xml_path(str(APP/'android/assets/scene-stand.xml'));d=mujoco.MjData(m);metrics=Metrics(m);qa=metrics.qa
wrists=metrics.upper[-2:];torso=metrics.upper[0];ret=[];up=[]
for s in source:
 gaps=[];positions=[]
 for q in s['qpos']:
  d.qpos[:7]=q[:7];d.qpos[qa]=q[7:];mujoco.mj_forward(m,d)
  gaps.append([min(d.geom_xpos[g,2]-m.geom_size[g,0] for g in gs) for gs in metrics.foot_geoms]);positions.append(d.xpos[[torso,*wrists]].copy())
 ret.append(np.array(gaps));up.append(np.array(positions))
summary['retarget']={name:dict(swing_sole_clearance_cm=stat(gap[ids,side]*100),swing_low_sole_fraction=float(np.mean(gap[ids,side]<.01))) for name,gap in zip(['off','on'],ret)}
summary['retarget']['wrist_target_change_cm']=stat(np.linalg.norm(up[1][:,1:]-up[0][:,1:],axis=2).mean(axis=1)*100)
summary['retarget']['root_target_change_cm']=stat(np.linalg.norm(source[1]['qpos'][:,:3]-source[0]['qpos'][:,:3],axis=1)*100)
summary['physical']={};rollouts=[];common_rows=[]
for variant in ['off','on']:
 r=np.load(out/'rollouts'/f'twist2-{variant}-s11.npz');result=json.loads((out/'rollouts'/f'twist2-{variant}-s11.json').read_text());rows=json.loads((out/'rollouts'/f'twist2-{variant}-s11.rows.json').read_text());metric=Metrics(m)
 for i,state in enumerate(r['state']):
  d.qpos[:]=state[:m.nq];d.qvel[:]=state[m.nq:];d.ctrl[:]=r['ctrl'][i];d.time=(i+1)*.02;mujoco.mj_forward(m,d)
  metric.sample(d,reference_at(source[0]['qpos'],100,d.time))
  # Keep contact metrics measured in the original forward simulation.
  for k in ['contact_slip_m_s','normal_load_n','slide_foot_fraction','low_moving_foot_fraction','braking_contact_events','upward_contact_events','camera_xy_m']:
   metric.rows[-1][k]=rows[i][k]
 n=len(rows);pi=np.minimum(np.rint(np.arange(1,n+1)*2).astype(int),len(phase)-1);ph=phase[pi];ix=np.where(ph>=0)[0];clear=np.array([[x['clearance_left_m'],x['clearance_right_m']] for x in rows])
 z=summarize(metric.rows);z.update(failure=result['failure'],simulated_seconds=result['simulated_seconds'],swing_sole_clearance_cm=stat(clear[ix,ph[ix]]*100),swing_low_sole_fraction=float(np.mean(clear[ix,ph[ix]]<.01)))
 summary['physical'][variant]=z;rollouts.append(r);common_rows.append(metric.rows)
 (out/f'common-{variant}.rows.json').write_text(json.dumps(metric.rows)+'\n')
summary['limitations']=['Single diagnostic trajectory; no independent test or phase ground truth','Support level is opposite raw ankle height, not STAGE floor or sole','Causal estimated swing plus smooth lift; no guaranteed 10cm throughout swing','Same current APK GMR/50Hz TWIST2 +500, common unpatched camera goal, fresh rollouts, 0.5s activation blend','Incoming full-foot hardware retargeting from the historical capture is not replayed; both use current APK 14-point GMR','Contact proxies are not confirmed stumbles; zero does not mean no trips']
(out/'metrics.json').write_text(json.dumps(summary,indent=2)+'\n')
np.savez_compressed(out/'comparison-data.npz',time=t,phase=phase,lift=lift,raw=raw,patched=patched,retarget_gap_off=ret[0],retarget_gap_on=ret[1])
fig,ax=plt.subplots(3,2,figsize=(15,10),sharex=True)
for k in range(2):
 ax[0,k].plot(t,(raw[:,6+k,1]-raw[:,7-k,1])*100,label='Original');ax[0,k].plot(t,(patched[:,6+k,1]-raw[:,7-k,1])*100,label='+10 cm + IK');ax[0,k].axhline(10,color='gray',ls=':');ax[0,k].set_title(('Left' if k==0 else 'Right')+' ankle above opposite ankle (cm)')
 ax[1,k].plot(t,ret[0][:,k]*100,label='Original');ax[1,k].plot(t,ret[1][:,k]*100,label='Patched');ax[1,k].set_title('GMR exact sphere-sole clearance (cm)')
 for variant,rows in zip(['Original','Patched'],common_rows):
  ax[2,k].plot(np.arange(1,len(rows)+1)*.02,[r['clearance_left_m' if k==0 else 'clearance_right_m']*100 for r in rows],label=variant,alpha=.8)
 ax[2,k].set_title('Physical robot exact sphere-sole clearance (cm)');ax[2,k].set_xlabel('Time in evaluated clip (s)')
 for j in range(3):
  ax[j,k].grid(alpha=.2);ax[j,k].legend();ax[j,k].axhline(0,color='black',lw=.6)
  ymin,ymax=ax[j,k].get_ylim();ax[j,k].fill_between(t,ymin,ymax,where=phase==k,color='#c1ddcc',alpha=.2);ax[j,k].set_ylim(ymin,ymax)
fig.suptitle('Virtual Legs ankle lift: source → GMR → physical TWIST2 | shaded = estimated swing',fontsize=15);fig.tight_layout();fig.savefig(out/'clearance-comparison.png',dpi=140);plt.close(fig)
print(json.dumps(summary,indent=2))
