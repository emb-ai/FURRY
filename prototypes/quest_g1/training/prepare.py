"""Quest capture -> calibrated G1 clips. Private outputs stay outside Git.

Split by whole protocol stage BEFORE chunking. No frame-level random split.
Native GMR does fixed bind-pose anatomy scaling and Y-up -> Z-up conversion.
"""
import argparse,csv,hashlib,json,os,pickle,subprocess,sys
from pathlib import Path
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation,Slerp
from scipy.signal import butter,sosfiltfilt

def read(path):
 with path.open() as f:names=f.readline().strip().split(',')
 a=np.loadtxt(path,delimiter=',',skiprows=1,ndmin=2)
 if not np.isfinite(a).all():raise ValueError(f'Nonfinite {path}')
 return a,{k:i for i,k in enumerate(names)}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def regions(mask):
 ids=np.flatnonzero(mask);return [x for x in np.split(ids,np.flatnonzero(np.diff(ids)>1)+1) if len(x)]
def splits_for_stage(stage,validation):return 'validation' if stage in validation else 'train'
def main():
 p=argparse.ArgumentParser();p.add_argument('episode',type=Path);p.add_argument('output',type=Path);p.add_argument('--assets',type=Path,required=True);p.add_argument('--retarget-bin',type=Path,required=True);p.add_argument('--validation-stages',nargs='+',type=int,default=[2,9,14]);args=p.parse_args()
 sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
 from inspect_capture import inspect_capture
 if not inspect_capture(args.episode)['protocol_completed']:raise ValueError('Capture must be complete and validated')
 out=args.output.resolve();out.mkdir(parents=True,exist_ok=True);(out/'sources').mkdir(exist_ok=True)
 i,ic=read(args.episode/'input.csv');b,bc=read(args.episode/'body.csv');t,tc=read(args.episode/'capture_timeline.csv');assert np.array_equal(i[:,ic['sequence']],b[:,bc['sequence']])
 seq=i[:,ic['sequence']].astype(np.int64);times=i[:,ic['receive_ns']]/1e9;wall=times-times[0];idx=np.searchsorted(seq,t[:,tc['sequence']]);acc=np.diff(t[:,tc['accepted_seconds']],prepend=0)>0
 used=np.zeros(len(i),bool);stage=np.full(len(i),-1);used[idx[acc]]=True;stage[idx[acc]]=t[acc,tc['stage']].astype(int)+1
 # Stage-transition rows are post-tick. Attribute them to the preceding stage.
 for j in np.flatnonzero(acc & (np.diff(t[:,tc['stage']],prepend=t[0,tc['stage']])>0)):
  stage[idx[j]]=int(t[j-1,tc['stage']])+1
 pos=b[:,[bc[f'pose_{j}_{k}'] for j in range(14) for k in range(3)]].reshape(-1,14,3)
 quat=b[:,[bc[f'pose_{j}_{k}'] for j in range(14) for k in range(3,7)]].reshape(-1,14,4)
 quat/=np.linalg.norm(quat,axis=2,keepdims=True)
 dp=np.linalg.norm(np.diff(pos,axis=0),axis=2).max(axis=1);angle=2*np.arccos(np.clip(abs((quat[1:]*quat[:-1]).sum(2)),0,1)).max(axis=1)
 consecutive=used[1:]&used[:-1]&(stage[1:]==stage[:-1])&(np.diff(times)<.05)
 jumps=np.flatnonzero(consecutive&((dp>.25)|(angle>np.pi/4)))+1
 exclusions=[]
 for j in jumps:
  lo,hi=wall[j]-.5,wall[j]+.5;used&=~((wall>=lo)&(wall<=hi));exclusions.append([float(lo),float(hi)])
 for side in ['left','right']:
  used&=(i[:,ic[side+'_flags']].astype(int)&3)==3;used&=i[:,ic[side+'_active']]==1
 used&=b[:,bc['valid']]==1
 cal=[]
 for e in csv.DictReader((args.episode/'events.csv').open()):
  if e['event']=='calibrate':cal.append(int(np.searchsorted(seq,int(e['sequence']))))
 if not cal:raise ValueError('Missing calibration')
 model=mujoco.MjModel.from_xml_path(str(args.assets/'gmr_model.xml'));fk=mujoco.MjData(model)
 physical=mujoco.MjModel.from_xml_path(str(args.assets/'scene.xml'));pd=mujoco.MjData(physical)
 qadr=[physical.joint(model.joint(j).name).qposadr[0] for j in range(1,model.njnt)]
 pads=[physical.geom(f'{side}_foot{j}_collision').id for side in ['left','right'] for j in [1,2,3]]
 assert all(physical.geom_type[g]==mujoco.mjtGeom.mjGEOM_CAPSULE for g in pads)
 links=[model.body(j).name for j in range(1,model.nbody)]
 bodycols=[bc[f'{kind}_{j}_{k}'] for j in range(14) for kind in ['pose','rest'] for k in range(7)]
 headcols=[ic[f'head_{k}'] for k in ['x','y','z','qw','qx','qy','qz']]
 def rawrow(j):return np.r_[seq[j],i[j,ic['xr_time_ns']],b[j,bc['time_ns']],b[j,bc['skeleton_version']],i[j,headcols],b[j,bodycols]]
 manifest={'schema':'quest-g1-motion-v1','source_episode':args.episode.name,'source_sha256':{n:sha(args.episode/n) for n in ['input.csv','body.csv','capture_timeline.csv','capture_plan.json']},'fps':100,'validation_stages':args.validation_stages,'split_unit':'whole protocol stage before chunking','exclusions_wall_seconds':exclusions,'scaling':'native MetaRetargeter: fixed rest-based leg/root and arm scales; camera tracking OFF; sole floor guard ON','filter':'6 Hz zero-phase Butterworth, order 2 on positions and joint angles; quaternion SLERP; floor guard reapplied after smoothing','clips':[]}
 # Run GMR continuously from each calibration, before splitting into clips.
 # Resetting IK at an arbitrary mid-turn frame can choose an invalid branch.
 epoch_data={};manifest['calibrations']=[]
 restcols=[bc[f'rest_{j}_{k}'] for j in range(14) for k in range(7)]
 for cn,cj in enumerate(cal):
  end=cal[cn+1] if cn+1<len(cal) else len(b)
  compatible=np.max(abs(b[cj:end,restcols]-b[cj,restcols]),axis=1)<1e-5
  ej=np.arange(cj,end)[compatible&(b[cj:end,bc['valid']]==1)]
  for lo,hi in exclusions:ej=ej[~((wall[ej]>=lo)&(wall[ej]<=hi))]
  src=out/'sources'/f'epoch{cn}.txt';dst=out/'sources'/f'epoch{cn}.csv'
  with src.open('w') as f:np.savetxt(f,np.vstack([rawrow(cj)]+[rawrow(j) for j in ej]),fmt='%.17g')
  subprocess.run([str(args.retarget_bin.resolve()),str(args.assets.resolve()),str(src),str(dst)],check=True)
  manifest['calibrations'].append(json.loads(Path(str(dst)+'.calibration.json').read_text()))
  ea,eac=read(dst);epoch_data[cj]=(ea,eac)
 for s in range(1,16):
  for runn,ids in enumerate(regions(used&(stage==s))):
   if times[ids[-1]]-times[ids[0]]<2:continue
   cj=max((j for j in cal if j<=ids[0]),default=None)
   if cj is None:raise ValueError('No preceding calibration')
   name=f'stage{s:02d}-run{runn:02d}';src=out/'sources'/f'{name}.txt';dst=out/'sources'/f'{name}.csv'
   ea,ac=epoch_data[cj];ei=np.searchsorted(ea[:,ac['sequence']],seq[ids]);a=ea[ei]
   if not np.array_equal(a[:,ac['sequence']],seq[ids]):raise ValueError('Missing retargeted frame')
   bt=a[:,ac['time_ns']]/1e9;keep=np.r_[True,np.diff(bt)>0];bt=bt[keep];q=a[keep,4:];bt-=bt[0]
   if len(bt)<10:continue
   grid=np.arange(0,bt[-1],.01);new=np.empty((len(grid),36));new[:,:3]=np.column_stack([np.interp(grid,bt,q[:,k]) for k in range(3)]);new[:,7:]=np.column_stack([np.interp(grid,bt,q[:,k]) for k in range(7,36)])
   new[:,3:7]=Slerp(bt,Rotation.from_quat(q[:,[4,5,6,3]]))(grid).as_quat()[:,[3,0,1,2]]
   cols=np.r_[0:3,7:36];new[:,cols]=sosfiltfilt(butter(2,6,fs=100,output='sos'),new[:,cols],axis=0)
   for j in range(1,model.njnt):
    k=model.jnt_qposadr[j];new[:,k]=np.clip(new[:,k],*model.jnt_range[j])
   lifts=[];local=[]
   for row in new:
    pd.qpos[:7]=row[:7];pd.qpos[qadr]=row[7:];mujoco.mj_kinematics(physical,pd)
    low=min(pd.geom_xpos[g,2]-physical.geom_size[g,0]-physical.geom_size[g,1]*abs(pd.geom_xmat[g,8]) for g in pads)
    lift=max(0.,-low);row[2]+=lift;lifts.append(lift);fk.qpos[:]=row;mujoco.mj_kinematics(model,fk);local.append(fk.xpos[1:].copy()-row[:3])
   local=np.asarray(local);split=splits_for_stage(s,args.validation_stages);(out/split).mkdir(exist_ok=True)
   # Split into <=30s pieces; all pieces of a stage remain in its assigned split.
   for start in range(0,len(new)-199,3000):
    stop=min(start+3000,len(new));clipq=new[start:stop].copy();clipq[:,:2]-=clipq[0,:2];clipid=f'{name}-{start:06d}'
    path=out/split/f'{clipid}.npz';np.savez_compressed(path,qpos=clipq,fps=100,local_body_pos=local[start:stop],link_body_list=links)
    payload={'fps':100,'root_pos':clipq[:,:3],'root_rot':clipq[:,[4,5,6,3]],'dof_pos':clipq[:,7:],'local_body_pos':local[start:stop],'link_body_list':links}
    with path.with_suffix('.pkl').open('wb') as f:pickle.dump(payload,f,protocol=4)
    vel=np.diff(clipq[:,7:],axis=0)*100
    manifest['clips'].append({'id':clipid,'file':str(path.relative_to(out)),'split':split,'stage':s,'calibration_sequence':int(seq[cj]),'source_start_wall_s':float(wall[ids[0]]+grid[start]),'source_end_wall_s':float(wall[ids[0]]+grid[stop-1]),'seconds':float((stop-start-1)/100),'max_joint_speed_rad_s':float(abs(vel).max()),'p99_joint_speed_rad_s':float(np.quantile(abs(vel),.99)),'max_floor_lift_m':float(max(lifts[start:stop])),'sha256':sha(path)})
   print(name,split,len(grid),'samples',flush=True)
 manifest['duration_seconds']={s:sum(c['seconds'] for c in manifest['clips'] if c['split']==s) for s in ['train','validation']}
 manifest['validation_fraction']=manifest['duration_seconds']['validation']/sum(manifest['duration_seconds'].values())
 (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
 for split in ['train','validation']:
  lines=['root_path: '+str(out),'motions:']
  for c in manifest['clips']:
   if c['split']==split:lines+=['- file: '+str(Path(c['file']).with_suffix('.pkl')),'  weight: '+str(c['seconds'])]
  (out/f'{split}.yaml').write_text('\n'.join(lines)+'\n')
 print(json.dumps(manifest['duration_seconds']),flush=True)
if __name__=='__main__':main()
