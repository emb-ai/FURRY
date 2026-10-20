from pathlib import Path
import sys,json,hashlib,shutil
import numpy as np
import mujoco
R=Path.cwd();O=R/'outputs/stance-finetune-20261013';T=R/'.tools/furry-training/prototypes/quest_g1/training';sys.path.insert(0,str(T));sys.path.insert(0,str(T.parent))
from mirror_dataset import augment
from motion_coordinates import world_offsets_to_root
old=R/'outputs/quest-finetune-20261009/window-displacement/clamp-dataset';oldmirror=old.parent/'clamp-mirrored';manifest=json.loads((old/'manifest.json').read_text());pre={k:json.loads((O/(k+'-prepared/manifest.json')).read_text()) for k in ['A','B']}
parents={k:{c['id']:c for c in m['clips']} for k,m in pre.items()};audit=[]
for k in ['A','B']:
 dest=O/(k+'-frozen');dest.mkdir(exist_ok=False);out=dict(manifest);out['clips']=[]
 for c in manifest['clips']:
  parent=c['id'].rsplit('-part',1)[0];pc=parents[k][parent];start=round((c['source_start_wall_s']-pc['source_start_wall_s'])*100);z=np.load(old/c['file']);n=len(z['qpos']);b=np.load(O/(k+'-prepared')/pc['file']);q=b['qpos'][start:start+n].copy();q[:,:2]-=q[0,:2];assert q.shape==z['qpos'].shape
  if k=='A':
   err=float(abs(q-z['qpos']).max());assert err<1e-10,(c['id'],err)
  p=dest/c['file'];p.parent.mkdir(exist_ok=True)
  if k=='A':shutil.copy2(old/c['file'],p)
  else:np.savez_compressed(p,qpos=q,fps=100,local_body_pos=b['local_body_pos'][start:start+n],link_body_list=b['link_body_list'])
  cc=dict(c,sha256=hashlib.sha256(p.read_bytes()).hexdigest());out['clips'].append(cc);audit.append({'branch':k,'id':c['id'],'frames':n,'old_qpos_max_diff':float(abs(q-z['qpos']).max()),'split':c['split'],'source_start_wall_s':c['source_start_wall_s']})
 out['stance_reduction_m']=0 if k=='A' else .12;(dest/'manifest.json').write_text(json.dumps(out,indent=2))
 augment(dest,O/(k+'-references'),R/'outputs/quest-finetune-20261008/assets')
# Control retains exact old files, including mirrors. Validate regenerated mirrors first.
a=O/'A-references';am=json.loads((a/'manifest.json').read_text());om=json.loads((oldmirror/'manifest.json').read_text());oc={c['id']:c for c in om['clips']}
for c in am['clips']:
 z=np.load(a/c['file']);oldz=np.load(oldmirror/oc[c['id']]['file']);assert all(np.array_equal(z[key],oldz[key]) for key in z.files),c['id']
 shutil.copy2(oldmirror/oc[c['id']]['file'],a/c['file'])
shutil.copy2(oldmirror/'manifest.json',a/'manifest.json')
(O/'split-replay-audit.json').write_text(json.dumps(audit,indent=2));print('45 original clips + 33 train mirrors; control arrays exact, old bytes preserved')
# All-frame FK/coordinate/limit/velocity validation. Keep derivatives as auditable sidecars.
model=mujoco.MjModel.from_xml_path(str(R/'outputs/quest-finetune-20261008/assets/gmr_model.xml'));d=mujoco.MjData(model);report=[]
for k in ['A','B']:
 base=O/(k+'-references');m=json.loads((base/'manifest.json').read_text());der=O/(k+'-derivatives');der.mkdir()
 for c in m['clips']:
  z=np.load(base/c['file']);q=z['qpos'];n=len(q);assert np.isfinite(q).all();assert np.max(abs(np.linalg.norm(q[:,3:7],axis=1)-1))<1e-10
  for j in range(1,model.njnt):
   lo,hi=model.jnt_range[j];v=q[:,model.jnt_qposadr[j]];assert v.min()>=lo-1e-5 and v.max()<=hi+1e-5,(k,c['id'],j)
  local=world_offsets_to_root(z['local_body_pos'],q[:,[4,5,6,3]]);maxfk=0.;world=[];qvel=[]
  for i,row in enumerate(q):
   d.qpos[:]=row;mujoco.mj_kinematics(model,d);wp=d.xpos[1:].copy();world.append(wp);recovered=(d.xmat[1].reshape(3,3)@local[i].T).T+row[:3];maxfk=max(maxfk,float(abs(wp-recovered).max()))
   v=np.zeros(model.nv);mujoco.mj_differentiatePos(model,v,.01,q[max(i-1,0)],row);qvel.append(v)
  assert maxfk<1e-10,(k,c['id'],maxfk)
  world=np.array(world);bv=np.diff(world,axis=0,prepend=world[:1])*100;qvel=np.array(qvel);assert np.isfinite(bv).all() and np.isfinite(qvel).all();np.savez_compressed(der/(c['id']+'.npz'),qvel=qvel,body_world_velocity=bv)
  report.append({'branch':k,'id':c['id'],'fk_root_frame_max_error_m':maxfk,'joint_limit_check':True,'max_dof_velocity_rad_s':float(abs(qvel[:,6:]).max()),'velocity_convention':'backward difference 100Hz; initial sample zero; MuJoCo root angular velocity local; body velocities world. Training MotionLib independently recomputes its original derivative/smoothing.'})
(O/'fk-velocity-audit.json').write_text(json.dumps(report,indent=2));print('FK/coordinate/limits/derivatives passed',len(report))
