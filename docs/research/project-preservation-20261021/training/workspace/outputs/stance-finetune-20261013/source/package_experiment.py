from pathlib import Path
import sys,json,shutil,hashlib,pickle,subprocess
import numpy as np
R=Path.cwd();O=R/'outputs/stance-finetune-20261013';P=O/'package';P.mkdir(exist_ok=True);C=P/'code';C.mkdir(exist_ok=True)
base=R/'outputs/quest-robust-legs-5k-20261010/adapter'
for p in base.glob('*.py'):shutil.copy2(p,C/p.name)
sys.path.insert(0,str(C));from motion_coordinates import world_offsets_to_root
from author_dataset import validate_motion
old=R/'outputs/quest-finetune-20261009/window-displacement/dataset';mb=(old/'manifest.json').read_bytes();assert hashlib.sha256(mb).hexdigest()=='f6cde2c4f205831ac38c832ab84c1b83d5b3fb5ed6e8ce427cdf7fdeac1803b4';m=json.loads(mb)
(P/'initial').mkdir(exist_ok=True)
for p in (O/'initial').iterdir():
 if p.name in ('checkpoint_000500.pt','config.json','metrics.jsonl','parity.json'):shutil.copy2(p,P/'initial'/p.name)
(P/'initial/manifest.json').write_bytes(mb)
for k in ['A','B']:
 dst=P/(k+'-references')
 if not dst.exists():shutil.copytree(O/(k+'-references'),dst)
# Existing A mixed dataset stays immutable remotely; B contains only modified Quest files.
b=P/'B-dataset';b.mkdir(exist_ok=True);bm=dict(m);bm['clips']=[];qm=json.loads((P/'B-references/manifest.json').read_text());qrows={c['id']:c for c in qm['clips']}
for c in m['clips']:
 cc=dict(c)
 if c['source']=='quest':
  z=np.load(P/'B-references'/qrows[c['id']]['file']);q=z['qpos'];x={'fps':float(z['fps']),'root_pos':q[:,:3],'root_rot':q[:,[4,5,6,3]],'dof_pos':q[:,7:],'local_body_pos':world_offsets_to_root(z['local_body_pos'],q[:,[4,5,6,3]]),'link_body_list':z['link_body_list'].tolist()};validate_motion(x)
  dst=b/c['file'];dst.parent.mkdir(parents=True,exist_ok=True)
  with dst.open('wb') as f:pickle.dump(x,f,protocol=4)
  npz=b/c['npz_file'];shutil.copy2(P/'B-references'/qrows[c['id']]['file'],npz)
  cc.update(sha256=hashlib.sha256(dst.read_bytes()).hexdigest(),npz_sha256=hashlib.sha256(npz.read_bytes()).hexdigest())
 bm['clips'].append(cc)
bm['quest_manifest_sha256']=hashlib.sha256((P/'B-references/manifest.json').read_bytes()).hexdigest();bm['intervention']={'stance_reduction_m':.12,'stage':'post-adapter before GMR SetTargets; feet only, pelvis yaw lateral; same frozen cuts/filter/floor preprocessing'}
(b/'manifest.json').write_text(json.dumps(bm,indent=2))
for split in ['train','validation']:
 # Same ordering and weights, therefore exactly same YAML bytes.
 shutil.copy2(old/(split+'.yaml'),b/(split+'.yaml'))
for c,d in zip(m['clips'],bm['clips']):
 for key in ('id','source','split','stage','parent_id','weight','seconds'):assert c.get(key)==d.get(key)
 if c['source']=='twist2_author':assert c==d
shutil.copy2(R/'.tools/furry-main-integration-20261013/prototypes/quest_g1/policies/quest-111586/checkpoint_000500-50hz.onnx',P/'initial/published.onnx')
plan={'id':'stance-finetune-20261013','initial_checkpoint_sha256':'13e93d837b014317ca1179303bc2b580c1a7940f2a5d868dae20bd36774d3a21','training_seeds':[11,22],'evaluation_seeds':[0,1],'additional_updates':1000,'start_update':500,'final_update':1500,'save_evaluate_updates':[750,1000,1250,1500],'num_envs':4096,'policy_hz':50,'sim_training_hz':500,'physics_eval_hz':1000,'max_concurrent_training':4,'allowed_gpus':[0,1,2,3,4],'reference_delta_m':{'A':0,'B':.12},'window_reward':True,'nonlooping':False,'leg_faults':False,'reward_and_architecture':'identical to111586500; source DaggerPPO unchanged','resume':'actor critic std Adam actor/critic normalization restored; algorithm counters reconstructed from original metrics; physical/history/RNG reset, legacy unsaved motion curriculum reset identically','validation':'same12 source recordings and original time cuts, no train leakage; initial+all saved actors crossed with A/B inputs; common A initial qpos/qvel/noise for both input variants; natural commands derived from each reference','promotion':False,'long_run_gate':'both4096env resume smokes pass exact restored-state checks, full baseline A/B completed, manifests/source verified','queue_timeout_hours':24}
(P/'experiment.json').write_text(json.dumps(plan,indent=2));(O/'data-summary.json').write_text(json.dumps({'originals':45,'train_originals':33,'train_mirrors':33,'validation':12,'author_train_unchanged':1024,'quest_sampling_mass':m['quest_probability_before_curriculum'],'split_preserved':True,'A_arrays_exact_reproduced':True,'A_reference_bytes_unchanged':True,'validation_stages':m['validation_stages']},indent=2))
print(json.dumps(plan,indent=2))
