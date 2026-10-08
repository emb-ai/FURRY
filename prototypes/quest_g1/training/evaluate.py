"""Full-clip fixed-seed baseline/candidate evaluation; never used for gradients."""
import argparse,json,time,hashlib
from pathlib import Path
import numpy as np
import onnxruntime as ort
from env import Motions,Env

def evaluate(assets,dataset,policy,source=None,split='validation',seeds=(0,),output=None,limit=None):
 motions=Motions(dataset,split);env=Env(assets,motions);env.max_steps=10000000
 if str(policy).endswith('.onnx'):
  opts=ort.SessionOptions();opts.intra_op_num_threads=1;opts.inter_op_num_threads=1;s=ort.InferenceSession(str(policy),sess_options=opts,providers=['CPUExecutionProvider']);name=s.get_inputs()[0].name
  def act(obs):return s.run(None,{name:obs[None]})[0][0]
 else:
  import torch
  from policy import load
  torch.set_num_threads(1);p=load(policy,source)
  def act(obs):
   with torch.no_grad():return p(torch.from_numpy(obs[None])).numpy()[0]
 results=[]
 for clip in range(len(motions.items) if limit is None else min(limit,len(motions.items))):
  for seed in seeds:
   env.rng=np.random.default_rng(seed);obs=env.reset(clip=clip,start=0,randomize=seed!=0);rows=[]
   while True:
    obs,_,done,info=env.step(act(obs));rows.append(info)
    if done:break
   result={'clip':motions.items[clip]['meta']['id'],'stage':motions.items[clip]['meta']['stage'],'seed':seed,'reference_seconds':(len(motions.items[clip]['q'])-1)/100,'survived_seconds':info['seconds'],'fell':info['fall']}
   for k in ['joint_rmse','key_rmse_m','slip_m_s','stumble','root_xy_error_m','tilt_deg']:result[k+'_mean']=float(np.mean([x[k] for x in rows]));result[k+'_p95']=float(np.percentile([x[k] for x in rows],95))
   results.append(result);print(json.dumps(result),flush=True)
 report={'policy_sha256':hashlib.sha256(Path(policy).read_bytes()).hexdigest(),'dataset_manifest_sha256':hashlib.sha256((Path(dataset)/'manifest.json').read_bytes()).hexdigest(),'split':split,'initialization':'reference qpos/qvel, zero history; seeds>0 add 0.005 rad joint noise','control_hz':100,'physics_hz':1000,'trials':results,'completion_fraction':float(np.mean([not x['fell'] for x in results])),'falls':sum(x['fell'] for x in results),'total_trials':len(results),'survived_seconds':sum(x['survived_seconds'] for x in results),'reference_seconds':sum(x['reference_seconds'] for x in results)}
 if output:Path(output).write_text(json.dumps(report,indent=2))
 return report
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--assets',required=True);p.add_argument('--dataset',required=True);p.add_argument('--policy',required=True);p.add_argument('--source');p.add_argument('--split',default='validation');p.add_argument('--seeds',nargs='+',type=int,default=[0]);p.add_argument('--output',required=True);p.add_argument('--limit',type=int);a=p.parse_args();evaluate(a.assets,a.dataset,a.policy,a.source,a.split,a.seeds,a.output,a.limit)
