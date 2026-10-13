"""Full-clip evaluation and paired-prefix baseline acceptance; no gradients."""
import argparse,json,hashlib
from pathlib import Path
from dataclasses import asdict
import numpy as np
import onnxruntime as ort
from env import Motions,Env
from rewards import RewardConfig

METRICS=['joint_rmse','key_rmse_m','slip_m_s','slide_rate','stumble','root_xy_error_m','tilt_deg','reward_rate']

def evaluate(assets,dataset,policy,source=None,split='validation',seeds=(0,),output=None,limit=None,reward_config=None,stop=None,clip_indices=None,control_hz=100,gait_metrics=False):
 from gait_evaluation import runtime_env_class, RAW_COLUMNS, summarize_gait
 config=reward_config or RewardConfig();motions=Motions(dataset,split);env=runtime_env_class(Env,control_hz,gait_metrics)(assets,motions,reward_config=config);env.max_steps=10000000
 columns=list(METRICS)+(RAW_COLUMNS if gait_metrics else [])
 if str(policy).endswith('.onnx'):
  opts=ort.SessionOptions();opts.intra_op_num_threads=1;opts.inter_op_num_threads=1;s=ort.InferenceSession(str(policy),sess_options=opts,providers=['CPUExecutionProvider']);name=s.get_inputs()[0].name
  def act(obs):return s.run(None,{name:obs[None]})[0][0]
 else:
  import torch
  from policy import load
  torch.set_num_threads(1);p=load(policy,source)
  def act(obs):
   with torch.no_grad():return p(torch.from_numpy(obs[None])).numpy()[0]
 results=[];traces={}
 indices=clip_indices if clip_indices is not None else range(len(motions.items) if limit is None else min(limit,len(motions.items)))
 for clip in indices:
  for seed in seeds:
   env.rng=np.random.default_rng(seed);obs=env.reset(clip=clip,start=0,randomize=seed!=0);rows=[]
   while True:
    if stop is not None and env.steps%100==0 and stop():raise InterruptedError('Stopped during evaluation rollout')
    if env.steps%env.policy_stride==0:action=act(obs)
    obs,_,done,info=env.step(action);rows.append([info[k] for k in columns])
    if done:break
   key=f"{motions.items[clip]['meta']['id']}__seed{seed}";arr=np.array(rows);traces[key]=arr
   result={'clip':motions.items[clip]['meta']['id'],'stage':motions.items[clip]['meta']['stage'],'seed':seed,'trace_key':key,'reference_seconds':(len(motions.items[clip]['q'])-1)/100,'survived_seconds':info['seconds'],'fell':info['fall'],'slide_path_m':info['slide_path_m']}
   for j,k in enumerate(METRICS):result[k+'_mean']=float(arr[:,j].mean());result[k+'_p95']=float(np.percentile(arr[:,j],95))
   if gait_metrics:result['gait']=summarize_gait(arr,columns)
   results.append(result);print(json.dumps({'phase':'evaluation','split':split,**result}),flush=True)
 report={'policy_sha256':hashlib.sha256(Path(policy).read_bytes()).hexdigest(),'dataset_manifest_sha256':hashlib.sha256((Path(dataset)/'manifest.json').read_bytes()).hexdigest(),'reward_config':asdict(config),'split':split,'initialization':'reference qpos/qvel, zero history; seeds>0 add 0.005 rad joint noise','control_hz':control_hz,'physics_hz':1000,'trace_hz':100,'history_seconds':10/control_hz,'scene_sha256':hashlib.sha256((Path(assets)/'scene.xml').read_bytes()).hexdigest(),'gait_schema':'fixed-qpos-reference-v1' if gait_metrics else None,'trace_columns':columns,'trials':results,'completion_fraction':float(np.mean([not x['fell'] for x in results])),'falls':sum(x['fell'] for x in results),'total_trials':len(results),'survived_seconds':sum(x['survived_seconds'] for x in results),'reference_seconds':sum(x['reference_seconds'] for x in results)}
 if output:
  output=Path(output);trace_path=output.with_suffix('.npz');np.savez_compressed(trace_path,**traces);report['trace_file']=trace_path.name;output.write_text(json.dumps(report,indent=2))
 report['_traces']=traces
 return report


def compare(baseline,candidate,tracking_tolerance=.05,slip_tolerance=.02,min_improvement=.02):
 if baseline['dataset_manifest_sha256']!=candidate['dataset_manifest_sha256']:raise ValueError('Different evaluation datasets')
 if baseline['trace_columns']!=candidate['trace_columns']:raise ValueError('Different metrics')
 for key in ('control_hz','trace_hz','scene_sha256','gait_schema','split','reward_config'):
  if baseline.get(key)!=candidate.get(key):raise ValueError('Different evaluation configuration: '+key)
 columns=baseline['trace_columns']
 from gait_evaluation import summarize_gait
 gait_pairs=[]
 b={r['trace_key']:r for r in baseline['trials']};c={r['trace_key']:r for r in candidate['trials']}
 if b.keys()!=c.keys():raise ValueError('Different clips or seeds')
 sums={p:np.zeros(len(columns)) for p in ['baseline','candidate']};count=0
 for key in b:
  n=min(len(baseline['_traces'][key]),len(candidate['_traces'][key]));start=min(30,n-1)
  for name,report in [('baseline',baseline),('candidate',candidate)]:sums[name]+=report['_traces'][key][start:n].sum(0)
  count+=n-start
  if baseline.get('gait_schema'):
   gait_pairs.append({'trace_key':key,'stage':b[key]['stage'],'prefix_seconds':n/100,**{name:summarize_gait(report['_traces'][key][:n],columns) for name,report in [('baseline',baseline),('candidate',candidate)]}})
 means={name:{k:float(x) for k,x in zip(columns,v/count) if not k.startswith('gait_')} for name,v in sums.items()}
 bm=means['baseline'];cm=means['candidate'];new_falls=[k for k in b if c[k]['fell'] and not b[k]['fell']]
 reasons=[]
 if new_falls:reasons.append('new_falls_on_baseline_successes')
 if candidate['falls']>baseline['falls']:reasons.append('more_falls')
 if candidate['survived_seconds']+.01<baseline['survived_seconds']*.99:reasons.append('less_survival')
 for k in ['joint_rmse','key_rmse_m','root_xy_error_m']:
  if cm[k]>bm[k]*(1+tracking_tolerance)+1e-4:reasons.append(k+'_regression')
 if cm['slide_rate']>bm['slide_rate']*(1+slip_tolerance)+1e-4:reasons.append('sliding_regression')
 # Survival alone must not select a policy that merely stands still.
 improved=(candidate['falls']<baseline['falls'] or cm['slide_rate']<bm['slide_rate']*(1-min_improvement)-1e-4 or cm['joint_rmse']<bm['joint_rmse']*(1-min_improvement)-1e-4)
 return {'non_regressing':not reasons,'accepted':not reasons and improved,'improved':improved,'reasons':reasons,'new_falls':new_falls,'matched_prefix_means':means,'paired_gait':gait_pairs,'paired_samples':count,'tolerances':{'tracking':tracking_tolerance,'sliding':slip_tolerance,'minimum_improvement':min_improvement}}

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--assets',required=True);p.add_argument('--dataset',required=True);p.add_argument('--policy',required=True);p.add_argument('--source');p.add_argument('--split',default='validation');p.add_argument('--seeds',nargs='+',type=int,default=[0]);p.add_argument('--output',required=True);p.add_argument('--limit',type=int);p.add_argument('--slide-weight',type=float,default=.5);p.add_argument('--legacy-reward',action='store_true');a=p.parse_args();evaluate(a.assets,a.dataset,a.policy,a.source,a.split,a.seeds,a.output,a.limit,RewardConfig(slide_weight=a.slide_weight,legacy=a.legacy_reward))
