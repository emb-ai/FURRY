"""Separate-process runtime evaluation; validation is never exposed to PPO."""
import argparse
import json
from pathlib import Path
import numpy as np
from evaluate import evaluate, compare
from rewards import RewardConfig

p=argparse.ArgumentParser()
for name in ['assets','dataset','policy','source','output']:p.add_argument('--'+name,required=True,type=Path)
p.add_argument('--baseline',type=Path)
p.add_argument('--control-hz',type=int,choices=(50,100),default=50)
p.add_argument('--split',choices=('train','validation'),default='validation')
p.add_argument('--stage',type=int)
p.add_argument('--seeds',nargs='+',type=int,default=[0,1])
p.add_argument('--limit',type=int)
p.add_argument('--window-metrics',action='store_true')
a=p.parse_args()
if a.window_metrics:
    # Same observable progress metric at runtime Hz; diagnostic only. No changes
    # to policy inputs, dynamics, rewards, initialization or acceptance criteria.
    import torch
    import evaluate as evaluator
    from window_reward import WindowReward, yaw_xyzw, wrap
    torch.set_num_threads(1)
    OriginalEnv = evaluator.Env
    class WindowMetricEnv(OriginalEnv):
        def window_state(self):
            actual = torch.as_tensor(self.data.qpos[:7].copy(),dtype=torch.float32)[None]
            reference = torch.as_tensor(self.motion['q'][self.k,:7].copy(),dtype=torch.float32)[None]
            return actual[:,:2],yaw_xyzw(actual[:,[4,5,6,3]]),reference[:,:2],yaw_xyzw(reference[:,[4,5,6,3]])
        def reset(self,*args,**kwargs):
            obs=super().reset(*args,**kwargs)
            self.window=WindowReward(1,'cpu',self.dt)
            self.window.reset(torch.tensor([0]),*self.window_state())
            return obs
        def step(self,action):
            obs,reward,done,info=super().step(action)
            self.window.advance(*self.window_state())
            info['window_displacement_error_m']=float(self.window.distance_error)
            info['window_displacement_huber']=float(self.window.position_penalty)
            info['window_turn_huber']=float(self.window.yaw_penalty)
            return obs,reward,done,info
    evaluator.Env=WindowMetricEnv
    evaluator.METRICS.extend(['window_displacement_error_m','window_displacement_huber','window_turn_huber'])
indices=None
if a.stage is not None:
    from env import Motions
    motions=Motions(a.dataset,a.split)
    indices=[i for i,item in enumerate(motions.items) if item['meta']['stage']==a.stage]
    if not indices:raise ValueError('No clips in requested stage/split')
report=evaluate(a.assets,a.dataset,a.policy,a.source,split=a.split,seeds=tuple(a.seeds),output=a.output,reward_config=RewardConfig(slide_weight=2.),clip_indices=indices,limit=a.limit,control_hz=a.control_hz,gait_metrics=True)
if a.baseline:
    base=json.loads(a.baseline.read_text())
    with np.load(a.baseline.with_suffix('.npz'),allow_pickle=False) as z:base['_traces']={k:z[k] for k in z.files}
    verdict=compare(base,report)
    a.output.with_suffix('.comparison.json').write_text(json.dumps(verdict,indent=2))
    print(json.dumps({'phase':'runtime_comparison',**verdict}),flush=True)
