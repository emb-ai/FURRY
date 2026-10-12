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
from evaluation_contract import contract, digest
eval_contract=contract(a.assets,a.dataset,a.source,a.control_hz,a.seeds,a.split,a.stage,a.limit,a.window_metrics)
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
# Upper-body diagnostic uses the same state/reference; no dynamics or inputs changed.
import evaluate as evaluator
from scipy.spatial.transform import Rotation
PreviousEnv=evaluator.Env
class UpperMetricEnv(PreviousEnv):
    def step(self,action):
        obs,reward,done,info=super().step(action)
        q=self.motion['q'][self.k]
        dq=self.data.qpos[self.qa]-q[7:]
        info['upper_joint_rmse']=float(np.sqrt(np.mean(dq[12:]**2)))
        actual=Rotation.from_euler('z',np.arctan2(2*(self.data.qpos[3]*self.data.qpos[6]+self.data.qpos[4]*self.data.qpos[5]),1-2*(self.data.qpos[5]**2+self.data.qpos[6]**2))).inv().apply(self.data.xpos[self.key]-self.data.qpos[:3])
        reference=Rotation.from_euler('z',np.arctan2(2*(q[3]*q[6]+q[4]*q[5]),1-2*(q[5]**2+q[6]**2))).inv().apply(self.ref.xpos[self.key]-q[:3])
        # Runtime markers: wrists, elbows and torso; these are not anatomical head markers.
        import mujoco
        names=[mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_BODY,int(i)) for i in self.key]
        selected=[i for i,n in enumerate(names) if 'wrist' in n or 'rubber_hand' in n or 'elbow' in n or 'head' in n or n=='torso_link']
        if len(selected)!=5:raise ValueError('Unexpected upper-body marker contract: '+str(names))
        info['upper_key_rmse_m']=float(np.sqrt(np.mean((actual[selected]-reference[selected])**2)))
        return obs,reward,done,info

evaluator.Env=UpperMetricEnv
evaluator.METRICS.extend(['upper_joint_rmse','upper_key_rmse_m'])
from clearance_evaluation import ClearanceRecorder, METRICS as CLEARANCE_METRICS
PreviousEnv=evaluator.Env
class ClearanceMetricEnv(PreviousEnv):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.clearance_recorder=ClearanceRecorder(self)
    def reset(self,*args,**kwargs):
        obs=super().reset(*args,**kwargs);self.clearance_recorder.reset();return obs
    def step(self,action):
        obs,reward,done,info=super().step(action);info.update(self.clearance_recorder.record());return obs,reward,done,info
evaluator.Env=ClearanceMetricEnv
evaluator.METRICS.extend(CLEARANCE_METRICS)
indices=None
if a.stage is not None:
    from env import Motions
    motions=Motions(a.dataset,a.split)
    indices=[i for i,item in enumerate(motions.items) if item['meta']['stage']==a.stage]
    if not indices:raise ValueError('No clips in requested stage/split')
report=evaluate(a.assets,a.dataset,a.policy,a.source,split=a.split,seeds=tuple(a.seeds),output=a.output,reward_config=RewardConfig(slide_weight=2.),clip_indices=indices,limit=a.limit,control_hz=a.control_hz,gait_metrics=True)
a.output.with_suffix('.fingerprint.json').write_text(json.dumps(dict(contract=eval_contract,policy_sha256=digest(a.policy)),indent=2))
if a.baseline:
    base=json.loads(a.baseline.read_text())
    with np.load(a.baseline.with_suffix('.npz'),allow_pickle=False) as z:base['_traces']={k:z[k] for k in z.files}
    verdict=compare(base,report)
    from clearance_evaluation import summarize_v2
    v2=summarize_v2(verdict)
    a.output.with_suffix('.clearance-verdict.json').write_text(json.dumps(v2,indent=2))
    a.output.with_suffix('.comparison.json').write_text(json.dumps(verdict,indent=2))
    print(json.dumps({'phase':'runtime_comparison_v2','accepted':v2['accepted'],'reasons':v2['reasons'],'legacy_accepted':verdict['accepted']}),flush=True)
