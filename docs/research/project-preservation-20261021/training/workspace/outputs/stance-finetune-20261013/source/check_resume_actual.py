"""Real source model/PPO/Adam continuation on identical synthetic CPU rollouts."""
import copy,json
from pathlib import Path
from types import SimpleNamespace as NS
import torch
from rsl_rl.modules import ActorCriticFuture
from rsl_rl.algorithms import DaggerPPO
from resume_state import restore
r=Path('/home/jovyan/vlad/FURRY/twist2-distributed-20261010/stance-finetune-20261013')
p=r/'initial/checkpoint_000500.pt'
c=json.loads((p.parent/'config.json').read_text());ck=torch.load(p,map_location='cpu',weights_only=True)
e=c['environment']['env'];torch.set_num_threads(1)
def construct():
 ac=ActorCriticFuture(num_observations=e['num_observations'],num_critic_observations=ck['critic_mean'].numel(),num_motion_observations=e['n_mimic_obs'],num_motion_steps=len(e['tar_motion_steps']),num_priop_observations=e['n_proprio'],num_history_steps=e['history_len'],num_actions=e['num_actions'],**c['upstream_training']['policy'])
 alg=DaggerPPO(NS(),ac,ac,teacher_loaded=False,device='cpu',**c['upstream_training']['algorithm'])
 return ac,alg
ac,alg=construct()
env=NS(device='cpu',total_env_steps_counter=0,global_counter=0,common_step_counter=0,delay=torch.tensor(0.));cfg=NS(noise=NS(noise_increasing_steps=50000),domain_rand=NS(action_delay=True,action_buf_len=8))
args=NS(**{k:c['finetune_overrides'].get(k,False) for k in ('leg_faults','window_reward','nonlooping','fast_critic')})
cm,cd,start,counter,report=restore(ck,p,ac,alg,env,cfg,args)
def same(x,y):
 if isinstance(x,torch.Tensor):return isinstance(y,torch.Tensor) and torch.equal(x,y)
 if isinstance(x,dict):return x.keys()==y.keys() and all(same(x[k],y[k]) for k in x)
 if isinstance(x,(list,tuple)):return len(x)==len(y) and all(same(v,w) for v,w in zip(x,y))
 return x==y
assert same(ac.state_dict(),ck['actor_critic']);assert same(alg.optimizer.state_dict(),ck['optimizer'])
assert torch.equal(cm,ck['critic_mean']) and torch.equal(cd,ck['critic_divisor'])
assert start==500 and counter==1050 and report['unsaved_environment_schedule_reset']
# Independent direct restore; the exact next source PPO update must match.
other,reference=construct();other.load_state_dict(ck['actor_critic']);reference.optimizer.load_state_dict(copy.deepcopy(ck['optimizer']))
reference.learning_rate=alg.learning_rate;reference.counter=counter;reference.dagger_coef=alg.dagger_coef
for instance in (alg,reference):
 torch.manual_seed(182);instance.init_storage(32,4,[e['num_observations']],[cm.numel()],[e['num_actions']])
 for _ in range(4):
  obs=torch.randn(32,e['num_observations'])*.1;priv=torch.randn(32,cm.numel())*.1
  instance.act(obs,priv,{})
  instance.process_env_step(torch.rand(32)*.1,torch.zeros(32,dtype=torch.bool),{})
 instance.compute_returns(torch.randn(32,cm.numel())*.1)
 torch.manual_seed(287);instance.update()
assert same(ac.state_dict(),other.state_dict()),'Next source PPO update changed on resume'
assert same(alg.optimizer.state_dict(),reference.optimizer.state_dict())
assert alg.counter==counter+1
bad=copy.copy(args);bad.leg_faults=True
try:restore(ck,p,ac,alg,env,cfg,bad)
except ValueError:pass
else:raise AssertionError('Mismatched condition accepted')
report.update(passed=True,bitwise_model_normalization_adam_restore=True,next_source_ppo_update_bitwise_equal=True,negative_condition_check=True,gpu_rollout_check='queued separately')
(r/'cpu-resume-check.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
