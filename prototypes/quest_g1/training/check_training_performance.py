"""Parity checks against pinned DaggerPPO; run in upstream environment."""
import isaacgym  # noqa: F401
import argparse,copy,json
from types import SimpleNamespace
import torch
from rsl_rl.algorithms import DaggerPPO
from rsl_rl.modules import ActorCriticFuture
from training_performance import critic_only_update,RolloutMetrics
from check_upstream_kl import TinyPolicy


def update_check(device,real,clipped):
 torch.manual_seed(17)
 ac=(ActorCriticFuture(num_observations=1432,num_critic_observations=1743,num_motion_observations=35,num_motion_steps=1,num_priop_observations=92,num_history_steps=10,num_actions=29,num_future_observations=35,num_future_steps=1,actor_hidden_dims=[512,512,256,128],critic_hidden_dims=[256,256,256],activation='silu',layer_norm=True) if real else TinyPolicy()).to(device)
 for p in ac.actor.parameters():p.requires_grad_(False)
 ac.std.requires_grad_(False)
 ac2=copy.deepcopy(ac)
 args=dict(learning_rate=2e-4,schedule='fixed',num_learning_epochs=5,num_mini_batches=4,value_loss_coef=2.,entropy_coef=.005,max_grad_norm=.1,use_clipped_value_loss=clipped,device=device)
 ref=DaggerPPO(None,ac,ac,**args);fast=DaggerPPO(None,ac2,ac2,**args)
 obs_dim,priv_dim,act_dim=(1432,1743,29) if real else (4,4,2)
 ref.init_storage(32,4,[obs_dim],[priv_dim],[act_dim])
 for repeat in range(2):
  with torch.no_grad():
   for _ in range(4):
    obs=torch.randn(32,obs_dim,device=device);priv=torch.randn(32,priv_dim,device=device)
    ref.act(obs,priv,{})
    ref.process_env_step(torch.randn(32,device=device),torch.zeros(32,device=device,dtype=torch.bool),{})
   ref.compute_returns(torch.randn(32,priv_dim,device=device))
  fast.storage=copy.deepcopy(ref.storage)
  torch.manual_seed(51+repeat);a=ref.update()
  torch.manual_seed(51+repeat);b=critic_only_update(fast)
  assert a[0]==b[0],(a[0],b[0])
  for k,v in ac.state_dict().items():
   assert torch.equal(v,ac2.state_dict()[k]),(real,clipped,repeat,k,float((v-ac2.state_dict()[k]).abs().max()))
  os1,os2=ref.optimizer.state_dict(),fast.optimizer.state_dict()
  for k,v in os1['state'].items():
   for name,value in v.items():
    assert torch.equal(value,os2['state'][k][name]),(k,name)
  assert ref.counter==fast.counter and ref.dagger_coef==fast.dagger_coef
 try:
  ac2.std.requires_grad_(True);critic_only_update(fast)
  raise AssertionError('Accepted trainable std')
 except ValueError:pass
 return {'model':'source ActorCriticFuture' if real else 'tiny','clipped':clipped,'updates':40,'parameters_optimizer_and_value_loss_bitwise_equal':True}


def metrics_check(device):
 torch.manual_seed(71);n=4096
 mask=torch.rand(1090,device=device)>.6;ids=mask.nonzero().flatten();metrics=RolloutMetrics(device,mask)
 expected=[0.,0.,0.,0.,0.,0.,0.]
 for i in range(24):
  motions=torch.randint(1090,(n,),device=device);r=torch.randn(n,device=device);d=torch.rand(n,device=device)>.9;t=torch.rand(n,device=device)>.7
  window=SimpleNamespace(position_penalty=torch.rand(n,device=device),yaw_penalty=torch.rand(n,device=device),distance_error=torch.rand(n,device=device)) if i%2 else None
  metrics.record_motion_ids(motions);metrics.record_step(r,d,t,window)
  row=[int(torch.isin(motions,ids).sum()),int(d.sum()),int((d&~t).sum()),float(r.mean()),float(window.position_penalty.mean()) if window else 0.,float(window.yaw_penalty.mean()) if window else 0.,float(window.distance_error.mean()) if window else 0.]
  expected=[a+b for a,b in zip(expected,row)]
 assert list(metrics.result())==expected,(metrics.result(),expected)
 return {'metrics_bitwise_equal':True,'rollout_steps':24,'num_envs':4096}


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--device',default='cpu');p.add_argument('--output');a=p.parse_args();torch.set_num_threads(1)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 report={'device':a.device,'critic_cases':[update_check(a.device,real,clip) for real in (False,True) for clip in (False,True)],'metrics':metrics_check(a.device)}
 print(json.dumps(report))
 if a.output:
  from pathlib import Path
  Path(a.output).write_text(json.dumps(report,indent=2))
