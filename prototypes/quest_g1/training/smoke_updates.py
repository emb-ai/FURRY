"""Real-policy, real-physics CPU/GPU smoke: separate optimizers and frozen teacher.

Not a learning-quality test; writes no trained policy. Uses training motions only.
"""
import argparse,json
import numpy as np,torch
from torch import nn
from torch.distributions import Normal
from env import Env,Motions
from policy import load
from critic import Critic
from ppo_utils import advantages
p=argparse.ArgumentParser();p.add_argument('--assets',required=True);p.add_argument('--dataset',required=True);p.add_argument('--initial',required=True);p.add_argument('--source',required=True);p.add_argument('--device',default='cpu');a=p.parse_args()
torch.set_num_threads(1);torch.manual_seed(123)
actor=load(a.initial,a.source,a.device);teacher=load(a.initial,a.source,a.device);teacher.requires_grad_(False)
initial={k:v.clone() for k,v in actor.state_dict().items()};e=Env(a.assets,Motions(a.dataset,'train'),123);o=e.reset();obs=[];cx=[];acts=[];lps=[];rewards=[];dones=[]
logstd=nn.Parameter(torch.full((29,),np.log(.05),device=a.device))
for step in range(64):
 obs.append(o.copy());cx.append(e.critic_observation())
 with torch.no_grad():dist=Normal(actor(torch.tensor(o[None],device=a.device)),logstd.exp());act=dist.sample();lps.append(float(dist.log_prob(act).sum()));acts.append(act.cpu().numpy()[0])
 o,r,done,_=e.step(acts[-1]);rewards.append(r);dones.append(done)
 if done:o=e.reset()
x=torch.tensor(np.array(obs),device=a.device);c=torch.tensor(np.array(cx),device=a.device);actions=torch.tensor(np.array(acts),device=a.device);old=torch.tensor(lps,device=a.device);critic=Critic(c.shape[1]).to(a.device);critic.fit_normalization(c)
with torch.no_grad():v=critic(c).cpu().numpy();last=critic(torch.tensor(e.critic_observation()[None],device=a.device)).cpu().numpy();target=teacher(x)
adv,ret=advantages(np.array(rewards,np.float32)[:,None],v[:,None],np.array(dones)[:,None],last);ADV=torch.tensor(adv.ravel(),device=a.device);ADV=(ADV-ADV.mean())/(ADV.std()+1e-8);RET=torch.tensor(ret.ravel(),device=a.device)
co=torch.optim.Adam(critic.parameters(),lr=3e-4);ao=torch.optim.Adam(list(actor.parameters())+[logstd],lr=1e-6)
vl=.5*(critic(c)-RET).square().mean();co.zero_grad();vl.backward();nn.utils.clip_grad_norm_(critic.parameters(),1.,error_if_nonfinite=True);co.step()
assert all(torch.equal(v,actor.state_dict()[k]) for k,v in initial.items())
dist=Normal(actor(x),logstd.exp());ratio=(dist.log_prob(actions).sum(-1)-old).exp();anchor=.5*((dist.mean-target)/.05).square().mean();loss=-torch.minimum(ratio*ADV,ratio.clamp(.8,1.2)*ADV).mean()+.1*anchor-.0001*dist.entropy().sum(-1).mean()
ao.zero_grad();loss.backward();nn.utils.clip_grad_norm_(list(actor.parameters())+[logstd],1.,error_if_nonfinite=True);ao.step()
assert all(torch.equal(v,teacher.state_dict()[k]) for k,v in initial.items());assert all(p.grad is None for p in teacher.parameters());assert all(torch.isfinite(p).all() for p in actor.parameters())
assert any(not torch.equal(v,actor.state_dict()[k]) for k,v in initial.items())
assert torch.equal(initial['mean'],actor.mean) and torch.equal(initial['divisor'],actor.divisor)
print(json.dumps({'real_rollout_steps':64,'critic_input_dim':c.shape[1],'separate_optimizers':True,'actor_frozen_during_critic_step':True,'teacher_unchanged':True,'actor_normalization_unchanged':True,'finite_actor_update':True,'value_loss':float(vl.detach()),'actor_loss':float(loss.detach())}))
