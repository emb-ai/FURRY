"""Bounded MuJoCo PPO adaptation of the exact recovered TWIST2 actor.

Fresh critic; frozen released observation normalization; 100Hz Quest PD.
Train/validation are disjoint protocol stages. No validation gradients.
"""
import argparse,json,time,os,hashlib,signal
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from env import Motions,Env
from policy import load
from evaluate import evaluate

def main():
 p=argparse.ArgumentParser();p.add_argument('--assets',required=True);p.add_argument('--dataset',required=True);p.add_argument('--initial',required=True);p.add_argument('--source',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda');p.add_argument('--envs',type=int,default=16);p.add_argument('--iterations',type=int,default=1000);p.add_argument('--horizon',type=int,default=128);p.add_argument('--seed',type=int,default=20261008);p.add_argument('--eval-every',type=int,default=100);a=p.parse_args()
 a.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1);torch.manual_seed(a.seed);np.random.seed(a.seed)
 if a.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('GPU allocation missing')
 actor=load(a.initial,a.source,a.device);actor.train();critic=nn.Sequential(nn.Linear(1432,512),nn.ELU(),nn.Linear(512,256),nn.ELU(),nn.Linear(256,1)).to(a.device);logstd=nn.Parameter(torch.full((29,),np.log(.05),device=a.device));optimizer=torch.optim.Adam([{'params':actor.parameters(),'lr':1e-5},{'params':critic.parameters(),'lr':3e-4},{'params':[logstd],'lr':1e-5}]);gamma=.99**.5;lam=.95**.5
 # Preserve the 50Hz discount and GAE decay per second at 100Hz.
 # Continuous reward terms already carry DT in Env.step; do not scale twice.
 motions=Motions(a.dataset,'train');envs=[Env(a.assets,motions,a.seed+j) for j in range(a.envs)];pool=ThreadPoolExecutor(max_workers=a.envs);obs=np.stack([e.reset(randomize=True) for e in envs]);best=(-1.,-float('inf'));start_time=time.time();stop=False
 def handler(*_):
  nonlocal stop
  stop=True
 signal.signal(signal.SIGTERM,handler);signal.signal(signal.SIGINT,handler)
 def value(x):return critic(torch.clamp((x-actor.mean)/actor.divisor,-10,10)).squeeze(-1)
 def checkpoint(path,it):
  tmp=path.with_suffix('.tmp');torch.save({'policy':actor.state_dict(),'critic':critic.state_dict(),'logstd':logstd.detach(),'optimizer':optimizer.state_dict(),'iteration':it,'schema':'quest-mujoco-ppo-v1'},tmp);os.replace(tmp,path)
 settings=vars(a).copy();settings['output']=str(a.output);settings.update({'actor_lr':1e-5,'critic_lr':3e-4,'gamma':gamma,'gae_lambda':lam,'clip':.2,'entropy_coef':.005,'fresh_critic_warmup_updates':10,'target_kl':.02,'normalization':'frozen ONNX','initial_sha256':hashlib.sha256(Path(a.initial).read_bytes()).hexdigest(),'dataset_sha256':hashlib.sha256((Path(a.dataset)/'manifest.json').read_bytes()).hexdigest(),'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0) if a.device=='cuda' else 'cpu'})
 (a.output/'config.json').write_text(json.dumps(settings,indent=2));print(json.dumps(settings),flush=True)
 def step_pair(x):return x[0].step(x[1])
 log=(a.output/'metrics.jsonl').open('a',buffering=1)
 for it in range(1,a.iterations+1):
  ob=[];actions=[];logs=[];values=[];rewards=[];dones=[];completed=[]
  for t in range(a.horizon):
   x=torch.as_tensor(obs,device=a.device)
   with torch.no_grad():mu=actor(x);dist=Normal(mu,logstd.exp());act=dist.sample();lp=dist.log_prob(act).sum(-1);v=value(x)
   results=list(pool.map(step_pair,zip(envs,act.cpu().numpy())));nobs=np.stack([r[0] for r in results]);r=np.array([r[1] for r in results],np.float32);d=np.array([r[2] for r in results]);trunc=np.array([r[2] and r[3]['truncated'] and not r[3]['fall'] for r in results])
   if trunc.any():
    with torch.no_grad():r[trunc]+=gamma*value(torch.as_tensor(nobs[trunc],device=a.device)).cpu().numpy()
   ob.append(obs.copy());actions.append(act.cpu().numpy());logs.append(lp.cpu().numpy());values.append(v.cpu().numpy());rewards.append(r);dones.append(d)
   for j in np.flatnonzero(d):completed.append(results[j][3]);nobs[j]=envs[j].reset(randomize=True)
   obs=nobs
  with torch.no_grad():last_v=value(torch.as_tensor(obs,device=a.device)).cpu().numpy()
  rewards=np.array(rewards);values=np.array(values);dones=np.array(dones);advantages=np.zeros_like(rewards);gae=np.zeros(a.envs)
  for t in reversed(range(a.horizon)):
   nv=last_v if t==a.horizon-1 else values[t+1];live=1-dones[t].astype(float);delta=rewards[t]+gamma*nv*live-values[t];gae=delta+gamma*lam*live*gae;advantages[t]=gae
  returns=advantages+values
  X=torch.as_tensor(np.array(ob).reshape(-1,1432),device=a.device);A=torch.as_tensor(np.array(actions).reshape(-1,29),device=a.device);OLD=torch.as_tensor(np.array(logs).ravel(),device=a.device);ADV=torch.as_tensor(advantages.ravel(),device=a.device);RET=torch.as_tensor(returns.ravel(),device=a.device);ADV=(ADV-ADV.mean())/(ADV.std()+1e-8)
  losses=[];kls=[];early=False
  for epoch in range(4):
   order=torch.randperm(len(X),device=a.device)
   for ix in order.split(512):
    mu=actor(X[ix]);dist=Normal(mu,logstd.exp());lp=dist.log_prob(A[ix]).sum(-1);ratio=(lp-OLD[ix]).exp();kl=((ratio-1)-(lp-OLD[ix])).mean()
    if it>10 and float(kl.detach())>.02:early=True;break
    policy_loss=-torch.minimum(ratio*ADV[ix],torch.clamp(ratio,.8,1.2)*ADV[ix]).mean();vl=((value(X[ix])-RET[ix])**2).mean();loss=.5*vl
    if it>10:loss=loss+policy_loss-.005*dist.entropy().sum(-1).mean()
    optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(list(actor.parameters())+list(critic.parameters())+[logstd],1.);optimizer.step()
    with torch.no_grad():logstd.clamp_(np.log(.02),np.log(.15))
    losses.append(float(loss.detach()));kls.append(float(kl.detach()))
   if early:break
  record={'iteration':it,'samples':it*a.envs*a.horizon,'elapsed_s':time.time()-start_time,'reward_mean':float(rewards.mean()),'loss_mean':float(np.mean(losses)) if losses else None,'kl_mean':float(np.mean(kls)) if kls else None,'std_mean':float(logstd.exp().mean().detach()),'completed_episodes':len(completed),'completed_falls':sum(x['fall'] for x in completed),'episode_seconds_mean':float(np.mean([x['seconds'] for x in completed])) if completed else None}
  print(json.dumps(record),flush=True);log.write(json.dumps(record)+'\n')
  if it%10==0 or it==a.iterations or stop:checkpoint(a.output/'latest.pt',it)
  if it%a.eval_every==0 or it==a.iterations:
   checkpoint(a.output/f'checkpoint_{it}.pt',it);ev=evaluate(a.assets,a.dataset,a.output/f'checkpoint_{it}.pt',a.source,output=a.output/f'validation_{it}.json')
   score=(ev['completion_fraction'],ev['survived_seconds'])
   if score>best:best=score;checkpoint(a.output/'best.pt',it)
  if stop:break
 checkpoint(a.output/'latest.pt',it);pool.shutdown();log.close()
if __name__=='__main__':main()
