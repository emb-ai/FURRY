"""Guarded adaptation: privileged critic, measured warm-up, anchored PPO."""
import argparse,json,time,os,hashlib,signal
from pathlib import Path
from dataclasses import asdict
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from env import Motions,Env
from rewards import RewardConfig
from policy import load
from critic import Critic,warmup,collect_mc,value_metrics,critic_ready
from evaluate import evaluate,compare
from ppo_utils import GAMMA,GAE_LAMBDA,advantages,actor_lr


def parser():
 p=argparse.ArgumentParser()
 for key in ['assets','dataset','initial','source']:p.add_argument('--'+key,required=True)
 p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda')
 for key,default in [('envs',16),('iterations',300),('horizon',128),('seed',20261008),('eval-every',25),('actor-warmup-updates',50),('critic-train-episodes',32),('critic-holdout-episodes',8),('critic-min-updates',100),('critic-max-updates',2000),('critic-probe-episodes',4),('regression-patience',3)]:p.add_argument('--'+key,type=int,default=default)
 for key,default in [('actor-lr',1e-5),('critic-lr',3e-4),('anchor-coef',.1),('entropy-coef',.0001),('slide-weight',.5),('critic-min-ev',.3),('critic-max-nrmse',.85),('critic-max-bias',.2)]:p.add_argument('--'+key,type=float,default=default)
 p.add_argument('--eval-seeds',nargs='+',type=int,default=[0,1,2]);p.add_argument('--legacy-reward',action='store_true')
 p.add_argument('--legacy-slip',action='store_true')
 p.add_argument('--warmup-only',action='store_true')
 return p


def main():
 a=parser().parse_args()
 for k in ['envs','iterations','horizon','eval_every','actor_warmup_updates','critic_train_episodes','critic_holdout_episodes','critic_min_updates','critic_max_updates','critic_probe_episodes','regression_patience']:
  if getattr(a,k)<1:raise ValueError(k+' must be positive')
 if a.critic_min_updates>a.critic_max_updates:raise ValueError('Critic update bounds')
 for k in ['actor_lr','critic_lr','anchor_coef','entropy_coef','slide_weight']:
  if not np.isfinite(getattr(a,k)) or getattr(a,k)<0:raise ValueError(k+' must be finite and nonnegative')
 a.output.mkdir(parents=True,exist_ok=True)
 if (a.output/'config.json').exists():raise RuntimeError('Use a new run directory; automatic resume is not supported')
 torch.set_num_threads(1);torch.manual_seed(a.seed);np.random.seed(a.seed)
 if a.device.startswith('cuda') and not torch.cuda.is_available():raise RuntimeError('GPU allocation missing')
 config=RewardConfig(slide_weight=a.slide_weight,legacy=a.legacy_reward,legacy_slip=a.legacy_slip)
 actor=load(a.initial,a.source,a.device);actor.train()
 teacher=load(a.initial,a.source,a.device);teacher.requires_grad_(False);teacher.eval()
 motions=Motions(a.dataset,'train');envs=[Env(a.assets,motions,a.seed+j,reward_config=config) for j in range(a.envs)]
 obs=np.stack([e.reset(randomize=True) for e in envs]);cobs=np.stack([e.critic_observation() for e in envs]);critic=Critic(cobs.shape[1]).to(a.device)
 logstd=nn.Parameter(torch.full((29,),np.log(.05),device=a.device));actor_opt=torch.optim.Adam([{'params':actor.parameters()},{'params':[logstd]}],lr=a.actor_lr);critic_opt=torch.optim.Adam(critic.parameters(),lr=a.critic_lr)
 start_time=time.time();stop=False;it=0;stop_reason=None;best_score=None;regressions=0;bad_critic=0
 def handler(*_):
  nonlocal stop
  stop=True
 signal.signal(signal.SIGTERM,handler);signal.signal(signal.SIGINT,handler)
 log_file=(a.output/'metrics.jsonl').open('a',buffering=1)
 def log(record):
  record={'elapsed_s':time.time()-start_time,**record};print(json.dumps(record),flush=True);log_file.write(json.dumps(record)+'\n')
 def checkpoint(path,iteration):
  tmp=path.with_suffix('.tmp');torch.save({'policy':actor.state_dict(),'critic':critic.state_dict(),'logstd':logstd.detach(),'actor_optimizer':actor_opt.state_dict(),'critic_optimizer':critic_opt.state_dict(),'iteration':iteration,'schema':'quest-mujoco-ppo-v2','reward_config':asdict(config),'critic_input_dim':cobs.shape[1],'actor_lr_warmup_updates':a.actor_warmup_updates},tmp);os.replace(tmp,path)
 settings=vars(a).copy();settings['output']=str(a.output);settings.update({'gamma':GAMMA,'gae_lambda':GAE_LAMBDA,'reward_config':asdict(config),'critic_input_dim':cobs.shape[1],'initial_sha256':hashlib.sha256(Path(a.initial).read_bytes()).hexdigest(),'dataset_sha256':hashlib.sha256((Path(a.dataset)/'manifest.json').read_bytes()).hexdigest(),'env_sha256':hashlib.sha256(Path(__file__).with_name('env.py').read_bytes()).hexdigest(),'training_source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(__file__).parent.glob('*.py'))},'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0) if a.device.startswith('cuda') else 'cpu','schema':'quest-mujoco-ppo-v2'})
 (a.output/'config.json').write_text(json.dumps(settings,indent=2));log({'phase':'config',**settings})
 pool=None
 try:
  readiness=warmup(critic,actor,a.assets,motions,config,a,GAMMA,lambda:stop,log,optimizer=critic_opt)
  checkpoint(a.output/'critic_warmup.pt',0)
  if not readiness['ready']:
   stop_reason='critic_not_ready';return
  if a.warmup_only:
   stop_reason='warmup_only';return
  if stop:raise InterruptedError('Stopped after warmup')
  baselines={}
  for split in ['validation','replay']:
   baselines[split]=evaluate(a.assets,a.dataset,a.initial,a.source,split,a.eval_seeds,a.output/f'baseline-{split}.json',reward_config=config,stop=lambda:stop)
   if stop:raise InterruptedError('Stopped during baseline evaluation')
  # Preserve the original actor as explicit fallback; best.pt is created only after acceptance.
  checkpoint(a.output/'baseline.pt',0)
  pool=ThreadPoolExecutor(max_workers=a.envs)
  def step_pair(x):return x[0].step(x[1])
  for it in range(1,a.iterations+1):
   ob=[];co=[];acts=[];old_logs=[];vals=[];rs=[];raw_rs=[];ds=[];completed=[];term_sums={};bootstrap_sum=0.
   for t in range(a.horizon):
    x=torch.as_tensor(obs,device=a.device);cx=torch.as_tensor(cobs,device=a.device)
    with torch.no_grad():mu=actor(x);dist=Normal(mu,logstd.exp());act=dist.sample();lp=dist.log_prob(act).sum(-1);v=critic(cx)
    results=list(pool.map(step_pair,zip(envs,act.cpu().numpy())));nobs=np.stack([r[0] for r in results]);ncobs=np.stack([e.critic_observation() for e in envs]);r=np.array([r[1] for r in results],np.float32);d=np.array([r[2] for r in results]);raw_rs.append(r.copy())
    # Reference end is a task terminal; only an artificial time limit bootstraps.
    timeout=np.array([r[2] and r[3]['time_limit'] and not r[3]['reference_end'] and not r[3]['fall'] for r in results])
    if timeout.any():
     with torch.no_grad():boot=GAMMA*critic(torch.as_tensor(ncobs[timeout],device=a.device)).cpu().numpy()
     r[timeout]+=boot;bootstrap_sum+=float(boot.sum())
    for result in results:
     for name,rate in result[3]['reward_terms'].items():term_sums[name]=term_sums.get(name,0.)+float(rate)
    ob.append(obs.copy());co.append(cobs.copy());acts.append(act.cpu().numpy());old_logs.append(lp.cpu().numpy());vals.append(v.cpu().numpy());rs.append(r);ds.append(d)
    for j in np.flatnonzero(d):completed.append(results[j][3]);nobs[j]=envs[j].reset(randomize=True);ncobs[j]=envs[j].critic_observation()
    obs,cobs=nobs,ncobs
   with torch.no_grad():last_v=critic(torch.as_tensor(cobs,device=a.device)).cpu().numpy()
   rs=np.asarray(rs);vals=np.asarray(vals);adv,ret=advantages(rs,vals,np.asarray(ds),last_v)
   X=torch.as_tensor(np.asarray(ob).reshape(-1,1432),device=a.device);C=torch.as_tensor(np.asarray(co).reshape(-1,cobs.shape[1]),device=a.device);A=torch.as_tensor(np.asarray(acts).reshape(-1,29),device=a.device);OLD=torch.as_tensor(np.asarray(old_logs).ravel(),device=a.device);ADV=torch.as_tensor(adv.ravel(),device=a.device);RET=torch.as_tensor(ret.ravel(),device=a.device);ADV=(ADV-ADV.mean())/(ADV.std(unbiased=False)+1e-8)
   with torch.no_grad():TEACHER=teacher(X)
   lr=actor_lr(it,a.actor_lr,a.actor_warmup_updates)
   for group in actor_opt.param_groups:group['lr']=lr
   values_loss=[];policy_losses=[];anchors=[];entropies=[];kls=[];actor_stopped=False
   for epoch in range(4):
    for ix in torch.randperm(len(X),device=a.device).split(512):
     vl=.5*(critic(C[ix])-RET[ix]).square().mean()
     if not torch.isfinite(vl):raise FloatingPointError('Non-finite critic loss')
     critic_opt.zero_grad();vl.backward();nn.utils.clip_grad_norm_(critic.parameters(),1.,error_if_nonfinite=True);critic_opt.step();values_loss.append(float(vl.detach()))
     if actor_stopped:continue
     dist=Normal(actor(X[ix]),logstd.exp());lp=dist.log_prob(A[ix]).sum(-1);delta=lp-OLD[ix];ratio=delta.exp();kl=((ratio-1)-delta).mean();kls.append(float(kl.detach()))
     if float(kl.detach())>.02:actor_stopped=True;continue
     pl=-torch.minimum(ratio*ADV[ix],ratio.clamp(.8,1.2)*ADV[ix]).mean()
     anchor=.5*((dist.mean-TEACHER[ix])/.05).square().mean();entropy=dist.entropy().sum(-1).mean();loss=pl+a.anchor_coef*anchor-a.entropy_coef*entropy
     if not torch.isfinite(loss):raise FloatingPointError('Non-finite actor loss')
     actor_opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(list(actor.parameters())+[logstd],1.,error_if_nonfinite=True);actor_opt.step()
     with torch.no_grad():logstd.clamp_(np.log(.02),np.log(.15))
     policy_losses.append(float(pl.detach()));anchors.append(float(anchor.detach()));entropies.append(float(entropy.detach()))
   with torch.no_grad():
    prediction=critic(C).cpu().numpy();drift=actor(X)-TEACHER;drift_rms=float(drift.square().mean().sqrt());critic_clipped=float((((C-critic.mean)/critic.scale).abs()>20).float().mean())
   def mean(x):return float(np.mean(x)) if x else None
   log({'phase':'ppo','iteration':it,'samples':it*a.envs*a.horizon,'actor_lr':lr,'environment_reward_mean':float(np.mean(raw_rs)),'bootstrap_added_sum':bootstrap_sum,'reward_term_rates':{k:v/(a.envs*a.horizon) for k,v in term_sums.items()},'policy_loss':mean(policy_losses),'value_loss':mean(values_loss),'anchor_loss':mean(anchors),'entropy':mean(entropies),'entropy_coefficient':a.entropy_coef,'ppo_kl':mean(kls),'actor_kl_stopped':actor_stopped,'actor_update_batches':len(policy_losses),'teacher_action_rms':drift_rms,'teacher_PD_target_rms_rad':.5*drift_rms,'critic_on_gae_targets':value_metrics(prediction,ret.ravel()),'critic_input_clipped_fraction':critic_clipped,'std_mean':float(logstd.exp().mean().detach()),'completed_episodes':len(completed),'completed_falls':sum(x['fall'] for x in completed)})
   if it%10==0 or it==a.iterations or stop:checkpoint(a.output/'latest.pt',it)
   if stop:raise InterruptedError('Stopped during PPO')
   if it%a.eval_every==0 or it==a.iterations:
    path=a.output/f'checkpoint_{it}.pt';checkpoint(path,it);comparisons={}
    for split in ['validation','replay']:
     ev=evaluate(a.assets,a.dataset,path,a.source,split,a.eval_seeds,a.output/f'{split}_{it}.json',reward_config=config,stop=lambda:stop)
     comparisons[split]=compare(baselines[split],ev)
     if stop:raise InterruptedError('Stopped during evaluation')
    # New on-policy, complete-episode targets reveal critic degradation hidden by TD loss.
    px,py,episodes=collect_mc(actor,a.assets,motions,config,a.critic_probe_episodes,a.seed+30000+it,a.device,GAMMA,logstd.exp().detach().cpu().numpy(),lambda:stop)
    with torch.no_grad():pv=torch.cat([critic(b) for b in torch.as_tensor(px,device=a.device).split(1024)]).cpu().numpy()
    probe=value_metrics(pv,py);healthy=critic_ready(probe,a.critic_min_ev,a.critic_max_nrmse,a.critic_max_bias);bad_critic=0 if healthy else bad_critic+1
    accepted=comparisons['validation']['accepted'] and comparisons['replay']['non_regressing'] and healthy
    severe=any(not c['non_regressing'] for c in comparisons.values());regressions=regressions+1 if severe else 0
    acceptance={'iteration':it,'accepted':accepted,'comparisons':comparisons,'critic_probe':probe,'critic_probe_episodes':episodes,'critic_healthy':healthy,'consecutive_regressions':regressions,'consecutive_bad_critic':bad_critic}
    (a.output/f'acceptance_{it}.json').write_text(json.dumps(acceptance,indent=2));log({'phase':'acceptance',**acceptance})
    # Prefer completion, then slip among candidates satisfying all tracking gates.
    cm=comparisons['validation']['matched_prefix_means']['candidate'];vr=json.loads((a.output/f'validation_{it}.json').read_text());score=(vr['completion_fraction'],-cm['slide_rate'],-cm['joint_rmse'])
    if accepted and (best_score is None or score>best_score):best_score=score;checkpoint(a.output/'best.pt',it)
    if regressions>=a.regression_patience:stop_reason='persistent_baseline_regression';break
    if bad_critic>=2:stop_reason='critic_regressed';break
  stop_reason=stop_reason or 'iteration_budget_complete'
 except InterruptedError as exc:
  stop_reason='interrupted';log({'phase':'stop','reason':str(exc)})
 except Exception as exc:
  stop_reason='error';log({'phase':'error','message':str(exc)});raise
 finally:
  checkpoint(a.output/'latest.pt',it)
  if pool:pool.shutdown()
  result={'stop_reason':stop_reason,'last_iteration':it,'accepted_checkpoint':str(a.output/'best.pt') if (a.output/'best.pt').exists() else None,'fallback':'original initial policy','elapsed_s':time.time()-start_time}
  (a.output/'result.json').write_text(json.dumps(result,indent=2));log({'phase':'result',**result});log_file.close()

if __name__=='__main__':main()
