"""Privileged value function and measured, actor-frozen Monte Carlo warm-up."""
import json
import numpy as np
import torch
from torch import nn
from env import Env


class Critic(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.register_buffer('mean', torch.zeros(size))
        self.register_buffer('scale', torch.ones(size))
        self.net = nn.Sequential(nn.Linear(size, 512), nn.ELU(), nn.Linear(512, 256), nn.ELU(), nn.Linear(256, 1))

    def fit_normalization(self, x):
        # Fit only to train rollouts, then freeze throughout PPO.
        with torch.no_grad():
            self.mean.copy_(x.mean(0))
            self.scale.copy_(x.std(0, unbiased=False).clamp_min(.05))

    def forward(self, x):
        return self.net(((x-self.mean)/self.scale).clamp(-20,20)).squeeze(-1)


def value_metrics(pred, target):
    pred=np.asarray(pred,dtype=float);target=np.asarray(target,dtype=float)
    error=pred-target;variance=float(np.var(target));std=max(variance**.5,1e-6)
    return {'rmse':float(np.sqrt(np.mean(error**2))), 'bias':float(np.mean(error)),
            'normalized_rmse':float(np.sqrt(np.mean(error**2))/std),
            'normalized_bias':float(abs(np.mean(error))/std),
            'explained_variance':float(1-np.var(error)/variance) if variance>1e-10 else None,
            'target_std':std,'target_mean':float(target.mean()),'samples':len(target)}


def critic_ready(metrics, min_ev=.3, max_nrmse=.85, max_bias=.2):
    return (metrics['explained_variance'] is not None and
            metrics['explained_variance']>=min_ev and
            metrics['normalized_rmse']<=max_nrmse and metrics['normalized_bias']<=max_bias)


def discounted_returns(rewards, gamma):
    out=np.empty(len(rewards),np.float32);total=0.
    for i in range(len(rewards)-1,-1,-1):
        total=float(rewards[i])+gamma*total;out[i]=total
    return out


def collect_mc(actor, assets, motions, config, episodes, seed, device, gamma, std, stop):
    """Complete source episodes or actual falls; no critic bootstraps in targets.

    Different seeds provide held-out stochastic rollouts from training motions.
    Validation motions never enter this dataset. Five-step subsampling reduces RAM.
    """
    e=Env(assets,motions,seed,reward_config=config);e.max_steps=10_000_000
    rng=np.random.default_rng(seed);xs=[];ys=[];summary=[]
    for episode in range(episodes):
        if stop():raise InterruptedError('Stopped during critic data collection')
        obs=e.reset(randomize=True);states=[];rewards=[];indices=[]
        while True:
            if e.steps%100==0 and stop():raise InterruptedError('Stopped during critic rollout')
            if e.steps%5==0:indices.append(e.steps);states.append(e.critic_observation())
            with torch.no_grad():mu=actor(torch.as_tensor(obs[None],device=device)).cpu().numpy()[0]
            act=(mu+rng.normal(0,std,mu.shape)).astype(np.float32)
            obs,r,done,info=e.step(act);rewards.append(r)
            if done:break
        target=discounted_returns(rewards,gamma);xs.extend(states);ys.extend(target[indices]);summary.append({'clip':e.motion['meta']['id'],'fall':info['fall'],'seconds':info['seconds']})
        print(json.dumps({'phase':'critic_rollouts','seed':seed,'episode':episode+1,'episodes':episodes,**summary[-1]}),flush=True)
    return np.asarray(xs,np.float32),np.asarray(ys,np.float32),summary


def warmup(critic, actor, assets, motions, config, args, gamma, stop, log, optimizer=None):
    device=args.device
    train_x,train_y,train_eps=collect_mc(actor,assets,motions,config,args.critic_train_episodes,args.seed+10000,device,gamma,.05,stop)
    test_x,test_y,test_eps=collect_mc(actor,assets,motions,config,args.critic_holdout_episodes,args.seed+20000,device,gamma,.05,stop)
    if min(len(train_y),len(test_y))<32:raise RuntimeError('Too few complete-rollout samples for critic readiness')
    # Save diagnostic evidence; these are private experiment outputs, never repo assets.
    np.savez_compressed(args.output/'critic_holdout.npz',states=test_x,returns=test_y)
    (args.output/'critic_rollouts.json').write_text(json.dumps({'train':train_eps,'holdout':test_eps,'targets':'full-episode Monte Carlo; actual falls and source ends terminate; stride5'},indent=2))
    x=torch.as_tensor(train_x,device=device);y=torch.as_tensor(train_y,device=device)
    vx=torch.as_tensor(test_x,device=device)
    critic.fit_normalization(x);opt=optimizer if optimizer is not None else torch.optim.Adam(critic.parameters(),lr=args.critic_lr)
    consecutive=0;best_error=float('inf');best_state=None;ready=False
    for update in range(1,args.critic_max_updates+1):
        if stop():raise InterruptedError('Stopped during critic fit')
        ix=torch.randint(len(x),(min(512,len(x)),),device=device)
        loss=.5*(critic(x[ix])-y[ix]).square().mean();opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(critic.parameters(),1.);opt.step()
        if update%25==0 or update==args.critic_max_updates:
            with torch.no_grad():pred=torch.cat([critic(b) for b in vx.split(1024)]).cpu().numpy()
            metrics=value_metrics(pred,test_y)
            good=critic_ready(metrics,args.critic_min_ev,args.critic_max_nrmse,args.critic_max_bias)
            consecutive=consecutive+1 if good and update>=args.critic_min_updates else 0
            record={'phase':'critic_warmup','update':update,'train_loss':float(loss.detach()),'holdout':metrics,'consecutive_ready':consecutive}
            log(record)
            if metrics['normalized_rmse']<best_error:
                best_error=metrics['normalized_rmse'];best_state={k:v.detach().cpu().clone() for k,v in critic.state_dict().items()}
            if consecutive>=2:ready=True;break
    # Keep the model that actually passed both checks, not a different historical model.
    if not ready and best_state is not None:critic.load_state_dict(best_state)
    with torch.no_grad():final=value_metrics(torch.cat([critic(b) for b in vx.split(1024)]).cpu().numpy(),test_y)
    report={'ready':ready,'updates':update,'holdout':final,'thresholds':{'min_ev':args.critic_min_ev,'max_nrmse':args.critic_max_nrmse,'max_normalized_bias':args.critic_max_bias},'actor_updates':0}
    (args.output/'critic_readiness.json').write_text(json.dumps(report,indent=2));return report
