"""BC initialization followed by standard SB3 PPO; small local runs are smoke tests."""
import argparse
import json
import time
import numpy as np
import torch
from native_env import NativeRubikEnv, ROOT
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

class Progress(BaseCallback):
    def __init__(self,name):
        super().__init__();self.name=name;self.episodes=[];self.moves=0;self.started=time.monotonic()
    def _on_step(self):
        for info,done in zip(self.locals['infos'],self.locals['dones']):
            self.moves+=int(info.get('completed_this_step',False))
            if done:self.episodes.append(dict(success=bool(info['success']),completed_moves=info['completed_moves'],failure=info['failure']))
        if self.n_calls%128==0:
            r=dict(status='running',timesteps=self.num_timesteps,completed_moves=self.moves,
                   completed_episodes=len(self.episodes),successful_episodes=sum(x['success'] for x in self.episodes),
                   wall_seconds=time.monotonic()-self.started)
            (ROOT/(self.name+'-progress.json')).write_text(json.dumps(r,indent=2))
            self.model.save(ROOT/(self.name+'-latest'))
        return True

def make_env(length,faces,signs,seed):
    def create():
        return Monitor(NativeRubikEnv(sequence_length=length,faces=faces,signs=signs,seed=seed,move_timeout=120))
    return create

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--bc-updates',type=int,default=1000)
    ap.add_argument('--ppo-steps',type=int,default=128)
    ap.add_argument('--sequence-length',type=int,default=2)
    ap.add_argument('--faces',type=int,nargs='+',default=[3,4])
    ap.add_argument('--signs',type=int,nargs='+',default=[-1])
    ap.add_argument('--seed',type=int,default=0)
    ap.add_argument('--envs',type=int,default=1)
    ap.add_argument('--load')
    ap.add_argument('--name',default='smoke')
    ap.add_argument('--learning-rate',type=float,default=1e-5)
    ap.add_argument('--target-kl',type=float,default=.02)
    args=ap.parse_args()
    torch.set_num_threads(2);torch.manual_seed(args.seed)
    factories=[make_env(args.sequence_length,args.faces,args.signs,args.seed+i) for i in range(args.envs)]
    env=SubprocVecEnv(factories,start_method='spawn') if args.envs>1 else factories[0]()
    model=PPO.load(args.load,env=env,device='cpu',learning_rate=args.learning_rate,target_kl=args.target_kl) if args.load else PPO('MlpPolicy',env,device='cpu',n_steps=128,batch_size=64,n_epochs=4,
              learning_rate=args.learning_rate,target_kl=args.target_kl,gamma=.995,gae_lambda=.95,clip_range=.2,
              policy_kwargs=dict(net_arch=dict(pi=[256,256],vf=[256,256]),log_std_init=-3),
              seed=args.seed,verbose=1)
    data=dict(np.load(ROOT/'demonstrations.npz'))
    obs=torch.from_numpy(data['observations']);actions=torch.from_numpy(data['actions'])
    train=np.flatnonzero(data['train']);valid=np.flatnonzero(~data['train'])
    rng=np.random.default_rng(args.seed)
    optimizer=torch.optim.Adam(model.policy.parameters(),lr=3e-4)
    log=[]
    def loss_for(ids):
        mean=model.policy.get_distribution(obs[ids]).distribution.mean
        return torch.mean((mean-actions[ids])**2)
    for step in range(args.bc_updates):
        ids=rng.choice(train,128,replace=True)
        loss=loss_for(ids);optimizer.zero_grad();loss.backward()
        torch.nn.utils.clip_grad_norm_(model.policy.parameters(),1.)
        optimizer.step()
        if step%100==0 or step==args.bc_updates-1:
            with torch.no_grad():validation=float(loss_for(valid))
            record=dict(update=step+1,train_mse=float(loss.detach()),validation_mse=validation)
            log.append(record);print('BC',json.dumps(record),flush=True)
    if args.bc_updates:
        model.save(ROOT/(args.name+'-bc-policy'))
        (ROOT/(args.name+'-bc-training.json')).write_text(json.dumps(log,indent=2))
    start=time.monotonic()
    progress=Progress(args.name)
    if args.ppo_steps:model.learn(total_timesteps=args.ppo_steps,reset_num_timesteps=True,callback=progress)
    model.save(ROOT/(args.name+'-policy'))
    result=dict(arguments=vars(args),bc_updates=args.bc_updates,ppo_steps=int(model.num_timesteps),
                ppo_wall_seconds=time.monotonic()-start,held_out_success_rate=None,
                completed_training_moves=progress.moves,training_episodes=progress.episodes,
                status='training pilot completed; episode outcomes are training results, no held-out evaluation',
                action_interface='two wrist poses plus two scalar Grip commands',
                lower_policy='unchanged HOMIE Standing',physics_modified=False,headset_deployed=False)
    (ROOT/(args.name+'-training-status.json')).write_text(json.dumps(result,indent=2))
    (ROOT/(args.name+'-progress.json')).write_text(json.dumps(dict(status='completed',
        timesteps=int(model.num_timesteps),completed_moves=progress.moves,
        completed_episodes=len(progress.episodes),successful_episodes=sum(x['success'] for x in progress.episodes)),indent=2))
    print(json.dumps(result,indent=2));env.close()

if __name__=='__main__':main()
