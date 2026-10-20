from pathlib import Path
import sys
import torch
from torch import nn
from diffusers import DDIMScheduler
sys.path.insert(0,str(Path(__file__).resolve().parent/'vendor'))
from diffusion_policy_3d.model.diffusion.conditional_unet1d import ConditionalUnet1D
HISTORY=8
HORIZON=16
ACTION_DIM=37

class BCRNN(nn.Module):
    def __init__(self,obs_dim):
        super().__init__()
        self.embed=nn.Sequential(nn.Linear(obs_dim,256),nn.LayerNorm(256),nn.SiLU())
        self.rnn=nn.GRU(256,256,2,batch_first=True)
        self.head=nn.Sequential(nn.Linear(256,256),nn.SiLU(),nn.Linear(256,HORIZON*ACTION_DIM))
    def forward(self,obs):
        h,_=self.rnn(self.embed(obs))
        return self.head(h[:,-1]).reshape(-1,HORIZON,ACTION_DIM)
    def predict(self,obs):return self(obs)

def scheduler():
    return DDIMScheduler(num_train_timesteps=100,beta_schedule='squaredcos_cap_v2',prediction_type='epsilon',clip_sample=False)

class Diffusion(nn.Module):
    def __init__(self,obs_dim):
        super().__init__()
        self.encoder=nn.Sequential(nn.Flatten(1),nn.Linear(HISTORY*obs_dim,256),nn.LayerNorm(256),nn.SiLU(),nn.Linear(256,256),nn.SiLU())
        self.unet=ConditionalUnet1D(input_dim=ACTION_DIM,local_cond_dim=None,global_cond_dim=256,diffusion_step_embed_dim=64,down_dims=[64,128,256],kernel_size=3,n_groups=8,condition_type='film')
    def forward(self,obs,x,t):return self.unet(x,t,global_cond=self.encoder(obs))
    def predict(self,obs):
        cond=self.encoder(obs);x=torch.randn((len(obs),HORIZON,ACTION_DIM),device=obs.device)
        sched=scheduler();sched.set_timesteps(16,device=obs.device)
        for t in sched.timesteps:x=sched.step(self.unet(x,t,global_cond=cond),t,x).prev_sample
        return x

def make(kind,obs_dim):return BCRNN(obs_dim) if kind=='bc-rnn' else Diffusion(obs_dim)
