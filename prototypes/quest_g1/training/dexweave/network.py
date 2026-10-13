"""Six anatomical tokens and five body action heads (DexWeave body-only)."""
import torch
from torch import nn
from .spec import ACTOR_SLICES, GROUPS

class AttentionBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv=nn.Linear(128,384)
        self.proj=nn.Linear(128,128)
        self.norm1=nn.LayerNorm(128);self.norm2=nn.LayerNorm(128)
        self.ff=nn.Sequential(nn.Linear(128,256),nn.ELU(),nn.Linear(256,128))
    def forward(self,x):
        b,n,_=x.shape
        q,k,v=self.qkv(x).reshape(b,n,3,4,32).permute(2,0,3,1,4).unbind(0)
        a=torch.softmax((q@k.transpose(-2,-1))*(32**-.5),dim=-1)
        x=self.norm1(x+self.proj((a@v).transpose(1,2).reshape(b,n,128)))
        return self.norm2(x+self.ff(x))

class AnatomicalActor(nn.Module):
    """All six body tokens communicate; no hand/object tokens in this scope.

    No dropout. Explicit matmul attention enables portable legacy ONNX export.
    Region-index buffers are persisted to make the input contract reviewable.
    """
    def __init__(self):
        super().__init__()
        regions=[list(range(58,70))]
        for ids in GROUPS.values():
            regions.append([off+i for off in (0,29,70,99,128) for i in ids])
        self.encoders=nn.ModuleList()
        for i,ids in enumerate(regions):
            self.register_buffer(f'indices_{i}',torch.tensor(ids,dtype=torch.long))
            self.encoders.append(nn.Sequential(nn.Linear(len(ids),128),nn.ELU(),nn.Linear(128,128)))
        self.part_embedding=nn.Parameter(torch.zeros(1,6,128))
        nn.init.normal_(self.part_embedding,std=.02)
        self.transformer=nn.Sequential(AttentionBlock(),AttentionBlock())
        # Token order: base, Lleg, Rleg, waist, Larm, Rarm.
        self.head_contexts=((0,1,3),(0,2,3),(0,3),(3,4),(3,5))
        self.heads=nn.ModuleList(nn.Sequential(nn.Linear(128*len(ctx),128),nn.ELU(),nn.Linear(128,len(ids)))
                               for ctx,ids in zip(self.head_contexts,GROUPS.values()))
    def forward(self,obs):
        tokens=torch.stack([enc(obs.index_select(-1,getattr(self,f'indices_{i}')))
                            for i,enc in enumerate(self.encoders)],dim=1)
        z=self.transformer(tokens+self.part_embedding)
        return torch.cat([head(torch.cat([z[:,i] for i in ctx],dim=-1))
                          for head,ctx in zip(self.heads,self.head_contexts)],dim=-1)

def actor_critic_class():
    """Keep the installed, version-pinned RSL-RL PPO distribution interface."""
    from rsl_rl.modules import ActorCritic
    class DexBodyActorCritic(ActorCritic):
        def __init__(self,num_actor_obs,num_critic_obs,num_actions,**kwargs):
            if (num_actor_obs,num_critic_obs,num_actions)!=(157,286,29):
                raise ValueError('Unexpected DexWeave body-only interface')
            super().__init__(num_actor_obs,num_critic_obs,num_actions,**kwargs)
            self.actor=AnatomicalActor()
    return DexBodyActorCritic

class NormalizedActor(nn.Module):
    def __init__(self,actor,mean,std,eps=0.):
        super().__init__();self.actor=actor
        self.register_buffer('mean',mean.detach().clone());self.register_buffer('std',std.detach().clone()+eps)
    def forward(self,obs):return self.actor((obs-self.mean)/self.std)
