"""DexWeave directed anatomy/object attention, adapted to 7+7 hand drivers."""
import torch
from torch import nn
from .spec import GROUPS,ACTOR_DIM,CRITIC_DIM

class Block(nn.Module):
    def __init__(self,mask):
        super().__init__();self.register_buffer('mask',mask)
        self.qkv=nn.Linear(128,384);self.proj=nn.Linear(128,128)
        self.norm1=nn.LayerNorm(128);self.norm2=nn.LayerNorm(128)
        self.ff=nn.Sequential(nn.Linear(128,256),nn.ELU(),nn.Linear(256,128))
    def forward(self,x):
        b,n,_=x.shape;q,k,v=self.qkv(x).reshape(b,n,3,4,32).permute(2,0,3,1,4).unbind(0)
        scores=(q@k.transpose(-1,-2))*(32**-.5)
        attn=torch.softmax(scores.masked_fill(~self.mask,float('-inf')),-1)
        x=self.norm1(x+self.proj((attn@v).transpose(1,2).reshape(b,n,128)))
        return self.norm2(x+self.ff(x))
def mlp(i,o):return nn.Sequential(nn.Linear(i,128),nn.ELU(),nn.Linear(128,o))

class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        # qref43,dqref29,anchor history21,orientation6,omega3,q43,dq29,previous43,object27.
        regions=[list(range(72,102))]
        for ids in GROUPS:
            regions.append([off+i for off in (0,102,174) for i in ids]+([43+i for i in ids]+[145+i for i in ids] if max(ids)<29 else []))
        self.encoders=nn.ModuleList()
        for i,ids in enumerate(regions):
            self.register_buffer(f'indices_{i}',torch.tensor(ids));self.encoders.append(mlp(len(ids),128))
        self.embedding=nn.Parameter(torch.randn(1,8,128)*.02)
        mask=torch.zeros(8,8,dtype=torch.bool);mask[:6,:6]=True
        for i in (6,7):mask[i,[3,4,5,i]]=True
        self.robot=nn.Sequential(Block(mask),Block(mask))
        self.object_encoder=mlp(27,128);self.object_embedding=nn.Parameter(torch.randn(1,1,128)*.02)
        fusion=torch.eye(6,dtype=torch.bool);fusion[:,5]=True
        self.fusion=Block(fusion)
        self.contexts=((0,1,3),(0,2,3),(0,3),(3,4),(3,5),(4,6),(5,7))
        self.heads=nn.ModuleList(mlp(len(ctx)*128,len(ids)) for ctx,ids in zip(self.contexts,GROUPS))
    def forward(self,obs):
        z=self.robot(torch.stack([e(obs.index_select(-1,getattr(self,f'indices_{i}'))) for i,e in enumerate(self.encoders)],1)+self.embedding)
        fused=self.fusion(torch.cat([z[:,3:],self.object_encoder(obs[:,-27:])[:,None]+self.object_embedding],1))
        upper=torch.cat([z[:,:3],fused[:,:5]],1)
        return torch.cat([head(torch.cat([(z if h<2 else upper)[:,i] for i in ctx],-1)) for h,(head,ctx) in enumerate(zip(self.heads,self.contexts))],-1)

def actor_critic_class():
    from rsl_rl.modules import ActorCritic
    class DexCupActorCritic(ActorCritic):
        def __init__(self,num_actor_obs,num_critic_obs,num_actions,**kwargs):
            assert (num_actor_obs,num_critic_obs,num_actions)==(ACTOR_DIM,CRITIC_DIM,43)
            super().__init__(num_actor_obs,num_critic_obs,num_actions,**kwargs);self.actor=Actor()
    return DexCupActorCritic
