"""Runnable image diffusion BC baseline and dataset integration check.

This is FURRY's baseline, not a reproduction of the unreleased TWIST2 learner:
ResNet18 is initialized from scratch, and the existing FURRY denoiser is used.
The dataset itself is independent of this learner. A checkpoint is not a
validated or automatically deployed robot policy.
"""
import argparse
import copy
import json
import math
from pathlib import Path
import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18
from .diffusion import Denoiser, cosine_alpha_bar
from .visual_dataset import SCHEMA, ACTION_DIM, digest


class VisualSequences(Dataset):
    def __init__(self, path, split='train', horizon=64, obs_horizon=2, in_memory=False):
        self.path=str(path);self.horizon=horizon;self.obs_horizon=obs_horizon;self.index=[];self.cache=None
        with h5py.File(path,'r') as h:
            if h.attrs['schema']!=SCHEMA: raise ValueError('Wrong dataset schema')
            self.names=[x.decode() for x in h['mask/'+split][:]]
            actions=[]
            for name in self.names:
                a=h['data/'+name+'/actions'][:]
                if a.shape[1]!=ACTION_DIM: raise ValueError('Wrong action dimension')
                actions.append(a)
                self.index.extend((name,t) for t in range(len(a)))
            if not actions: raise ValueError(f'Empty {split} split')
            a=np.concatenate(actions)
            self.mean=a.mean(0);self.std=np.maximum(a.std(0),.01)
            if in_memory:
                self.cache={n:(h['data/'+n+'/obs/agentview_image'][:],h['data/'+n+'/obs/previous_command'][:],x)
                            for n,x in zip(self.names,actions)}
    def __len__(self): return len(self.index)
    def __getitem__(self, index):
        name,t=self.index[index]
        if self.cache is not None:
            images,previous,actions=self.cache[name];n=len(actions)
            obs_ids=np.clip(np.arange(t-self.obs_horizon+1,t+1),0,n-1)
            target_ids=np.clip(np.arange(t,t+self.horizon),0,n-1)
            return self.sample(images[obs_ids].transpose(0,3,1,2).astype('float32')/255,previous[obs_ids],actions[target_ids],t,n)
        with h5py.File(self.path,'r') as h:
            d=h['data/'+name];n=d.attrs['num_samples']
            obs_ids=np.clip(np.arange(t-self.obs_horizon+1,t+1),0,n-1)
            # Individual reads preserve repeated padded indices with HDF5.
            rgb=np.stack([d['obs/agentview_image'][i] for i in obs_ids]).transpose(0,3,1,2).astype('float32')/255
            command=np.stack([d['obs/previous_command'][i] for i in obs_ids])
            target_ids=np.clip(np.arange(t,t+self.horizon),0,n-1)
            actions=np.stack([d['actions'][i] for i in target_ids])
        return self.sample(rgb,command,actions,t,n)
    def sample(self,rgb,command,actions,t,n):
        return {'rgb':torch.from_numpy(rgb),'command':torch.from_numpy((command-self.mean)/self.std),
                'action':torch.from_numpy((actions-self.mean)/self.std),
                'mask':torch.from_numpy((np.arange(t,t+self.horizon)<n).astype('float32'))}


class UnetBlock(nn.Module):
    """Conv1d-GroupNorm-Mish x2 with FiLM (scale and bias) from the condition, as in Diffusion Policy."""
    def __init__(self,inputs,outputs,condition_dim,kernel=5,groups=8):
        super().__init__()
        conv=lambda i:nn.Sequential(nn.Conv1d(i,outputs,kernel,padding=kernel//2),nn.GroupNorm(groups,outputs),nn.Mish())
        self.blocks=nn.ModuleList((conv(inputs),conv(outputs)))
        self.condition=nn.Sequential(nn.Mish(),nn.Linear(condition_dim,outputs*2))
        self.skip=nn.Conv1d(inputs,outputs,1) if inputs!=outputs else nn.Identity()
    def forward(self,x,condition):
        scale,bias=self.condition(condition).unsqueeze(-1).chunk(2,dim=1)
        return self.blocks[1](scale*self.blocks[0](x)+bias)+self.skip(x)


class ConditionalUnet1D(nn.Module):
    """Diffusion Policy's 1D temporal U-Net (Chi et al. 2023): (batch, time, action) -> noise."""
    def __init__(self,action_dim,obs_dim,obs_horizon,dims=(256,512,1024),step_dim=256,kernel=5):
        super().__init__()
        self.step_dim=step_dim
        self.step_mlp=nn.Sequential(nn.Linear(step_dim,step_dim*4),nn.Mish(),nn.Linear(step_dim*4,step_dim))
        c=step_dim+obs_dim*obs_horizon;pairs=list(zip((action_dim,)+dims[:-1],dims))
        self.down=nn.ModuleList(nn.ModuleList((UnetBlock(i,o,c,kernel),UnetBlock(o,o,c,kernel),
            nn.Conv1d(o,o,3,2,1) if k<len(pairs)-1 else nn.Identity())) for k,(i,o) in enumerate(pairs))
        self.mid=nn.ModuleList((UnetBlock(dims[-1],dims[-1],c,kernel),UnetBlock(dims[-1],dims[-1],c,kernel)))
        self.up=nn.ModuleList(nn.ModuleList((UnetBlock(o*2,i,c,kernel),UnetBlock(i,i,c,kernel),
            nn.ConvTranspose1d(i,i,4,2,1))) for i,o in reversed(pairs[1:]))
        self.head=nn.Sequential(nn.Conv1d(dims[0],dims[0],kernel,padding=kernel//2),nn.GroupNorm(8,dims[0]),nn.Mish(),
                                nn.Conv1d(dims[0],action_dim,1))
    def forward(self,noisy,steps,observation):
        half=self.step_dim//2
        frequencies=torch.exp(-math.log(10000)*torch.arange(half,device=noisy.device)/(half-1))
        angles=steps.float()[:,None]*frequencies[None]
        condition=torch.cat((self.step_mlp(torch.cat((angles.sin(),angles.cos()),-1)),observation.flatten(1)),-1)
        x=noisy.transpose(1,2);skips=[]
        for a,b,down in self.down: x=b(a(x,condition),condition);skips.append(x);x=down(x)
        for block in self.mid: x=block(x,condition)
        for a,b,up in self.up: x=up(b(a(torch.cat((x,skips.pop()),1),condition),condition))
        return self.head(x).transpose(1,2)


class VisualDenoiser(nn.Module):
    """pretrained: ImageNet ResNet18 with frozen BatchNorm statistics and ImageNet input normalization
    (load_weights fetches them; checkpoints restore without it). crop<1: evaluation uses the centre
    crop of that fraction resized back, matching random crops in training."""
    def __init__(self,width=64,arch='film',pretrained=False,crop=1.,load_weights=False):
        super().__init__();self.pretrained=pretrained;self.crop=crop
        from torchvision.models import ResNet18_Weights
        from torchvision.ops import FrozenBatchNorm2d
        self.encoder=resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained and load_weights else None);self.encoder.fc=nn.Identity()
        # Small batches/temporally correlated frames: avoid batch-stat dependence.
        def replace(module):
            for name,child in module.named_children():
                if isinstance(child,nn.BatchNorm2d):
                    if pretrained:
                        frozen=FrozenBatchNorm2d(child.num_features,eps=child.eps)
                        for key in ('weight','bias','running_mean','running_var'): getattr(frozen,key).copy_(getattr(child,key).detach())
                        setattr(module,name,frozen)
                    else: setattr(module,name,nn.GroupNorm(8,child.num_features))
                else: replace(child)
        replace(self.encoder)
        self.register_buffer('pixel_mean',torch.tensor([.485,.456,.406]).view(1,3,1,1),persistent=False)
        self.register_buffer('pixel_std',torch.tensor([.229,.224,.225]).view(1,3,1,1),persistent=False)
        self.denoiser=(ConditionalUnet1D(ACTION_DIM,512+ACTION_DIM,2) if arch=='unet' else
                       Denoiser(ACTION_DIM,512+ACTION_DIM,2,width=width))
    def encode(self,images):
        """images (N,3,H,W) in [0,1]; training-time crops are applied by the caller."""
        if not self.training and self.crop<1:
            h,w=images.shape[-2:];ch,cw=round(h*self.crop),round(w*self.crop);y,x=(h-ch)//2,(w-cw)//2
            images=nn.functional.interpolate(images[...,y:y+ch,x:x+cw],size=(h,w),mode='bilinear',align_corners=False)
        if self.pretrained: images=(images-self.pixel_mean)/self.pixel_std
        return self.encoder(images)
    def forward(self,action,timestep,rgb,command):
        b,t=rgb.shape[:2];features=self.encode(rgb.reshape(b*t,*rgb.shape[2:])).reshape(b,t,-1)
        return self.denoiser(action,timestep,torch.cat((features,command),dim=-1))


def random_shift(rgb, pad):
    """DrQ-style shift: replicate-pad and crop back, the same offset for both frames of a sample."""
    if pad<=0: return rgb
    b,t,c,h,w=rgb.shape;x=nn.functional.pad(rgb.reshape(b,t*c,h,w),(pad,)*4,mode='replicate')
    offset=torch.randint(0,2*pad+1,(b,2)).tolist()
    return torch.stack([x[i,:,y:y+h,z:z+w] for i,(y,z) in enumerate(offset)]).reshape(b,t,c,h,w)


def augment(rgb, crop, color):
    """Random crop of a ``crop`` fraction resized back and brightness/contrast/saturation jitter of
    +-``color``; the same draw for both frames of a sample. rgb (B,T,3,H,W) in [0,1]."""
    b,t,c,h,w=rgb.shape;x=rgb.reshape(b,t*c,h,w)
    if crop<1:
        ch,cw=round(h*crop),round(w*crop)
        offset=zip(torch.randint(0,h-ch+1,(b,)).tolist(),torch.randint(0,w-cw+1,(b,)).tolist())
        x=nn.functional.interpolate(torch.stack([x[i,:,y:y+ch,z:z+cw] for i,(y,z) in enumerate(offset)]),
                                    size=(h,w),mode='bilinear',align_corners=False)
    x=x.reshape(b,t,c,h,w)
    if color>0:
        draw=lambda:(1+(torch.rand(b,1,1,1,1,device=x.device)*2-1)*color)
        gray=lambda v:(v*torch.tensor([.299,.587,.114],device=v.device).view(1,1,3,1,1)).sum(2,keepdim=True)
        x=x*draw()
        m=gray(x).mean((-1,-2),keepdim=True);x=(x-m)*draw()+m
        g=gray(x);x=(x-g)*draw()+g
        x=x.clamp(0,1)
    return x


@torch.no_grad()
def conditioning_check(model, batches, schedule, device, times=(20,50,80)):
    """Valid eps MSE at fixed noise levels and the implied x0 MSE (1.0 = unconditional mean)."""
    model.eval();result={}
    for t in times:
        total=count=0.;generator=torch.Generator().manual_seed(t)
        for batch in batches:
            action=batch['action'].to(device);noise=torch.randn(action.shape,generator=generator).to(device)
            a=schedule[t];noisy=a.sqrt()*action+(1-a).sqrt()*noise
            with torch.autocast('cuda',torch.bfloat16,enabled=device.startswith('cuda')):
                features=model.encode(batch['rgb'].to(device).flatten(0,1)).float().reshape(len(action),2,-1)
            prediction=model.denoiser(noisy,torch.full((len(action),),t,device=device),torch.cat((features,batch['command'].to(device)),-1))
            mask=batch['mask'].to(device)[...,None]
            total+=float(((prediction-noise).square()*mask).sum());count+=float(mask.sum())*ACTION_DIM
        result[f'eps_t{t}']=total/count;result[f'x0_t{t}']=total/count*float((1-schedule[t])/schedule[t])
    model.train();return result


def train(path, output, steps=10000, batch_size=8, width=64, seed=0, device='cpu', workers=0, in_memory=False,
          lr=1e-4, warmup=0, cosine=False, ema=0., shift=0, save_every=0, arch='film', weight_decay=.01, horizon=64, beta1=.9,
          pretrained=False, crop=1., color=0.):
    if steps<1: raise ValueError('Positive steps required')
    output=Path(output)
    if output.exists(): raise FileExistsError(output)
    torch.manual_seed(seed);np.random.seed(seed)
    dataset=VisualSequences(path,horizon=horizon,in_memory=in_memory)
    valid=VisualSequences(path,'valid',horizon=horizon,in_memory=in_memory);valid.mean,valid.std=dataset.mean,dataset.std
    valid_batches=list(DataLoader(torch.utils.data.Subset(valid,range(0,len(valid),8)),batch_size=64))
    loader=DataLoader(dataset,batch_size=batch_size,shuffle=True,num_workers=workers,drop_last=True,
                      persistent_workers=workers>0,pin_memory=device.startswith('cuda'))
    model=VisualDenoiser(width,arch,pretrained,crop,load_weights=True).to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=lr,betas=(beta1,.999),weight_decay=weight_decay)
    average=copy.deepcopy(model).requires_grad_(False) if ema>0 else model
    def factor(step):
        f=min(1.,(step+1)/warmup) if warmup else 1.
        return f*(.5*(1+math.cos(math.pi*step/steps)) if cosine else 1.)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,factor)
    schedule=cosine_alpha_bar(100).to(device);iterator=iter(loader);losses=[];checks={}
    output.parent.mkdir(parents=True,exist_ok=True)
    def save(target,step):
        report={'schema':'furry-visual-dp-baseline-v1','dataset_sha256':digest(path),'steps':step,
                'train_episodes':dataset.names,'samples':len(dataset),'action_dim':ACTION_DIM,'obs_horizon':2,'prediction_horizon':horizon,
                'encoder':'ResNet18 ImageNet, frozen BatchNorm' if pretrained else 'ResNet18 from scratch (not R3M)',
                'pretrained':pretrained,'crop':crop,'color':color,'arch':arch,'width':width,'weight_decay':weight_decay,'beta1':beta1,'seed':seed,'batch_size':batch_size,
                'lr':lr,'warmup':warmup,'cosine':cosine,'ema':ema,'shift':shift,
                'model_weights':'EMA' if ema>0 else 'raw','loss_first':losses[0],'loss_last':losses[-1],
                'loss_last_1000':float(np.mean(losses[-1000:])),'valid_conditioning':checks,
                'autonomous_evaluation':False,'deployed':False}
        torch.save({'model':average.state_dict(),'raw_model':model.state_dict() if ema>0 else None,
                    'optimizer':optimizer.state_dict(),'action_mean':torch.from_numpy(dataset.mean),
                    'action_std':torch.from_numpy(dataset.std),'metadata':report},target)
        target.with_suffix('.json').write_text(json.dumps(report,indent=2));return report
    for step in range(steps):
        try: batch=next(iterator)
        except StopIteration: iterator=iter(loader);batch=next(iterator)
        batch={k:v.to(device,non_blocking=True) for k,v in batch.items()};action=batch['action']
        noise=torch.randn_like(action);times=torch.randint(0,len(schedule),(len(action),),device=device)
        alpha=schedule[times,None,None]
        noisy=alpha.sqrt()*action+(1-alpha).sqrt()*noise
        rgb=augment(random_shift(batch['rgb'],shift),crop,color)
        # bf16 only for the image encoder; the action denoiser stays in float32.
        with torch.autocast('cuda',torch.bfloat16,enabled=device.startswith('cuda')):
            features=model.encode(rgb.flatten(0,1)).float().reshape(len(action),rgb.shape[1],-1)
        prediction=model.denoiser(noisy,times,torch.cat((features,batch['command']),dim=-1))
        loss=((prediction-noise).square()*batch['mask'][...,None]).sum()/(batch['mask'].sum()*ACTION_DIM)
        if not torch.isfinite(loss): raise RuntimeError('Nonfinite training loss')
        optimizer.zero_grad();loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not torch.isfinite(norm): raise RuntimeError('Nonfinite gradient')
        optimizer.step();scheduler.step();losses.append(float(loss.detach()))
        if ema>0:
            decay=min(ema,(1+step)/(10+step))
            torch._foreach_lerp_(list(average.parameters()),list(model.parameters()),1-decay)
        if step==0 or (step+1)%100==0: print(f'step={step+1} loss={losses[-1]:.6f} lr={scheduler.get_last_lr()[0]:.2e}',flush=True)
        if save_every and (step+1)%save_every==0 and step+1<steps:
            checks[step+1]=conditioning_check(average,valid_batches,schedule,device)
            print(f'step={step+1} valid {json.dumps(checks[step+1])}',flush=True)
            save(output.with_name(f'{output.stem}-{(step+1)//1000}k.pt'),step+1)
    checks[steps]=conditioning_check(average,valid_batches,schedule,device)
    print(f'step={steps} valid {json.dumps(checks[steps])}',flush=True)
    return save(output,steps)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('dataset');p.add_argument('output')
    p.add_argument('--steps',type=int,default=10000);p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--width',type=int,default=64);p.add_argument('--device',default='cpu');p.add_argument('--threads',type=int,default=4)
    p.add_argument('--workers',type=int,default=0,help='DataLoader workers');p.add_argument('--in-memory',action='store_true',help='load all frames into RAM')
    p.add_argument('--lr',type=float,default=1e-4);p.add_argument('--warmup',type=int,default=0);p.add_argument('--cosine',action='store_true')
    p.add_argument('--ema',type=float,default=0.,help='EMA decay of saved weights (0 = raw weights)')
    p.add_argument('--shift',type=int,default=0,help='random image shift in pixels');p.add_argument('--save-every',type=int,default=0)
    p.add_argument('--arch',choices=('film','unet'),default='film',help='film: small FURRY denoiser; unet: Diffusion Policy ConditionalUnet1D')
    p.add_argument('--weight-decay',type=float,default=.01);p.add_argument('--beta1',type=float,default=.9)
    p.add_argument('--horizon',type=int,default=64,help='predicted chunk length (unet needs a multiple of 4)')
    p.add_argument('--pretrained',action='store_true',help='ImageNet ResNet18 (frozen BatchNorm) instead of from scratch')
    p.add_argument('--crop',type=float,default=1.,help='random crop fraction in training, centre crop in evaluation')
    p.add_argument('--color',type=float,default=0.,help='brightness/contrast/saturation jitter amplitude')
    a=p.parse_args();torch.set_num_threads(a.threads)
    print(json.dumps(train(a.dataset,a.output,a.steps,a.batch_size,a.width,device=a.device,workers=a.workers,in_memory=a.in_memory,
                           lr=a.lr,warmup=a.warmup,cosine=a.cosine,ema=a.ema,shift=a.shift,save_every=a.save_every,
                           arch=a.arch,weight_decay=a.weight_decay,horizon=a.horizon,beta1=a.beta1,
                           pretrained=a.pretrained,crop=a.crop,color=a.color),indent=2))


if __name__=='__main__': main()
