"""Runnable image diffusion BC baseline and dataset integration check.

This is FURRY's baseline, not a reproduction of the unreleased TWIST2 learner:
ResNet18 is initialized from scratch, and the existing FURRY denoiser is used.
The dataset itself is independent of this learner. A checkpoint is not a
validated or automatically deployed robot policy.
"""
import argparse
import json
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
    def __init__(self, path, split='train', horizon=64, obs_horizon=2):
        self.path=str(path);self.horizon=horizon;self.obs_horizon=obs_horizon;self.index=[]
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
    def __len__(self): return len(self.index)
    def __getitem__(self, index):
        name,t=self.index[index]
        with h5py.File(self.path,'r') as h:
            d=h['data/'+name];n=d.attrs['num_samples']
            obs_ids=np.clip(np.arange(t-self.obs_horizon+1,t+1),0,n-1)
            # Individual reads preserve repeated padded indices with HDF5.
            rgb=np.stack([d['obs/agentview_image'][i] for i in obs_ids]).transpose(0,3,1,2).astype('float32')/255
            command=np.stack([d['obs/previous_command'][i] for i in obs_ids])
            target_ids=np.clip(np.arange(t,t+self.horizon),0,n-1)
            actions=np.stack([d['actions'][i] for i in target_ids])
        return {'rgb':torch.from_numpy(rgb),'command':torch.from_numpy((command-self.mean)/self.std),
                'action':torch.from_numpy((actions-self.mean)/self.std),
                'mask':torch.from_numpy((np.arange(t,t+self.horizon)<n).astype('float32'))}


class VisualDenoiser(nn.Module):
    def __init__(self,width=64):
        super().__init__();self.encoder=resnet18(weights=None);self.encoder.fc=nn.Identity()
        # Small batches/temporally correlated frames: avoid batch-stat dependence.
        def replace(module):
            for name,child in module.named_children():
                if isinstance(child,nn.BatchNorm2d): setattr(module,name,nn.GroupNorm(8,child.num_features))
                else: replace(child)
        replace(self.encoder)
        self.denoiser=Denoiser(ACTION_DIM,512+ACTION_DIM,2,width=width)
    def forward(self,action,timestep,rgb,command):
        b,t=rgb.shape[:2];features=self.encoder(rgb.reshape(b*t,*rgb.shape[2:])).reshape(b,t,-1)
        return self.denoiser(action,timestep,torch.cat((features,command),dim=-1))


def train(path, output, steps=10000, batch_size=8, width=64, seed=0, device='cpu'):
    if steps<1: raise ValueError('Positive steps required')
    output=Path(output)
    if output.exists(): raise FileExistsError(output)
    torch.manual_seed(seed);np.random.seed(seed)
    dataset=VisualSequences(path)
    loader=DataLoader(dataset,batch_size=batch_size,shuffle=True,num_workers=0)
    model=VisualDenoiser(width).to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4)
    schedule=cosine_alpha_bar(100).to(device);iterator=iter(loader);losses=[]
    for step in range(steps):
        try: batch=next(iterator)
        except StopIteration: iterator=iter(loader);batch=next(iterator)
        batch={k:v.to(device) for k,v in batch.items()};action=batch['action']
        noise=torch.randn_like(action);times=torch.randint(0,len(schedule),(len(action),),device=device)
        alpha=schedule[times,None,None]
        noisy=alpha.sqrt()*action+(1-alpha).sqrt()*noise
        prediction=model(noisy,times,batch['rgb'],batch['command'])
        loss=((prediction-noise).square()*batch['mask'][...,None]).sum()/(batch['mask'].sum()*ACTION_DIM)
        if not torch.isfinite(loss): raise RuntimeError('Nonfinite training loss')
        optimizer.zero_grad();loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not torch.isfinite(norm): raise RuntimeError('Nonfinite gradient')
        optimizer.step();losses.append(float(loss.detach()))
        if step==0 or (step+1)%100==0: print(f'step={step+1} loss={losses[-1]:.6f}',flush=True)
    output.parent.mkdir(parents=True,exist_ok=True)
    report={'schema':'furry-visual-dp-baseline-v1','dataset_sha256':digest(path),'steps':steps,
            'train_episodes':dataset.names,'samples':len(dataset),'action_dim':ACTION_DIM,'obs_horizon':2,'prediction_horizon':64,
            'encoder':'ResNet18 from scratch (not R3M)','width':width,'seed':seed,'loss_first':losses[0],
            'loss_last':losses[-1],'autonomous_evaluation':False,'deployed':False}
    torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'action_mean':torch.from_numpy(dataset.mean),
                'action_std':torch.from_numpy(dataset.std),'metadata':report},output)
    output.with_suffix('.json').write_text(json.dumps(report,indent=2));return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('dataset');p.add_argument('output')
    p.add_argument('--steps',type=int,default=10000);p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--width',type=int,default=64);p.add_argument('--device',default='cpu');p.add_argument('--threads',type=int,default=4)
    a=p.parse_args();torch.set_num_threads(a.threads)
    print(json.dumps(train(a.dataset,a.output,a.steps,a.batch_size,a.width,device=a.device),indent=2))


if __name__=='__main__': main()
