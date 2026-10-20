"""Cup image diffusion BC: pinned iDP3 U-Net/EMA, pretrained R3M, FURRY adapter.

This is an author-style adaptation, not the unreleased exact TWIST2 learner.
No robot rollout success is inferred from denoising validation loss.
"""
from pathlib import Path
import argparse,copy,hashlib,json,math,os,random,sys,time
import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
from torchvision.models import resnet18
from torchvision import transforms as T
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
PREDICTION_TYPE = "epsilon"
O=Path(__file__).resolve().parent
sys.path.insert(0,str(O/'vendor'))
from diffusion_policy_3d.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy_3d.model.diffusion.ema_model import EMAModel

def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for c in iter(lambda:f.read(1024*1024),b''):h.update(c)
 return h.hexdigest()

def fit_normalizer(path):
 with h5py.File(path,'r') as h:
  names=[s.decode() for s in h['mask/train'][:]]
  out={}
  for key in ['actions','obs/previous_command']:
   x=np.concatenate([h['data/'+n+'/'+key][:] for n in names]);lo=x.min(0);hi=x.max(0);span=hi-lo;constant=span<1e-4
   scale=2/np.where(constant,2,span);offset=-1-scale*lo;offset[constant]=-lo[constant]
   out[key]={'scale':scale.astype('float32'),'offset':offset.astype('float32'),'std':x.std(0).astype('float32'),'min':lo,'max':hi}
 return out

class Sequences(Dataset):
 def __init__(self,path,split,norm,horizon=64,action_steps=48):
  self.path=str(path);self.norm=norm;self.horizon=horizon;self.episodes={};self.index=[];self.handle=None
  with h5py.File(path,'r') as h:
   assert h.attrs['schema']=='furry-visual-il-v1'
   self.names=[s.decode() for s in h['mask/'+split][:]]
   for name in self.names:
    d=h['data/'+name];a=d['actions'][:];c=d['obs/previous_command'][:]
    self.episodes[name]=(a,c);self.index.extend((name,t) for t in range(max(0,len(a)-horizon+action_steps+1)))
 def __len__(self):return len(self.index)
 def __getitem__(self,k):
  if self.handle is None:self.handle=h5py.File(self.path,'r')
  name,t=self.index[k];a,c=self.episodes[name];n=len(a)
  # Upstream alignment: observation slots t-1,t; prediction starts t-1.
  # Execute prediction[1:49], i.e. current t through t+47.
  obs=np.clip([t-1,t],0,n-1);act=np.clip(np.arange(t-1,t-1+self.horizon),0,n-1)
  image=np.stack([self.handle['data/'+name+'/obs/agentview_image'][int(i)] for i in obs]).transpose(0,3,1,2)
  na=self.norm['actions'];nc=self.norm['obs/previous_command']
  return {'image':torch.from_numpy(image.copy()),'command':torch.from_numpy(c[obs]*nc['scale']+nc['offset']),
          'action':torch.from_numpy(a[act]*na['scale']+na['offset'])}

class Policy(nn.Module):
 def __init__(self,weights,norm,load_pretrained=True):
  super().__init__();self.encoder=resnet18(weights=None);self.encoder.fc=nn.Identity()
  if load_pretrained:
   state=torch.load(weights,map_location='cpu',weights_only=True)['r3m']
   prefix='module.convnet.';enc={k[len(prefix):]:v for k,v in state.items() if k.startswith(prefix)}
   self.encoder.load_state_dict(enc,strict=True)
  self.augment=nn.Sequential(T.RandomCrop(212),T.Resize(224,antialias=True),T.RandomRotation((-5,5)),T.ColorJitter(.3,.4,.5,.08))
  self.register_buffer('rgb_mean',torch.tensor([.485,.456,.406])[None,:,None,None]);self.register_buffer('rgb_std',torch.tensor([.229,.224,.225])[None,:,None,None])
  self.register_buffer('command_noise_std',torch.from_numpy(.1*norm['obs/previous_command']['std']*norm['obs/previous_command']['scale']))
  self.register_buffer('action_scale',torch.from_numpy(norm['actions']['scale']));self.register_buffer('action_offset',torch.from_numpy(norm['actions']['offset']))
  self.unet=ConditionalUnet1D(input_dim=39,local_cond_dim=None,global_cond_dim=2*(512+39),diffusion_step_embed_dim=128,down_dims=[256,512,1024],kernel_size=5,n_groups=8,condition_type='film')
 def encode(self,b):
  im=b['image'].float()/255.;batch,obs=im.shape[:2];im=im.flatten(0,1)
  if self.training:im=self.augment(im)
  features=self.encoder((im-self.rgb_mean)/self.rgb_std).reshape(batch,-1)
  command=b['command']
  if self.training:command=command+torch.randn_like(command)*self.command_noise_std
  # Match upstream TimmObsEncoder: flattened RGB history then command history.
  return torch.cat([features,command.reshape(batch,-1)],-1)
 def forward(self,noisy,t,b):return self.unet(noisy,t,global_cond=self.encode(b))
 @torch.no_grad()
 def predict(self,b,scheduler):
  self.eval();cond=self.encode(b);x=torch.randn((len(cond),64,39),device=cond.device)
  scheduler.set_timesteps(10,device=cond.device)
  for t in scheduler.timesteps:
   pred=self.unet(x,t,global_cond=cond);x=scheduler.step(pred,t,x).prev_sample
  return (x[:,1:49]-self.action_offset)/self.action_scale

@torch.no_grad()
def update_ema(ema,model):
 ema.step(model)
 # Upstream EMA updates parameters only. R3M has BatchNorm buffers, which
 # must follow the trained encoder for validation/inference to be meaningful.
 source=dict(model.named_buffers())
 for name,buffer in ema.averaged_model.named_buffers():buffer.copy_(source[name])

def scheduler():return DDIMScheduler(num_train_timesteps=50,beta_start=.0001,beta_end=.02,beta_schedule='squaredcos_cap_v2',clip_sample=True,set_alpha_to_one=True,steps_offset=0,prediction_type=PREDICTION_TYPE)
def loss(model,b,sched):
 noise=torch.randn_like(b['action']);t=torch.randint(0,50,(len(noise),),device=noise.device)
 noisy=sched.add_noise(b['action'],noise,t)
 target=noise if PREDICTION_TYPE=='epsilon' else b['action']
 return (model(noisy,t,b)-target).square().mean()
def move(b,device):return {k:v.to(device,non_blocking=True) for k,v in b.items()}
def save_atomic(obj,path):
 tmp=path.with_suffix('.partial');torch.save(obj,tmp);tmp.replace(path)

def main():
 global PREDICTION_TYPE
 p=argparse.ArgumentParser();p.add_argument('--prediction-type',choices=['epsilon','sample'],default='epsilon');p.add_argument('--data',type=Path,default=O/'cup44-no-shadow-causal.hdf5');p.add_argument('--weights',type=Path,default=O/'weights/r3m-resnet18-encoder.pt');p.add_argument('--out',type=Path,default=O/'run-seed0');p.add_argument('--device',default='cuda');p.add_argument('--epochs',type=int,default=301);p.add_argument('--batch-size',type=int,default=32);p.add_argument('--workers',type=int,default=4);p.add_argument('--seed',type=int,default=0);p.add_argument('--max-steps',type=int,default=0);p.add_argument('--resume',action='store_true');p.add_argument('--threads',type=int,default=8);args=p.parse_args();PREDICTION_TYPE=args.prediction_type
 torch.set_num_threads(args.threads);torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
 if args.out.exists() and (args.out/'latest.pt').exists() and not args.resume:raise FileExistsError(args.out)
 args.out.mkdir(parents=True,exist_ok=True)
 norm=fit_normalizer(args.data);train=Sequences(args.data,'train',norm);valid=Sequences(args.data,'valid',norm)
 assert not(set(train.names)&set(valid.names))
 loader=DataLoader(train,batch_size=args.batch_size,shuffle=True,num_workers=args.workers,pin_memory=args.device.startswith('cuda'),persistent_workers=args.workers>0)
 val=DataLoader(valid,batch_size=args.batch_size,shuffle=False,num_workers=0)
 model=Policy(args.weights,norm).to(args.device);ema_model=copy.deepcopy(model);ema=EMAModel(ema_model,power=.75)
 initial_encoder=model.encoder.conv1.weight.detach().clone();initial_unet=next(model.unet.parameters()).detach().clone()
 optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,betas=(.95,.999),eps=1e-8,weight_decay=1e-6)
 total=args.epochs*len(loader)
 def lr_mult(s):return float(s)/500 if s<500 else max(0.,.5*(1+math.cos(math.pi*(s-500)/max(1,total-500))))
 lr=torch.optim.lr_scheduler.LambdaLR(optimizer,lr_mult);sched=scheduler();step=0;start_epoch=0;best=float('inf');started=time.time()
 metadata={'schema':'furry-cup-idp3-r3m-v1','upstream_commit':'f5b27faabbb8952672dd3d6a21d0ba762d5dfabb','dataset_sha256':sha(args.data),'prediction_type':PREDICTION_TYPE,'r3m_sha256':sha(args.weights),'args':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},'train_episodes':len(train.names),'valid_episodes':len(valid.names),'train_sequences':len(train),'valid_sequences':len(valid),'parameters':sum(x.numel() for x in model.parameters()),'autonomous_evaluation':False,'deployed':False,'render_shadows':False,'normalization':'train-only limits, action and command separately','proprio_noise':'0.1 * training standard deviation per channel, train only','execution':'64 predicted; indices1:49 execute 48 steps at30Hz'}
 (args.out/'metadata.json').write_text(json.dumps(metadata,indent=2));print(json.dumps(metadata),flush=True)
 if args.resume:
  ck=torch.load(args.out/'latest.pt',map_location=args.device,weights_only=False)
  assert ck['metadata']['dataset_sha256']==metadata['dataset_sha256']
  assert ck['metadata']['prediction_type']==PREDICTION_TYPE
  model.load_state_dict(ck['model']);ema_model.load_state_dict(ck['ema']);optimizer.load_state_dict(ck['optimizer']);lr.load_state_dict(ck['lr']);step=ck['step'];start_epoch=ck['epoch']+1;best=ck['best'];ema.optimization_step=step
  torch.set_rng_state(ck['rng'].cpu());np.random.set_state(ck['np_rng']);random.setstate(ck['py_rng'])
  if 'cuda_rng' in ck and args.device.startswith('cuda'):torch.cuda.set_rng_state_all([state.cpu() for state in ck['cuda_rng']])
  (args.out/'resume-verification.json').write_text(json.dumps({'checkpoint_epoch':ck['epoch'],'checkpoint_step':step,'prediction_type':PREDICTION_TYPE,'optimizer_loaded':True,'ema_loaded':True,'scheduler_loaded':True,'cpu_cuda_numpy_python_rng_loaded':True,'lr':lr.get_last_lr()},indent=2))
 for epoch in range(start_epoch,args.epochs):
  model.train();loss_sum=0.;count=0
  for b in loader:
   b=move(b,args.device);optimizer.zero_grad(set_to_none=True);l=loss(model,b,sched)
   if not torch.isfinite(l):raise RuntimeError('Nonfinite loss')
   l.backward()
   # No extra clipping/early-stop heuristic; retain upstream optimization.
   if not torch.stack([torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]).all():raise RuntimeError('Nonfinite gradient')
   optimizer.step();lr.step();update_ema(ema,model);step+=1;loss_sum+=float(l.detach())*len(b['action']);count+=len(b['action'])
   if step==3:
    changes={'encoder_max_abs_update':float((model.encoder.conv1.weight-initial_encoder).abs().max().detach()),'unet_max_abs_update':float((next(model.unet.parameters())-initial_unet).abs().max().detach()),'step':step}
    assert changes['encoder_max_abs_update']>0 and changes['unet_max_abs_update']>0
    (args.out/'optimizer-check.json').write_text(json.dumps(changes,indent=2));print(json.dumps(changes),flush=True)
   if step==1 or step%50==0:print(json.dumps({'step':step,'epoch':epoch,'loss':float(l.detach()),'lr':lr.get_last_lr()[0],'elapsed_s':time.time()-started}),flush=True)
   if args.max_steps and step>=args.max_steps:break
  result={'epoch':epoch,'step':step,'train_loss':loss_sum/count,'elapsed_s':time.time()-started}
  if epoch%5==0 or epoch==args.epochs-1 or (args.max_steps and step>=args.max_steps):
   ema_model.eval();s=0.;n=0
   # Loss magnitudes differ by parameterization: do not rank epsilon vs sample by this loss.
   # Fixed validation noise; preserve training RNG sequence.
   devices=[torch.device(args.device).index or 0] if args.device.startswith('cuda') else []
   with torch.random.fork_rng(devices=devices),torch.no_grad():
    torch.manual_seed(12345)
    for vb in val:
     vb=move(vb,args.device);vl=loss(ema_model,vb,sched);s+=float(vl)*len(vb['action']);n+=len(vb['action'])
    result['validation_noise_mse']=s/n
    pred=ema_model.predict({k:v[:1] for k,v in vb.items()},scheduler());assert pred.shape==(1,48,39) and torch.isfinite(pred).all()
   if result['validation_noise_mse']<best:
    best=result['validation_noise_mse'];save_atomic({'ema':ema_model.state_dict(),'normalizer':norm,'metadata':metadata,'epoch':epoch,'step':step,'validation_noise_mse':best},args.out/'best.pt')
  if epoch%10==0 or epoch==args.epochs-1:
   save_atomic({'ema':ema_model.state_dict(),'normalizer':norm,'metadata':metadata,'epoch':epoch,'step':step},args.out/f'ema-epoch{epoch:03d}.pt')
  checkpoint={'model':model.state_dict(),'ema':ema_model.state_dict(),'optimizer':optimizer.state_dict(),'lr':lr.state_dict(),'normalizer':norm,'metadata':metadata,'epoch':epoch,'step':step,'best':best,'rng':torch.get_rng_state(),'np_rng':np.random.get_state(),'py_rng':random.getstate()}
  if args.device.startswith('cuda'):checkpoint['cuda_rng']=torch.cuda.get_rng_state_all()
  if epoch%10==0 or epoch==args.epochs-1 or (args.max_steps and step>=args.max_steps):
   save_atomic(checkpoint,args.out/'latest.pt')
  with (args.out/'metrics.jsonl').open('a') as f:f.write(json.dumps(result)+'\n')
  (args.out/'status.json').write_text(json.dumps(dict(result,status='running',best_validation_noise_mse=best),indent=2));print(json.dumps(result),flush=True)
  if args.max_steps and step>=args.max_steps:break
 (args.out/'status.json').write_text(json.dumps(dict(result,status='completed',best_validation_noise_mse=best,autonomous_evaluation=False),indent=2))
if __name__=='__main__':main()
