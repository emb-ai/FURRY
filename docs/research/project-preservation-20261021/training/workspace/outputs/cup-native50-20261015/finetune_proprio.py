"""New-data fine-tuning from completed FURRY checkpoints; no demo validation split."""
from pathlib import Path
import argparse,copy,hashlib,json,os,sys,time,random
import numpy as np
import torch
from torch.utils.data import DataLoader

def save(obj,path):
 tmp=path.with_suffix('.partial');torch.save(obj,tmp);tmp.replace(path)

def main():
 p=argparse.ArgumentParser();p.add_argument('--kind',choices=['act','dp-epsilon','dp-sample'],required=True);p.add_argument('--base',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--steps',type=int,default=20000);p.add_argument('--lr',type=float,default=1e-5);p.add_argument('--clip-range',type=float,default=1.);p.add_argument('--save-every',type=int,default=5000);a=p.parse_args()
 sys.path.insert(0,str(a.base));a.out.mkdir(exist_ok=False);torch.set_num_threads(4);torch.manual_seed(42);np.random.seed(42);random.seed(42)
 ck=torch.load(a.source,map_location='cpu',weights_only=False);source_step=int(ck['step']);device='cuda';ema=None
 metadata={'kind':a.kind,'source_checkpoint':str(a.source),'source_step':source_step,'source_sha256':hashlib.sha256(a.source.read_bytes()).hexdigest(),'dataset_sha256':hashlib.sha256(a.data.read_bytes()).hexdigest(),'new_data_finetune':True,'normalization':'frozen source normalization','optimizer':'restored source Adam moments; new constant learning rate','lr':a.lr,'target_updates':a.steps,'validation':'10 fixed random native layouts; no validation demonstrations','clip_sample_range':a.clip_range,'autonomous_evaluation':False}
 if a.kind=='act':
  from train_act import ACTConfig,ACTPolicy,FeatureType,PolicyFeature,CupDataset
  from proprio import statistics,ProprioDataset
  norm=copy.deepcopy(ck['stats']);norm['obs/proprio']=statistics(a.data);ds=ProprioDataset(CupDataset(a.data,'train',norm),norm['obs/proprio']);cfg=ACTConfig(input_features={'observation.images.ego':PolicyFeature(FeatureType.VISUAL,(3,224,224)),'observation.state':PolicyFeature(FeatureType.STATE,(131,))},output_features={'action':PolicyFeature(FeatureType.ACTION,(39,))},chunk_size=64,n_action_steps=8,device=device,pretrained_backbone_weights=None)
  model=ACTPolicy(cfg).to(device);adapted={}
  for k,v in model.state_dict().items():
   src=ck['model'][k]
   if v.shape!=src.shape:
    assert k.endswith('robot_state_input_proj.weight') and src.shape==(512,39) and v.shape==(512,131),(k,src.shape,v.shape)
    expanded=torch.zeros_like(v,device='cpu');expanded[:,:39]=src;adapted[k]=expanded
   else:adapted[k]=src
  model.load_state_dict(adapted,strict=True);optcfg=cfg.get_optimizer_preset();optim=optcfg.build(model.get_optim_params());metadata['upstream_commit']=ck['metadata']['upstream_commit'];metadata['state_semantics']='previous39commands + measured43robot_joints +43joint_velocities +3root_angular_velocity +3projected_gravity; no object state';metadata['state_dim']=131;metadata['expanded_input_layers']=[k for k in adapted if adapted[k].shape!=ck['model'][k].shape]
 else:
  import train_dp_resume as dp
  dp.PREDICTION_TYPE=a.kind.split('-')[1];assert ck['metadata']['prediction_type']==dp.PREDICTION_TYPE
  norm=ck['normalizer'];ds=dp.Sequences(a.data,'train',norm);model=dp.Policy(None,norm,load_pretrained=False).to(device);model.load_state_dict(ck['model'],strict=True);emodel=copy.deepcopy(model);emodel.load_state_dict(ck['ema'],strict=True);ema=dp.EMAModel(emodel,power=.75);ema.optimization_step=source_step
  optim=torch.optim.AdamW(model.parameters(),lr=a.lr,betas=(.95,.999),eps=1e-8,weight_decay=1e-6);sched=dp.scheduler();sched.register_to_config(clip_sample_range=a.clip_range);metadata['prediction_type']=dp.PREDICTION_TYPE
 assert all(torch.equal(v.detach().cpu(),adapted[k]) for k,v in model.state_dict().items())
 # Zero-added columns preserve old initial inference for every measured state.
 assert all(torch.count_nonzero(adapted[k][:,39:])==0 for k in metadata['expanded_input_layers'])
 optim.load_state_dict(ck['optimizer'])
 for param,st in optim.state.items():
  for key in ['exp_avg','exp_avg_sq']:
   if key in st and st[key].shape!=param.shape:
    old=st[key];assert old.shape==(512,39) and param.shape==(512,131);new=torch.zeros_like(param);new[:,:39]=old;st[key]=new
 metadata['optimizer']='source Adam restored; added input columns have zero moments'
 metadata['normalization']='source normalization preserved; new92 measured features normalized by train-only statistics'
 oldcfg=ACTConfig(input_features={'observation.images.ego':PolicyFeature(FeatureType.VISUAL,(3,224,224)),'observation.state':PolicyFeature(FeatureType.STATE,(39,))},output_features={'action':PolicyFeature(FeatureType.ACTION,(39,))},chunk_size=64,n_action_steps=8,device=device,pretrained_backbone_weights=None)
 old=ACTPolicy(oldcfg).to(device).eval();old.load_state_dict(ck['model'],strict=True);model.eval();errors=[]
 prior_tf32=torch.backends.cuda.matmul.allow_tf32;prior_cudnn=torch.backends.cudnn.allow_tf32;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 with torch.inference_mode():
  for idx in [0,len(ds)//2,len(ds)-1]:
   sample=ds[idx];b={k:v[None].to(device) for k,v in sample.items()};orig=dict(b);orig['observation.state']=b['observation.state'][:,:39];errors.append(float((model.predict_action_chunk(b)-old.predict_action_chunk(orig)).abs().max()))
 assert max(errors)<1e-4,errors
 metadata['initial_prediction_parity_max_abs']=errors;torch.backends.cuda.matmul.allow_tf32=prior_tf32;torch.backends.cudnn.allow_tf32=prior_cudnn;del old;torch.cuda.empty_cache();ds.base.handle.close();ds.base.handle=None
 for g in optim.param_groups:g['lr']=a.lr
 import h5py
 with h5py.File(a.data,'r') as h:
  assert h.attrs['fps']==50
 metadata['action_hz']=50;metadata['sampling']='native policy phase step modulo20=0';metadata['chunk_duration_s']=64/50;metadata['execution_duration_s']=8/50
 metadata['train_episodes']=ds.names;metadata['valid_episodes']=[];metadata['train_samples']=len(ds)
 (a.out/'metadata.json').write_text(json.dumps(metadata,indent=2));(a.out/'initialization-verification.json').write_text(json.dumps({'model_exact':False,'old_weights_exact':True,'new_input_columns_zero':True,'optimizer_loaded':True,'source_normalization_exact':True,'new_measured_state_normalization':'train-only mean/std','ema_loaded':ema is not None,'source_step':source_step},indent=2));del ck
 loader=DataLoader(ds,batch_size=32,shuffle=True,num_workers=2,persistent_workers=True,pin_memory=True);it=iter(loader);started=time.time();initial=next(model.parameters()).detach().clone()
 for step in range(1,a.steps+1):
  try:b=next(it)
  except StopIteration:it=iter(loader);b=next(it)
  b={k:v.to(device,non_blocking=True) for k,v in b.items()};model.train();optim.zero_grad(set_to_none=True)
  loss=model(b)[0] if a.kind=='act' else dp.loss(model,b,sched)
  if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
  loss.backward()
  if a.kind=='act':torch.nn.utils.clip_grad_norm_(model.parameters(),optcfg.grad_clip_norm,error_if_nonfinite=True)
  elif not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):raise RuntimeError('Nonfinite gradient')
  optim.step()
  if ema is not None:dp.update_ema(ema,model)
  if step==1 or step%100==0 or step==a.steps:
   r={'status':'running','kind':a.kind,'pid':os.getpid(),'step':step,'source_step':source_step,'target_steps':a.steps,'loss':float(loss.detach()),'elapsed_s':time.time()-started};(a.out/'status.json').write_text(json.dumps(r));print(json.dumps(r),flush=True)
  if step==1 or step%a.save_every==0 or step==a.steps:
   c={'model':model.state_dict(),'optimizer':optim.state_dict(),'metadata':metadata,'step':step,'source_step':source_step,'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'np_rng':np.random.get_state(),'py_rng':random.getstate()}
   if ema is None:c['stats']=norm
   else:c.update(ema=emodel.state_dict(),normalizer=norm,ema_optimization_step=ema.optimization_step,epoch=step//len(loader))
   save(c,a.out/'latest.pt')
   if step%a.save_every==0:save({k:v for k,v in c.items() if k not in ['optimizer','rng','cuda_rng','np_rng','py_rng']},a.out/f'weights-{step:06d}.pt')
   with (a.out/'metrics.jsonl').open('a') as f:f.write(json.dumps(r)+'\n')
  if step==10:
   change=float((next(model.parameters()).detach()-initial).abs().max());assert change>0;(a.out/'optimizer-check.json').write_text(json.dumps({'step':step,'parameter_max_abs_change':change,'new_input_column_max_abs':{k:float(model.state_dict()[k][:,39:].abs().max()) for k in metadata['expanded_input_layers']}}))
 r['status']='completed';r['autonomous_evaluation']=False;(a.out/'status.json').write_text(json.dumps(r));print(json.dumps(r),flush=True)

if __name__=='__main__':main()
