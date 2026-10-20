"""Overfit diagnostic: fresh R3M/U-Net, one original train episode, no augmentation.

The original train-only normalization is retained to isolate the learning path.
This experiment is deliberately not a generalization or validation benchmark.
"""
from pathlib import Path
import argparse,copy,json,sys,time
import h5py,numpy as np,torch
O=Path(__file__).resolve().parent
ROOT=O.parent if (O.parent/'train.py').exists() else O.parent/'cup-dp-training-20261015'
sys.path.insert(0,str(ROOT))
from train import Policy,Sequences,fit_normalizer,loss,scheduler,EMAModel,update_ema

def main():
 p=argparse.ArgumentParser();p.add_argument('--steps',type=int,default=3000);p.add_argument('--out',type=Path,default=O/'single-run');args=p.parse_args()
 torch.set_num_threads(4);torch.manual_seed(0);np.random.seed(0)
 args.out.mkdir(exist_ok=False)
 data=ROOT/'cup44-no-shadow-causal.hdf5';norm=fit_normalizer(data);seq=Sequences(data,'train',norm);name=seq.names[0]
 seq.index=[x for x in seq.index if x[0]==name];samples=[seq[i] for i in range(len(seq))]
 batch={k:torch.stack([x[k] for x in samples]).to('cuda') for k in samples[0]}
 with h5py.File(data) as h:episode=h['data/'+name].attrs['episode']
 model=Policy(ROOT/'weights/r3m-resnet18-encoder.pt',norm).to('cuda');model.augment=torch.nn.Identity();model.command_noise_std.zero_()
 avg=copy.deepcopy(model);ema=EMAModel(avg,power=.75);opt=torch.optim.AdamW(model.parameters(),lr=1e-4,betas=(.95,.999),eps=1e-8,weight_decay=1e-6)
 sched=scheduler();ids=torch.tensor(sorted(set([0,30,60,90,len(samples)//2,len(samples)-1])),device='cuda');ids=ids[ids<len(samples)]
 test={k:v[ids] for k,v in batch.items()};truth=(test['action'][:,1:49]-model.action_offset)/model.action_scale
 best=float('inf');start=time.time()
 meta=dict(experiment='one-episode-overfit-diagnostic',episode=episode,windows=len(samples),steps=args.steps,augmentation=False,command_noise=False,normalizer='original 40 train episodes',initialization='fresh U-Net + public pretrained R3M',generalization_claim=False)
 (args.out/'spec.json').write_text(json.dumps(meta,indent=2))
 with (args.out/'metrics.jsonl').open('w',buffering=1) as f:
  for step in range(1,args.steps+1):
   model.train();take=torch.randint(len(samples),(32,),device='cuda');b={k:v[take] for k,v in batch.items()};opt.zero_grad(set_to_none=True);l=loss(model,b,sched)
   if not torch.isfinite(l):raise RuntimeError('Nonfinite loss')
   l.backward();opt.step();update_ema(ema,model)
   if step==1 or step%100==0 or step==args.steps:
    record=dict(step=step,loss=float(l.detach()),elapsed_s=time.time()-start)
    with torch.random.fork_rng(devices=[0]),torch.no_grad():
     for label,net in [('raw',model),('ema',avg)]:
      torch.manual_seed(123);pred=net.predict(test,scheduler());err=pred-truth
      record[label]=dict(arms_deg=float(err[:,:,21:35].abs().mean()*180/np.pi),grip_mae=float(err[:,:,35:37].abs().mean()),initial_grip=pred[0,:,35:37].cpu().tolist())
      score=float((err*model.action_scale).square().mean())
      if score<best:
       best=score;torch.save(dict(ema=net.state_dict(),normalizer=norm,metadata=meta,step=step,epoch=None,validation_noise_mse=None,diagnostic_generated_mse=score,selected=label),args.out/'best.pt')
    f.write(json.dumps(record)+'\n');print(json.dumps(dict({k:v for k,v in record.items() if k not in ['raw','ema']},arms_deg=record['ema']['arms_deg'],grip_mae=record['ema']['grip_mae'])),flush=True)
    (args.out/'status.json').write_text(json.dumps(dict(status='completed' if step==args.steps else 'running',**record),indent=2))
if __name__=='__main__':main()
