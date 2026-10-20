"""Bounded state-only imitation pilot; whole-episode split, train-only statistics."""
from pathlib import Path
import argparse,copy,hashlib,json,math,os,time
import numpy as np
import torch
from models import make,scheduler,HISTORY,HORIZON,ACTION_DIM

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(obj,p):
    tmp=p.with_suffix('.partial');torch.save(obj,tmp);tmp.replace(p)

def load_data(root,device):
    meta=json.loads((root/'dataset.json').read_text());assert sha(root/'dataset.npz')==meta['sha256']
    z=np.load(root/'dataset.npz');allobs=[];allact=[];inds={s:[] for s in ('train','valid')};offset=0
    for row in meta['episodes']:
        key=row['key'];obs=z[key+'_obs'];act=z[key+'_action'];n=len(obs)
        assert np.isfinite(obs).all() and np.isfinite(act).all()
        assert np.array_equal(obs[1:,-ACTION_DIM:],act[:-1]) and (obs[0,-ACTION_DIM:]==0).all()
        allobs.append(obs);allact.append(act)
        for t in range(n):inds[row['split']].append((np.clip(np.arange(t-HISTORY+1,t+1),0,n-1)+offset,np.clip(np.arange(t,t+HORIZON),0,n-1)+offset))
        offset+=n
    trainobs=np.concatenate([z[r['key']+'_obs'] for r in meta['episodes'] if r['split']=='train'])
    trainact=np.concatenate([z[r['key']+'_action'] for r in meta['episodes'] if r['split']=='train'])
    stats={'obs_mean':trainobs.mean(0),'obs_std':np.maximum(trainobs.std(0),.05),'action_mean':trainact.mean(0),'action_std':np.maximum(trainact.std(0),.05)}
    obs=torch.tensor((np.concatenate(allobs)-stats['obs_mean'])/stats['obs_std'],device=device)
    act=torch.tensor((np.concatenate(allact)-stats['action_mean'])/stats['action_std'],device=device)
    idx={s:(torch.tensor(np.stack([x[0] for x in rows]),device=device),torch.tensor(np.stack([x[1] for x in rows]),device=device)) for s,rows in inds.items()}
    return meta,stats,obs,act,idx

def main():
    p=argparse.ArgumentParser();p.add_argument('--kind',choices=['bc-rnn','diffusion'],required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--steps',type=int,default=12000);p.add_argument('--seconds',type=int,default=1800);p.add_argument('--batch',type=int,default=128);p.add_argument('--device',default='cuda');p.add_argument('--seed',type=int,default=1510)
    a=p.parse_args();root=Path(__file__).resolve().parent;assert not a.out.exists();a.out.mkdir()
    torch.set_num_threads(4);torch.manual_seed(a.seed);np.random.seed(a.seed)
    if a.device=='cuda':assert torch.cuda.is_available();torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    meta,stats,obs,act,idx=load_data(root,a.device);model=make(a.kind,obs.shape[-1]).to(a.device);ema=copy.deepcopy(model)
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-5);sched=scheduler();initial=next(model.parameters()).detach().clone()
    info={'kind':a.kind,'seed':a.seed,'dataset_sha256':meta['sha256'],'train_episodes':meta['train'],'valid_episodes':meta['valid'],'obs_dim':obs.shape[-1],
          'history':HISTORY,'horizon':HORIZON,'execute':4,'fps':50,'parameters':sum(x.numel() for x in model.parameters()),'max_steps':a.steps,'max_seconds':a.seconds,
          'device':torch.cuda.get_device_name() if a.device=='cuda' else a.device,'gpu_uuid':os.environ.get('CUDA_VISIBLE_DEVICES'),'lower_policy':meta['family'],
          'source_sha256':{f:sha(root/f) for f in ['models.py','train.py','state.py','export.py']},'normalization':'train-only mean/std, floor .05','checkpoint_selection':'EMA generated action normalized MSE on 256 fixed held-out windows, not denoising loss',
          'deployed':False}
    (a.out/'metadata.json').write_text(json.dumps(info,indent=2));print(json.dumps(info),flush=True)
    start=time.monotonic();best=float('inf');losses=[]
    for step in range(1,a.steps+1):
        model.train();oi,ai=idx['train'];sel=torch.randint(len(oi),(a.batch,),device=a.device);x=obs[oi[sel]];y=act[ai[sel]]
        x=x+.01*torch.randn_like(x)
        if a.kind=='bc-rnn':loss=(model(x)-y).square().mean()
        else:
            noise=torch.randn_like(y);t=torch.randint(0,100,(len(y),),device=a.device);loss=(model(x,sched.add_noise(y,noise,t),t)-noise).square().mean()
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        rate=3e-4*min(1.,step/200)*(.1+.9*.5*(1+math.cos(math.pi*min(step/a.steps,1))))
        for g in opt.param_groups:g['lr']=rate
        opt.step();decay=min(.995,(1+step)/(10+step))
        with torch.no_grad():
            for ep,mp in zip(ema.parameters(),model.parameters()):ep.lerp_(mp,1-decay)
        losses.append(float(loss.detach()))
        final=step==a.steps or time.monotonic()-start>=a.seconds
        if step==3:
            changed=float((next(model.parameters())-initial).abs().max().detach());assert changed>0
            (a.out/'optimizer-check.json').write_text(json.dumps({'step':step,'max_abs_update':changed,'finite_loss':True}))
        if step==1 or step%100==0 or final:
            status={'step':step,'loss':float(np.mean(losses)),'elapsed_s':time.monotonic()-start,'lr':rate};losses=[]
            (a.out/'status.json').write_text(json.dumps(status));print(json.dumps(status),flush=True)
        if step%1000==0 or final:
            ema.eval();oi,ai=idx['valid'];sel=torch.linspace(0,len(oi)-1,min(256,len(oi)),device=a.device).long();errors=[];physical=[]
            with torch.random.fork_rng(devices=[torch.cuda.current_device()] if a.device=='cuda' else []),torch.inference_mode():
                torch.manual_seed(9191)
                for ss in sel.split(64):
                    pred=ema.predict(obs[oi[ss]]);truth=act[ai[ss]];errors.append((pred-truth).square().mean().item());physical.append(((pred-truth)*torch.tensor(stats['action_std'],device=a.device)).abs().mean((0,1)).cpu().numpy())
            score=float(np.mean(errors));mae=np.mean(physical,axis=0)
            check={'step':step,'elapsed_s':time.monotonic()-start,'generated_mse':score,'arm_mae_deg':float(np.degrees(mae[21:35]).mean()),'grip_mae':float(mae[35:37].mean())}
            with (a.out/'validation.jsonl').open('a') as f:f.write(json.dumps(check)+'\n')
            ck={'kind':a.kind,'step':step,'model':model.state_dict(),'ema':ema.state_dict(),'stats':stats,'metadata':info,'validation':check}
            save(ck,a.out/'latest.pt')
            if score<best:best=score;save(ck,a.out/'best.pt')
            print(json.dumps({'validation':check}),flush=True)
        if final:break
    (a.out/'complete.json').write_text(json.dumps({'status':'completed','steps':step,'elapsed_s':time.monotonic()-start,'best_generated_mse':best,'latest_sha256':sha(a.out/'latest.pt'),'best_sha256':sha(a.out/'best.pt'),'task_success_evaluated':False},indent=2))
if __name__=='__main__':main()
