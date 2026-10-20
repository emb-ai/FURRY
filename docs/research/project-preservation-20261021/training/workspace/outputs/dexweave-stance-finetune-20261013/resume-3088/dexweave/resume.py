"""Restore saved learning state; simulator/RNG/motion-sampling state is fresh."""
import hashlib,json,time
from pathlib import Path
import torch
EXPECTED='34c9e3a3cfc29df1d14a0b91dba01e5efba957650837b7194ade45fabfebb036'
def restore(path,policy,norm,cnorm,alg,out):
 path=Path(path);digest=hashlib.sha256(path.read_bytes()).hexdigest()
 allowed={EXPECTED:2977,'655dfc897f5a7eaeabab48fd376819c52cafa846513b8a609f2908f54c4d8928':3088}
 assert digest in allowed,digest
 ck=torch.load(path,map_location=next(policy.parameters()).device,weights_only=False)
 assert ck['iteration']==allowed[digest]
 policy.load_state_dict(ck['policy'],strict=True);norm.load_state_dict(ck['normalizer'],strict=True);cnorm.load_state_dict(ck['critic_normalizer'],strict=True)
 alg.optimizer.load_state_dict(ck['optimizer']);alg.learning_rate=float(alg.optimizer.param_groups[0]['lr'])
 for obj,key in [(policy,'policy'),(norm,'normalizer'),(cnorm,'critic_normalizer')]:
  assert all(torch.equal(v,ck[key][k].to(v.device)) for k,v in obj.state_dict().items()),key
 def same(a,b):
  if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a.cpu(),b.cpu())
  if isinstance(a,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
  if isinstance(a,(list,tuple)):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
  return a==b
 loaded=alg.optimizer.state_dict()
 # Newer local Torch adds this false default to Adam; server Torch is pinned 2.5.1.
 extras=[]
 for group,saved in zip(loaded['param_groups'],ck['optimizer']['param_groups']):
  for key in set(group)-set(saved):
   assert key=='decoupled_weight_decay' and group[key] is False,(key,group[key])
   extras.append(key);del group[key]
 assert same(loaded,ck['optimizer'])
 report=dict(checkpoint_sha256=digest,source_iteration=ck['iteration'],networks_normalizers_optimizer_exact=True,learning_rate=alg.learning_rate,physical_state_rng_motion_sampler_reset=True,exact_trajectory_resume=False)
 (Path(out)/'resume-verified.json').write_text(json.dumps(report,indent=2));return ck['iteration']
def pair_stop(directory,label,update,budget_reached,timeout=600):
 """Both branches stop at the same completed update when either hits its budget."""
 directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
 mine=directory/f'{label}-{update:05d}.json';tmp=mine.with_suffix('.tmp');tmp.write_text(json.dumps(dict(update=update,budget_reached=budget_reached)));tmp.replace(mine)
 peer=directory/f'{"B" if label=="A" else "A"}-{update:05d}.json';start=time.monotonic()
 while not peer.exists():
  if (directory/'ABORT').exists():raise RuntimeError('Paired branch failed')
  if time.monotonic()-start>timeout:raise TimeoutError('Paired branch stopped progressing')
  time.sleep(.02)
 other=json.loads(peer.read_text());assert other['update']==update
 return budget_reached or other['budget_reached']
