"""Exercise the full trainer control flow with small deterministic task fixtures.

Real Torch optimizer/GAE/checkpoint code; physical parity is tested separately.
"""
import json,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
try:
 import torch
 from torch import nn
 import train
 from evaluate import METRICS
except ImportError:
 torch=None

@unittest.skipIf(torch is None,'PyTorch environment required')
class TrainerFlowTests(unittest.TestCase):
 def run_case(self,ready):
  class Actor(nn.Module):
   def __init__(self):
    super().__init__();self.w=nn.Parameter(torch.zeros(29));self.register_buffer('mean',torch.zeros(1432));self.register_buffer('divisor',torch.ones(1432))
   def forward(self,x):return self.w.expand(len(x),29)
  class Environment:
   def __init__(self,*args,**kwargs):self.steps=0
   def reset(self,**kwargs):self.steps=0;return np.zeros(1432,np.float32)
   def critic_observation(self):return np.r_[np.zeros(1432),self.steps/10].astype(np.float32)
   def step(self,a):
    self.steps+=1;done=self.steps==3
    return np.zeros(1432,np.float32),float(.1-np.square(a).sum()),done,{'fall':False,'time_limit':False,'reference_end':done,'reward_terms':{'tracking':10.,'slip':0.}}
  def fake_warmup(critic,actor,*args,**kwargs):
   x=torch.zeros((4,1433));critic.fit_normalization(x);opt=kwargs['optimizer'];loss=(critic(x)-1).square().mean();opt.zero_grad();loss.backward();opt.step()
   return {'ready':ready}
  def fake_evaluate(*args,**kwargs):
   path=Path(args[6]);report={'dataset_manifest_sha256':'x','trace_columns':METRICS,'trials':[{'trace_key':'a','fell':False}], 'falls':0,'survived_seconds':1.,'completion_fraction':1.}
   path.write_text(json.dumps(report));report['_traces']={'a':np.ones((100,len(METRICS)))};return report
  with tempfile.TemporaryDirectory() as td:
   p=Path(td);(p/'manifest.json').write_text('{}');(p/'initial.pt').write_bytes(b'fixture');out=p/'run'
   args=['train.py','--assets',td,'--dataset',td,'--initial',str(p/'initial.pt'),'--source','fixture','--output',str(out),'--device','cpu','--envs','2','--iterations','2','--horizon','4','--eval-every','2','--critic-train-episodes','1','--critic-holdout-episodes','1','--critic-probe-episodes','1']
   px=np.zeros((32,1433),np.float32);py=np.linspace(0,1,32,dtype=np.float32)
   with patch.object(sys,'argv',args),patch.object(train,'load',side_effect=lambda *a,**k:Actor()),patch.object(train,'Env',Environment),patch.object(train,'Motions',return_value=SimpleNamespace()),patch.object(train,'warmup',side_effect=fake_warmup),patch.object(train,'evaluate',side_effect=fake_evaluate),patch.object(train,'collect_mc',return_value=(px,py,[])),patch.object(train,'critic_ready',return_value=True):train.main()
   result=json.loads((out/'result.json').read_text());checkpoint=torch.load(out/'latest.pt',weights_only=True);metrics=[json.loads(l) for l in (out/'metrics.jsonl').read_text().splitlines()]
   self.assertTrue(checkpoint['critic_optimizer']['state']) # warm-up Adam moments retained
   if ready:
    self.assertEqual(result['last_iteration'],2);ppo=[r for r in metrics if r['phase']=='ppo'];self.assertEqual(len(ppo),2);self.assertGreater(ppo[0]['actor_update_batches'],0);self.assertGreater(ppo[1]['actor_lr'],ppo[0]['actor_lr']);self.assertTrue(checkpoint['actor_optimizer']['state'])
    self.assertFalse((out/'best.pt').exists()) # equal to baseline is not an improvement
   else:
    self.assertEqual(result['stop_reason'],'critic_not_ready');self.assertEqual(result['last_iteration'],0);self.assertFalse(checkpoint['actor_optimizer']['state']);self.assertTrue(torch.equal(checkpoint['policy']['w'],torch.zeros(29)))
 def test_failed_readiness_blocks_actor(self):self.run_case(False)
 def test_ready_critic_allows_ramped_ppo_and_acceptance(self):self.run_case(True)

if __name__=='__main__':unittest.main()
