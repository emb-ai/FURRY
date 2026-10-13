"""Recover the released actor exactly; critic/optimizer are newly initialized."""
import importlib.util
from pathlib import Path
import numpy as np
import torch
from torch import nn

def upstream_actor(source):
 spec=importlib.util.spec_from_file_location('twist_actor_source',source);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 return mod.ActorFuture(num_observations=1432,num_motion_observations=35,num_priop_observations=92,num_motion_steps=1,num_future_observations=35,num_future_steps=1,motion_latent_dim=128,future_latent_dim=128,num_actions=29,actor_hidden_dims=[512,512,256,128],activation=nn.SiLU(),history_latent_dim=128,num_history_steps=10,layer_norm=True)
class Policy(nn.Module):
 def __init__(self,source):
  super().__init__();self.actor=upstream_actor(source);self.register_buffer('mean',torch.zeros(1432));self.register_buffer('divisor',torch.ones(1432))
 def forward(self,obs):return self.actor((obs-self.mean)/self.divisor)
 def train(self,mode=True):
  super().train(mode);self.actor.eval() # PPO log-probability must not change due to Dropout masks.
  return self

def recover(onnx_path,source):
 import onnx
 from onnx import numpy_helper
 graph=onnx.load(str(onnx_path));weights={x.name:numpy_helper.to_array(x).copy() for x in graph.graph.initializer};p=Policy(source)
 p.actor.load_state_dict({k[len('actor.') :]:torch.from_numpy(v) for k,v in weights.items() if k.startswith('actor.')},strict=True)
 # Resolve normalization through graph edges, not exporter-generated tensor names.
 sub=next(n for n in graph.graph.node if n.op_type=='Sub' and n.input[0]=='input');div=next(n for n in graph.graph.node if n.op_type=='Div' and n.input[0]==sub.output[0])
 p.mean.copy_(torch.from_numpy(weights[sub.input[1]]));p.divisor.copy_(torch.from_numpy(weights[div.input[1]]))
 if not torch.all(p.divisor>0):raise ValueError('Invalid normalization')
 return p.eval()
def load(path,source,device='cpu'):
 p=Policy(source);p.load_state_dict(torch.load(path,map_location='cpu',weights_only=True)['policy']);return p.to(device).eval()

def bootstrap(onnx_path,source,destination):
 import onnxruntime as ort,json,hashlib
 torch.set_num_threads(1);p=recover(onnx_path,source);rng=np.random.default_rng(20261008)
 obs=np.concatenate([np.zeros((1,1432),np.float32),(p.mean.numpy()+rng.normal(size=(511,1432))*.5*p.divisor.numpy()).astype(np.float32)])
 opts=ort.SessionOptions();opts.intra_op_num_threads=1;session=ort.InferenceSession(str(onnx_path),sess_options=opts,providers=['CPUExecutionProvider'])
 with torch.no_grad():pred=p(torch.from_numpy(obs)).numpy()
 expected=session.run(None,{session.get_inputs()[0].name:obs})[0];error=float(abs(pred-expected).max())
 if not np.allclose(pred,expected,rtol=2e-4,atol=2e-4):raise ValueError(f'Actor parity failed: {error}')
 torch.save({'policy':p.state_dict(),'source_onnx_sha256':hashlib.sha256(Path(onnx_path).read_bytes()).hexdigest(),'critic_restored':False},destination)
 report={'max_absolute_error':error,'samples':len(obs),'actor_parameters':sum(x.numel() for x in p.parameters()),'critic_restored':False,'normalizer':'retained from released ONNX, frozen'}
 Path(str(destination)+'.json').write_text(json.dumps(report,indent=2));print(report)
if __name__=='__main__':
 import argparse
 a=argparse.ArgumentParser();a.add_argument('onnx');a.add_argument('source');a.add_argument('output');x=a.parse_args();bootstrap(x.onnx,x.source,x.output)
