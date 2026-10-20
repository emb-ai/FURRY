"""Measured robot state only: excludes cup and goal; keeps previous command features."""
import numpy as np

def features(q, v):
 q=np.asarray(q);v=np.asarray(v);assert q.shape[-1]==57 and v.shape[-1]==55
 w,x,y,z=np.moveaxis(q[...,3:7],-1,0)
 # World down expressed in the pelvis frame; quaternion wxyz.
 gravity=np.stack([2*(w*y-x*z),-2*(y*z+w*x),2*(x*x+y*y)-1],axis=-1)
 return np.concatenate([q[...,7:50],v[...,6:49],v[...,3:6],gravity],axis=-1).astype(np.float32)

def statistics(path):
 import h5py
 with h5py.File(path) as h:
  xs=[features(d['obs/physical_qpos'][:],d['obs/physical_qvel'][:]) for d in h['data'].values()]
 x=np.concatenate(xs);assert x.shape[1]==92 and np.isfinite(x).all()
 return {'mean':x.mean(0),'std':np.maximum(x.std(0),1e-3)}

class ProprioDataset:
 def __init__(self,base,stats):
  import h5py
  self.base=base;self.names=base.names;self.values={}
  with h5py.File(base.path) as h:
   for name in self.names:
    d=h['data'][name];self.values[name]=(features(d['obs/physical_qpos'][:],d['obs/physical_qvel'][:])-stats['mean'])/stats['std']
 def __len__(self):return len(self.base)
 def __getitem__(self,i):
  import torch
  b=self.base[i];name,t=self.base.index[i];b['observation.state']=torch.cat([b['observation.state'],torch.from_numpy(self.values[name][t])]);return b
