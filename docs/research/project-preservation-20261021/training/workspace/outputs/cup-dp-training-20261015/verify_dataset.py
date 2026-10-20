from pathlib import Path
import json,h5py,numpy as np
from train import fit_normalizer,Sequences,sha
O=Path(__file__).resolve().parent;p=O/'cup44-no-shadow-causal.hdf5';report=json.loads((O/'dataset-report.json').read_text());assert report['sha256']==sha(p)
with h5py.File(p,'r') as h:
 train={x.decode() for x in h['mask/train'][:]};valid={x.decode() for x in h['mask/valid'][:]};assert len(train)==40 and len(valid)==4 and not train&valid
 n=0
 for name,d in h['data'].items():
  a=d['actions'][:];prev=d['obs/previous_command'][:];assert a.shape[1]==39 and np.isfinite(a).all();np.testing.assert_array_equal(prev[1:],a[:-1]);assert np.all(prev[0]==0)
  ts=d['timestamps'][:];src=d['source_sim_time'][:];assert np.all(src<=ts+1e-8);assert np.all(np.diff(ts)>0);assert d['obs/agentview_image'].shape==(len(a),224,224,3)
  for k in range(0,len(a),32):
   rgb=d['obs/agentview_image'][k:k+32];assert rgb.std()>1
  n+=len(a)
 assert n==12982
 r={'episodes':44,'train_episodes':40,'valid_episodes':4,'frames':n,'sha256':sha(p),'split_disjoint':True,'previous_command_exact':True,'finite_actions':True,'causal_sampling':True,'all_images_decoded':True}
(O/'dataset-verification.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
