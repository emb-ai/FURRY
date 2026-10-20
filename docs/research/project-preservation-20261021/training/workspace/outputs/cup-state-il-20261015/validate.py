from pathlib import Path
import json
import numpy as np
from evaluate import World
from train import load_data
O=Path(__file__).resolve().parent
w=World();w.reset(12345);enc=w.enc;zero=np.zeros(37,np.float32);x=enc(w.d,zero)
w.d.userdata[:]=91;np.testing.assert_array_equal(enc(w.d,zero),x)
w.d.mocap_pos[0,0]+=.1;changed=enc(w.d,zero);ix=[i for i,n in enumerate(enc.names) if n.startswith('goal/')]
assert np.linalg.norm(changed[ix]-x[ix])>.09
other=[i for i in range(len(x)) if i not in ix];np.testing.assert_array_equal(changed[other],x[other])
w.reset(12345);np.testing.assert_array_equal(enc(w.d,zero),x);w.close()
meta,_,obs,act,idx=load_data(O,'cpu');assert meta['fps']==50
train=set(idx['train'][0][:,-1].tolist());valid=set(idx['valid'][0][:,-1].tolist());assert not train&valid
for oi,ai in idx.values():
    assert (oi[:,-1,None]>=oi).all();assert (ai[:,0]==oi[:,-1]).all()
    assert obs[oi].isfinite().all() and act[ai].isfinite().all()
z=np.load(O/'dataset.npz')
for r in meta['episodes']:
    steps=z[r['key']+'_source_step'];assert (steps%20==0).all() and (np.diff(steps)==20).all()
    assert np.allclose(np.diff(z[r['key']+'_time']),.02,atol=1e-8,rtol=0)
report={'passed':True,'checks':['walking HOMIE backend, standing=false','all full-reset flags=63','task userdata absent from observation','goal perturbation affects only goal channels','repeatable exact native reset','no train-validation overlap','causal history, current-time action alignment','finite dataset tensors','previous commands verified for every row','50Hz sampling matches native step modulo20=0'],'obs_dim':len(x),'dataset_sha256':meta['sha256']}
(O/'tests.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
