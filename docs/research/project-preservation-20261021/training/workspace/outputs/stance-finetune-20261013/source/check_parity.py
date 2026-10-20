from pathlib import Path
import sys,json,hashlib
import numpy as np
R=Path.cwd();O=R/'outputs/stance-finetune-20261013';onnx=R/'.tools/furry-main-integration-20261013/prototypes/quest_g1/policies/quest-111586/checkpoint_000500-50hz.onnx';ck=O/'initial/checkpoint_000500.pt'
if sys.argv[1]=='torch':
 import torch
 sys.path.insert(0,str(R/'.tools/furry-training/prototypes/quest_g1/training'))
 from policy import load
 torch.set_num_threads(1);m=load(ck,R/'outputs/quest-finetune-20261008/bundle/upstream/actor_critic_future.py');rng=np.random.default_rng(7);x=(m.mean.numpy()+rng.normal(size=(1024,1432))*.5*m.divisor.numpy()).astype('float32')
 with torch.no_grad():y=m(torch.from_numpy(x)).numpy()
 np.savez(O/'initial/parity-samples.npz',x=x,y=y)
else:
 import onnxruntime as ort
 d=np.load(O/'initial/parity-samples.npz');opt=ort.SessionOptions();opt.intra_op_num_threads=1;s=ort.InferenceSession(str(onnx),sess_options=opt,providers=['CPUExecutionProvider']);z=s.run(None,{s.get_inputs()[0].name:d['x']})[0];err=float(abs(d['y']-z).max());assert np.allclose(d['y'],z,rtol=2e-4,atol=2e-4);meta=s.get_modelmeta().custom_metadata_map;assert meta['g1_policy_hz']=='50'
 report={'checkpoint_sha256':hashlib.sha256(ck.read_bytes()).hexdigest(),'published_onnx':str(onnx),'onnx_sha256':hashlib.sha256(onnx.read_bytes()).hexdigest(),'samples':len(z),'max_absolute_error':err,'metadata':meta};(O/'initial/parity.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
