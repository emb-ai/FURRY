"""Export candidate only after numerical agreement with its PyTorch actor."""
import argparse,json
from pathlib import Path
import numpy as np,torch,onnxruntime as ort
from policy import load
p=argparse.ArgumentParser();p.add_argument('checkpoint');p.add_argument('source');p.add_argument('output');a=p.parse_args();torch.set_num_threads(1);model=load(a.checkpoint,a.source)
example=torch.zeros(1,1432);torch.onnx.export(model,example,a.output,input_names=['input'],output_names=['output'],dynamic_axes={'input':{0:'batch'},'output':{0:'batch'}},opset_version=17,dynamo=False)
rng=np.random.default_rng(7);x=(model.mean.numpy()+rng.normal(size=(128,1432))*.5*model.divisor.numpy()).astype(np.float32);opts=ort.SessionOptions();opts.intra_op_num_threads=1;s=ort.InferenceSession(a.output,sess_options=opts,providers=['CPUExecutionProvider'])
with torch.no_grad():expected=model(torch.from_numpy(x)).numpy()
actual=s.run(None,{'input':x})[0];error=float(abs(actual-expected).max());assert np.allclose(actual,expected,rtol=2e-4,atol=2e-4),error;Path(a.output+'.parity.json').write_text(json.dumps({'max_absolute_error':error,'samples':128}));print('ONNX parity',error)
