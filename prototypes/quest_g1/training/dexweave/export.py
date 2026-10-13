"""Export our own checkpoint with empirical normalization and verify parity."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import torch
import onnxruntime as ort
from rsl_rl.modules import EmpiricalNormalization
from .network import AnatomicalActor,NormalizedActor

def main():
    p=argparse.ArgumentParser();p.add_argument('checkpoint',type=Path);p.add_argument('output',type=Path);args=p.parse_args()
    torch.set_num_threads(1)
    # Only load locally created/trusted experiment checkpoints.
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    actor=AnatomicalActor();actor.load_state_dict({k[len('actor.'):]:v for k,v in checkpoint['policy'].items() if k.startswith('actor.')})
    norm=EmpiricalNormalization([157]);norm.load_state_dict(checkpoint['normalizer']);norm.eval()
    model=NormalizedActor(actor,norm.mean,norm.std,norm.eps).eval()
    torch.manual_seed(812);sample=torch.randn(8,157)*norm.std+norm.mean
    args.output.parent.mkdir(exist_ok=True,parents=True)
    # Torch 2.5 supports this legacy exporter by default; the later Mac
    # environment needs the explicit dynamo=False switch.
    kwargs={'dynamo':False} if int(torch.__version__.split('.')[1])>=6 else {}
    torch.onnx.export(model,sample,str(args.output),input_names=['obs'],output_names=['actions'],
                      dynamic_axes={'obs':{0:'batch'},'actions':{0:'batch'}},opset_version=17,**kwargs)
    session=ort.InferenceSession(str(args.output),providers=['CPUExecutionProvider']);max_error=0.
    for n in (1,7,32,128):
        x=torch.randn(n,157)*norm.std+norm.mean
        with torch.no_grad():expected=model(x).numpy()
        actual=session.run(None,{'obs':x.numpy()})[0];max_error=max(max_error,float(abs(actual-expected).max()))
        np.testing.assert_allclose(actual,expected,atol=5e-6,rtol=5e-5)
    report={'checkpoint_sha256':hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),'onnx_sha256':hashlib.sha256(args.output.read_bytes()).hexdigest(),
            'iteration':checkpoint['iteration'],'input_dim':157,'action_dim':29,'normalization_eps':norm.eps,'parity_max_abs_error':max_error,
            'samples':168,'trained_reference_sha256':checkpoint['contract']['reference_sha256'],'smoke':checkpoint['contract']['smoke']}
    if checkpoint['contract'].get('reference_kind')=='training_manifest':
        report['training_manifest_sha256']=report.pop('trained_reference_sha256')
        report['training_references']=checkpoint['contract']['training_references']
    args.output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))

if __name__=='__main__':main()
