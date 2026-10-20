from pathlib import Path
import hashlib,json,sys
import numpy as np
import torch

ROOT=Path('/home/jovyan/vlad/FURRY/cup-fast-check-20261015')
sys.path.insert(0,str(ROOT))
from train_act import ACTConfig,ACTPolicy,FeatureType,PolicyFeature,CupDataset

torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
torch.manual_seed(0)
import argparse
p=argparse.ArgumentParser();p.add_argument('--device',default='cuda');p.add_argument('--data',type=Path,default=ROOT.parent/'cup-native50-20261015/cup-native50.hdf5');p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
out=args.out
out.mkdir(parents=True,exist_ok=True)
checkpoint=args.source
c=torch.load(checkpoint,map_location='cpu',weights_only=False)
cfg=ACTConfig(input_features={'observation.images.ego':PolicyFeature(FeatureType.VISUAL,(3,224,224)),'observation.state':PolicyFeature(FeatureType.STATE,(c['metadata'].get('state_dim',39),))},output_features={'action':PolicyFeature(FeatureType.ACTION,(39,))},chunk_size=64,n_action_steps=8,device=args.device,pretrained_backbone_weights=None)
policy=ACTPolicy(cfg).to(args.device).eval()
policy.load_state_dict(c['model'],strict=True)

class Export(torch.nn.Module):
    def __init__(self,p):
        super().__init__();self.policy=p
    def forward(self,image,state):
        return self.policy.predict_action_chunk({'observation.images.ego':image,'observation.state':state})

model=Export(policy).eval()
dataset=CupDataset(args.data,'train',c['stats'])
if c['metadata'].get('state_dim',39)==131:
    from proprio import ProprioDataset
    dataset=ProprioDataset(dataset,c['stats']['obs/proprio'])
samples=[dataset[i] for i in [0,len(dataset)//2,len(dataset)-1]]
inputs=[(s['observation.images.ego'][None].to(args.device),s['observation.state'][None].to(args.device)) for s in samples]
with torch.inference_mode():
    traced=torch.jit.trace(model,inputs[0],check_inputs=inputs[1:])
    traced=torch.jit.freeze(traced)
    errors=[float((model(*x)-traced(*x)).abs().max()) for x in inputs]
    assert max(errors)<1e-4,errors
    traced.save(str(out/'act-inference.pt'))
    np.savez(out/'parity.npz',image=inputs[0][0].cpu().numpy(),state=inputs[0][1].cpu().numpy(),expected=model(*inputs[0]).cpu().numpy())
stats={k:{kk:vv.tolist() for kk,vv in v.items()} for k,v in c['stats'].items()}
(out/'stats.json').write_text(json.dumps(stats))
report={'checkpoint':str(checkpoint),'checkpoint_step':c['step'],'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),'export_sha256':hashlib.sha256((out/'act-inference.pt').read_bytes()).hexdigest(),'parity_max_abs':errors,'metadata':c['metadata'],'execution_steps':8,'deterministic_latent':'zero, official inference','config':cfg.to_dict() if hasattr(cfg,'to_dict') else str(cfg)}
(out/'export.json').write_text(json.dumps(report,indent=2,default=str))
print(json.dumps({k:report[k] for k in ['checkpoint_step','checkpoint_sha256','export_sha256','parity_max_abs']}),flush=True)
