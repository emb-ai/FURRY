from pathlib import Path
import json,torch
from evaluate import O,inspect_episode,run
from train import Policy
if __name__=='__main__':
 torch.set_num_threads(4);ck=torch.load(O/'latest-ema.pt',map_location='cpu',weights_only=False);norm=ck['normalizer'];model=Policy(None,norm,load_pretrained=False);model.load_state_dict(ck['ema']);model.eval().to('mps');del ck
 manifest=json.loads((O.parent/'cup-training-plan-20261015/training-manifest.json').read_text());results=[]
 for split,h in [('train',48),('valid',48),('train',8),('valid',8)]:
  row=manifest['splits'][split][0];e=inspect_episode(Path(manifest['raw_root'])/row['episode']);r=run(e,'policy',model,norm,'mps',0,90,h,'latest');r['split']=split;results.append(r);(O/'results-latest-suite.json').write_text(json.dumps(results,indent=2))
