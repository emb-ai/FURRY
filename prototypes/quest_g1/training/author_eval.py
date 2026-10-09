"""Separate-process runtime evaluation; validation is never exposed to PPO."""
import argparse
import json
from pathlib import Path
import numpy as np
from evaluate import evaluate, compare
from rewards import RewardConfig

p=argparse.ArgumentParser()
for name in ['assets','dataset','policy','source','output']:p.add_argument('--'+name,required=True,type=Path)
p.add_argument('--baseline',type=Path)
a=p.parse_args()
report=evaluate(a.assets,a.dataset,a.policy,a.source,split='validation',seeds=(0,1),output=a.output,reward_config=RewardConfig(slide_weight=2.))
if a.baseline:
    base=json.loads(a.baseline.read_text())
    with np.load(a.baseline.with_suffix('.npz'),allow_pickle=False) as z:base['_traces']={k:z[k] for k in z.files}
    verdict=compare(base,report)
    a.output.with_suffix('.comparison.json').write_text(json.dumps(verdict,indent=2))
    print(json.dumps({'phase':'runtime_comparison',**verdict}),flush=True)
