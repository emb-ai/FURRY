"""One original-simulator finetune with independent MuJoCo baseline/checkpoints.

Runs inside one Slurm allocation. No automatic promotion or headset deployment.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--num-envs',type=int,default=4096);p.add_argument('--iterations',type=int,default=1000)
p.add_argument('--init-checkpoint',type=Path);p.add_argument('--init-manifest',type=Path)
p.add_argument('--window-reward',action='store_true');p.add_argument('--sole-urdf',type=Path);p.add_argument('--eval-assets',type=Path);p.add_argument('--eval-dataset',type=Path)
a=p.parse_args()
if bool(a.init_checkpoint) != bool(a.init_manifest):p.error('--init-checkpoint and --init-manifest must be provided together')
root=a.root.resolve();out=root/'runs'/os.environ['SLURM_JOB_ID'];out.mkdir(parents=True,exist_ok=False)
old=Path.home()/'furry/quest-adapt-v3-night-20261008-r3'
py=Path.home()/'furry/quest-adapt-20261008/venv/bin/python'
# Use one explicit, immutable runtime/dataset variant for both baseline and candidates.
env=dict(os.environ,PYTHONPATH=str(old/'training'),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
base_cmd=[str(py),str(root/'adapter/author_eval.py'),'--assets',str(a.eval_assets or old/'assets'),'--dataset',str(a.eval_dataset or old/'dataset'),'--source',str(old/'upstream/actor_critic_future.py')]
if a.window_reward:base_cmd+=['--window-metrics']
def evaluate(policy,dest,baseline=None):
    cmd=base_cmd+['--policy',str(policy),'--output',str(dest)]
    if baseline:cmd+=['--baseline',str(baseline)]
    with dest.with_suffix('.log').open('w') as log:subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
if a.eval_dataset:
    from checkpoint_lineage import audit_evaluation
    evaluation_audit=audit_evaluation(json.loads((root/'dataset/manifest.json').read_text()),a.eval_dataset)
    (out/'evaluation-split-audit.json').write_text(json.dumps(evaluation_audit,indent=2))
(out/'pipeline-config.json').write_text(json.dumps({k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},indent=2))
if a.init_checkpoint:
    import torch
    from checkpoint_lineage import audit
    ancestor=torch.load(a.init_checkpoint,map_location='cpu',weights_only=True)
    lineage=audit(ancestor,a.init_manifest,json.loads((root/'dataset/manifest.json').read_text()))
    (out/'initialization-audit.json').write_text(json.dumps(lineage,indent=2))
    del ancestor
baseline=out/'baseline.json'
evaluate(a.init_checkpoint or root/'TWIST2/assets/ckpts/twist2_1017_20k.onnx',baseline)
train_out=out/'training'
cmd=[sys.executable,str(root/'adapter/author_finetune.py'),'--root',str(root),'--output',str(train_out),'--headless','--num_envs',str(a.num_envs),'--iterations',str(a.iterations)]
if a.init_checkpoint:cmd+=['--init-checkpoint',str(a.init_checkpoint),'--init-manifest',str(a.init_manifest)]
if a.window_reward:cmd+=['--window-reward']
if a.sole_urdf:cmd+=['--sole-urdf',str(a.sole_urdf)]
results=[];done=set()
with (out/'training.log').open('w') as log:
    proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
    try:
        while True:
            for checkpoint in sorted(train_out.glob('checkpoint_*.pt')):
                if checkpoint.name in done:continue
                dest=out/(checkpoint.stem+'.json')
                evaluate(checkpoint,dest,baseline)
                verdict=json.loads(dest.with_suffix('.comparison.json').read_text())
                results.append({'checkpoint':checkpoint.name,**verdict});done.add(checkpoint.name)
                (out/'runtime-results.json').write_text(json.dumps(results,indent=2))
                print(json.dumps({'checkpoint':checkpoint.name,'accepted':verdict['accepted'],'reasons':verdict['reasons']}),flush=True)
            if proc.poll() is not None:
                if proc.returncode:raise RuntimeError(f'Trainer exited {proc.returncode}; inspect training.log')
                if any(c.name not in done for c in train_out.glob('checkpoint_*.pt')):continue
                break
            time.sleep(20)
        if proc.returncode:raise RuntimeError(f'Trainer exited {proc.returncode}; inspect training.log')
    finally:
        if proc.poll() is None:
            (train_out/'STOP').touch()
            try:proc.wait(timeout=60)
            except subprocess.TimeoutExpired:proc.terminate()
status={'phase':'pipeline_complete','results':results,'training_output':str(train_out),'deployment':False,'validation':'12 held-out Quest clips, seeds 0/1; explicit eval-dataset and eval-assets used identically for baseline/checkpoints. Previously inspected validation, not an independent test.','selection':'Report baseline acceptance only; no automatic deployment.'}
(out/'result.json').write_text(json.dumps(status,indent=2));print(json.dumps(status),flush=True)
