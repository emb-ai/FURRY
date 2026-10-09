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
a=p.parse_args();root=a.root.resolve();out=root/'runs'/os.environ['SLURM_JOB_ID'];out.mkdir(parents=True,exist_ok=False)
old=Path.home()/'furry/quest-adapt-v3-night-20261008-r3'
py=Path.home()/'furry/quest-adapt-20261008/venv/bin/python'
# Use immutable runtime physics and the exact original Quest validation files.
env=dict(os.environ,PYTHONPATH=str(old/'training'),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
base_cmd=[str(py),str(root/'adapter/author_eval.py'),'--assets',str(old/'assets'),'--dataset',str(old/'dataset'),'--source',str(old/'upstream/actor_critic_future.py')]
def evaluate(policy,dest,baseline=None):
    cmd=base_cmd+['--policy',str(policy),'--output',str(dest)]
    if baseline:cmd+=['--baseline',str(baseline)]
    with dest.with_suffix('.log').open('w') as log:subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
baseline=out/'baseline.json'
evaluate(root/'TWIST2/assets/ckpts/twist2_1017_20k.onnx',baseline)
train_out=out/'training'
cmd=[sys.executable,str(root/'adapter/author_finetune.py'),'--root',str(root),'--output',str(train_out),'--headless','--num_envs',str(a.num_envs),'--iterations',str(a.iterations)]
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
            if proc.poll() is not None:break
            time.sleep(20)
        if proc.returncode:raise RuntimeError(f'Trainer exited {proc.returncode}; inspect training.log')
    finally:
        if proc.poll() is None:
            (train_out/'STOP').touch()
            try:proc.wait(timeout=60)
            except subprocess.TimeoutExpired:proc.terminate()
status={'phase':'pipeline_complete','results':results,'training_output':str(train_out),'deployment':False,'validation':'12 original held-out Quest clips, seeds 0/1. Previously inspected validation, not an independent test.','selection':'Report baseline acceptance only; no automatic deployment.'}
(out/'result.json').write_text(json.dumps(status,indent=2));print(json.dumps(status),flush=True)
