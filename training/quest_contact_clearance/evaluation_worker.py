"""CPU-only checkpoint evaluation, preserving the old gate and paired prefixes."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np
from cycle_runner import atomic, SCHEDULE, BASELINE, sha


def load(path):
    report=json.loads(path.read_text())
    with np.load(path.with_suffix('.npz'),allow_pickle=False) as z:
        report['_traces']={k:z[k].copy() for k in z.files}
    return report


def comparisons(code):
    from evaluate import compare
    from locomotion_verdict import summarize
    folder=code/'evaluations';baseline_path=folder/'baseline.json'
    if not baseline_path.exists():return []
    rows=[]
    for seed in (11,22):
        for update in SCHEDULE:
            for condition in 'ABC':
                path=folder/('%s-s%d-%06d.json'%(condition,seed,update))
                if not path.exists():continue
                result=dict(condition=condition,seed=seed,actor_updates=update,comparisons={})
                pairs=[('original',baseline_path)]
                if condition in 'BC':pairs.append(('A',folder/('A-s%d-%06d.json'%(seed,update))))
                if condition=='C':pairs.append(('B',folder/('B-s%d-%06d.json'%(seed,update))))
                for label,base in pairs:
                    if not base.exists():continue
                    dest=path.with_suffix('.vs-'+label+'.json')
                    if dest.exists():verdict=json.loads(dest.read_text())
                    else:
                        raw=compare(load(base),load(path))
                        verdict=dict(raw=raw,gait_gate=summarize(raw))
                        atomic(dest,verdict)
                    result['comparisons'][label]=verdict['gait_gate']
                rows.append(result)
    atomic(folder/'summary.json',dict(rows=rows,diagnostic_validation=True,independent_test=False))
    return rows


def main(root,code):
    lock=(code/'evaluation.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    folder=code/'evaluations';folder.mkdir(exist_ok=True)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    def evaluate(policy,dest,record=False):
        if dest.exists() and dest.with_suffix('.npz').exists():return
        args=['/home/jovyan/.local/bin/uv','run','--no-project','--python',str(root/'eval-env/bin/python'),
              str(code/'locomotion_eval.py'),'--assets',str(root/'runtime-assets'),
              '--dataset',str(root/'clamp-mirrored'),'--policy',str(policy),
              '--source',str(root/'TWIST2/rsl_rl/rsl_rl/modules/actor_critic_future.py'),
              '--output',str(dest),'--control-hz','50','--window-metrics']
        if record:args+=['--record-state',str(folder/(dest.stem+'-states'))]
        with dest.with_suffix('.log').open('a') as log:
            subprocess.run(args,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    if sha(root/'comparison-baseline.pt')!=BASELINE:raise ValueError('Wrong baseline')
    evaluate(root/'comparison-baseline.pt',folder/'baseline.json',True)
    while True:
        state_path=code/'queue-state.json'
        state=json.loads(state_path.read_text()) if state_path.exists() else {'jobs':[]}
        for row in state['jobs']:
            if 'output' not in row:continue
            for update in SCHEDULE:
                checkpoint=Path(row['output'])/('checkpoint_%06d.pt'%update)
                if checkpoint.exists():
                    dest=folder/('%s-s%d-%06d.json'%(row['condition'],row['seed'],update))
                    evaluate(checkpoint,dest,update in (200,5000))
        comparisons(code)
        if state['jobs'] and all(r['phase'] in ('complete','technical_failure') for r in state['jobs']):break
        time.sleep(30)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--compare-only',action='store_true');a=p.parse_args();code=Path(__file__).resolve().parent
    if a.compare_only:comparisons(code)
    else:main(a.root.resolve(),code)
