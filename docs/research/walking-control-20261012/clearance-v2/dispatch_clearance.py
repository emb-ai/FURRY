"""Explicit bounded experiment dispatch. Main matrix requires a separate call."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def dump(path,obj):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2));tmp.replace(path)


def idle(gpus):
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],universal_newlines=True)
    data={int(r.split(',')[0]):list(map(int,r.split(',')[1:])) for r in raw.splitlines()}
    if any(data[g][0]>=100 or data[g][1]!=0 for g in gpus):raise RuntimeError('Requested GPU busy: '+str(data))
    return data


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['smoke','baseline','launch','status','run'])
    p.add_argument('--root',required=True,type=Path);p.add_argument('--spec',type=Path);p.add_argument('--smoke-gpu',type=int,default=0)
    a=p.parse_args();root=a.root.resolve();code=Path(__file__).resolve().parent
    if a.action=='run':
        row=json.loads(a.spec.read_text());proc=subprocess.Popen(row['argv'],cwd=root)
        dump(a.spec.with_suffix('.running.json'),dict(child_pid=proc.pid,wrapper_pid=os.getpid()))
        rc=proc.wait();dump(a.spec.with_suffix('.exit.json'),dict(returncode=rc,finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
        return rc
    manifest=json.loads((code/'transfer-manifest.json').read_text())
    for name,expected in manifest['files'].items():
        if hashlib.sha256((code/name).read_bytes()).hexdigest()!=expected:raise RuntimeError('Code changed: '+name)
    expected_init='5bbfbddf34b19d8be8c7b37ecfbe61cdef83f03c759f22305fb64d0ee0e06a7f'
    expected_data='f6cde2c4f205831ac38c832ab84c1b83d5b3fb5ed6e8ce427cdf7fdeac1803b4'
    for path,expected in [(root/'comparison-baseline.pt',expected_init),(root/'dataset/manifest.json',expected_data)]:
        if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:raise RuntimeError('Shared input changed: '+str(path))
    def spawn(label,gpu,argv,seed):
        spec=code/(label+'-process.json')
        if spec.exists():raise RuntimeError('Already dispatched: '+label)
        row=dict(label=label,gpu=gpu,argv=argv,seed=seed,started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),log=str(code/(label+'.log')),deployment=False)
        dump(spec,row)
        env=dict(os.environ,PYTHONHASHSEED=str(seed))
        if gpu is not None:env['CUDA_VISIBLE_DEVICES']=str(gpu)
        with Path(row['log']).open('xb') as log:
            proc=subprocess.Popen([sys.executable,str(__file__),'run','--root',str(root),'--spec',str(spec)],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        row['wrapper_pid']=proc.pid;dump(spec,row);return row
    init=['--init-checkpoint',str(root/'comparison-baseline.pt'),'--init-manifest',str(root/'locomotion-ancestor-manifest.json')]
    common=['--root',str(root),'--fast-critic','--nonlooping','--window-reward','--sole-urdf',str(root/'assets/g1_main_spheres.urdf')]+init
    smoke=code/'smoke-clearance-v2'
    if a.action=='smoke':
        dump(code/'smoke-gpu-before.json',idle([a.smoke_gpu]))
        if smoke.exists():raise RuntimeError('Existing smoke output')
        argv=[str(root/'env/bin/python'),str(code/'clearance_finetune.py'),'--output',str(smoke),'--headless','--num_envs','4096','--sim_device','cuda:0','--rl_device','cuda:0','--device','cuda:0','--num_threads','1','--seed','11','--smoke','--velocity-clearance']+common
        print(json.dumps(spawn('smoke-clearance-v2',a.smoke_gpu,argv,11)));return 0
    if a.action=='baseline':
        if (code/'baseline-v2.json').exists():raise RuntimeError('Existing baseline output')
        argv=[str(root/'eval-env/bin/python'),str(code/'locomotion_eval.py'),'--assets',str(root/'runtime-assets'),'--dataset',str(root/'clamp-mirrored'),'--policy',str(root/'comparison-baseline.pt'),'--source',str(root/'TWIST2/rsl_rl/rsl_rl/modules/actor_critic_future.py'),'--output',str(code/'baseline-v2.json'),'--control-hz','50','--window-metrics']
        print(json.dumps(spawn('baseline-clearance-v2',None,argv,0)));return 0
    if a.action=='launch':
        if (code/'submitted.json').exists():raise RuntimeError('Matrix already submitted')
        last=json.loads((smoke/'metrics.jsonl').read_text().splitlines()[-1]);cfg=json.loads((smoke/'config.json').read_text())
        checks=json.loads((smoke/'gpu-contract-checks.json').read_text())
        if last.get('phase')!='complete' or last.get('actor_updates')!=1 or not checks['success']:raise RuntimeError('GPU smoke incomplete')
        if cfg['velocity_clearance']['critic_inputs']!=1743 or not cfg['ppo_config_matches_upstream']:raise RuntimeError('Unexpected smoke contract')
        from evaluation_contract import contract,baseline_matches
        expected=contract(root/'runtime-assets',root/'clamp-mirrored',root/'TWIST2/rsl_rl/rsl_rl/modules/actor_critic_future.py',50,[0,1],'validation')
        if not baseline_matches(code/'baseline-v2.json',root/'comparison-baseline.pt',expected):raise RuntimeError('Fresh matching baseline required')
        dump(code/'matrix-gpu-before.json',idle([0,1,2,3]));rows=[]
        for condition in ['control','clearance']:
            for seed in [11,22]:
                label='walking-clearance-v2-%s-s%d'%(condition,seed)
                if (root/'runs'/label).exists():raise RuntimeError('Existing run '+label)
                argv=[str(root/'env/bin/python'),str(code/'clearance_pipeline.py'),'--run-id',label,'--num-envs','4096','--iterations','1000','--save-every','250','--seed',str(seed),'--control-hz','50','--eval-assets',str(root/'runtime-assets'),'--eval-dataset',str(root/'clamp-mirrored'),'--comparison-baseline',str(root/'comparison-baseline.pt'),'--baseline-report',str(code/'baseline-v2.json')]+common
                if condition=='clearance':argv+=['--velocity-clearance']
                rows.append(spawn(label,len(rows),argv,seed))
                dump(code/'submitted.json',dict(runs=rows,actor_updates=1000,save_every=250,initial_checkpoint_sha256=expected_init,dataset_manifest_sha256=expected_data,source_ppo_unchanged=True,deployment=False))
        print(json.dumps(rows));return 0
    result=[]
    for spec in sorted(code.glob('*-process.json')):
        row=json.loads(spec.read_text())
        for suffix,key in [('.running.json','process'),('.exit.json','exit')]:
            path=spec.with_suffix(suffix)
            if path.exists():row[key]=json.loads(path.read_text())
        try:os.kill(row['wrapper_pid'],0);row['wrapper_alive']=True
        except (ProcessLookupError,KeyError):row['wrapper_alive']=False
        output=smoke if row['label']=='smoke-clearance-v2' else root/'runs'/row['label']/'training'
        if (output/'metrics.jsonl').exists():row['last_metrics']=json.loads((output/'metrics.jsonl').read_text().splitlines()[-1])
        row['last_log_lines']=Path(row['log']).read_text(errors='replace').splitlines()[-4:]
        result.append(row)
    dump(code/'status.json',dict(checked_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),runs=result));print(json.dumps(result,indent=2));return 0


if __name__=='__main__':raise SystemExit(main())
