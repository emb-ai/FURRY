"""One-GPU training command and durable three-slot work-conserving queue."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

BASELINE = '5bbfbddf34b19d8be8c7b37ecfbe61cdef83f03c759f22305fb64d0ee0e06a7f'
SCHEDULE = [100, 200] + list(range(250, 5001, 250))


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path, obj):
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj, indent=2));tmp.replace(path)


def command(root, code, condition, seed, output, pilot=False, updates=20, phase=None):
    config = code/(('pilot-' if pilot else 'frozen-')+condition+'.json')
    args = ['/home/jovyan/.local/bin/uv','run','--no-project','--python',str(root/'env/bin/python'),
            str(code/'locomotion_finetune.py'),'--root',str(root),'--output',str(output),
            '--experiment-config',str(config),'--phase-table',str(phase or code/'phase.npz'),
            '--lower-body-coefficient','.25','--nonlooping','--window-reward','--fast-critic',
            '--sole-urdf',str(root/'assets/g1_main_spheres.urdf'),
            '--init-checkpoint',str(root/'comparison-baseline.pt'),
            '--init-manifest',str(root/'locomotion-ancestor-manifest.json'),
            '--iterations','5000','--headless','--num_envs','4096','--seed',str(seed),
            '--sim_device','cuda:0','--rl_device','cuda:0','--device','cuda:0','--num_threads','1']
    if pilot: args += ['--smoke','--smoke-actor-updates',str(updates)]
    settings = json.loads(config.read_text())
    if settings.get('actor_command_bank'):
        args += ['--actor-command-bank', str(code/settings['actor_command_bank'])]
    return args


def gpu_inventory():
    raw = subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,memory.used,utilization.gpu',
                                   '--format=csv,noheader,nounits'],text=True)
    apps = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid',
                                    '--format=csv,noheader'],text=True)
    occupied = set(apps.splitlines())
    result=[]
    for line in raw.splitlines():
        uuid,name,memory,util=[x.strip() for x in line.split(',')]
        if 'H100' in name and uuid not in occupied and int(memory)<100 and int(util)==0:
            result.append(uuid)
    return result


def verify_frozen(root, code, condition):
    receipt = json.loads((code/'readiness.json').read_text())
    if not receipt.get('input_alignment_approved'):
        raise ValueError('Queue paused by the revised scope: Virtual Legs alignment comes first')
    config = code/('frozen-'+condition+'.json')
    if not receipt['conditions'][condition]['ready']:
        return False
    if sha(config) != receipt['conditions'][condition]['config_sha256']:
        raise ValueError('Frozen configuration changed')
    for path,digest in receipt['files'].items():
        if sha(code/path) != digest: raise ValueError('Code/teacher changed after smoke: '+path)
    if sha(root/'comparison-baseline.pt') != BASELINE:
        raise ValueError('Wrong common checkpoint')
    return True


def run_queue(root, code):
    lock=(code/'queue.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state_path=code/'queue-state.json'
    if state_path.exists():
        raise RuntimeError('Queue state already exists; inspect PIDs before explicit recovery')
    # Prioritize the seed11 comparisons; if C is unready, keep A/B seeds moving.
    jobs=[dict(condition=c,seed=s,phase='pending',actor_updates_target=5000)
          for s in (11,22) for c in 'ABC']
    state=dict(schema=1,max_physical_gpus=3,assigned_uuids=[],jobs=jobs,checkpoint_schedule=SCHEDULE,
               comparison_baseline_sha256=BASELINE,started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
               deployment=False)
    active={}
    while True:
        for index,(proc,gpu,log) in list(active.items()):
            if proc.poll() is None: continue
            row=jobs[index];log.close();row['exit_code']=proc.returncode
            metrics=Path(row['output'])/'metrics.jsonl'
            last=json.loads(metrics.read_text().splitlines()[-1]) if metrics.exists() else {}
            row['phase']='complete' if proc.returncode==0 and last.get('phase')=='complete' and last.get('actor_updates')==5000 else 'technical_failure'
            row['last_metric']=last
            del active[index]
        idle=gpu_inventory()
        for index,row in enumerate(jobs):
            if len(active)>=3: break
            if row['phase']!='pending' or not verify_frozen(root,code,row['condition']): continue
            reserved={v[1] for v in active.values()}
            candidates=[g for g in idle if g not in reserved and (g in state['assigned_uuids'] or len(state['assigned_uuids'])<3)]
            if not candidates: break
            gpu=candidates[0]
            if gpu not in state['assigned_uuids']:state['assigned_uuids'].append(gpu)
            name='contact-v1-%s-s%d'%(row['condition'],row['seed'])
            out=root/'runs'/name
            if out.exists(): raise RuntimeError('Output already exists '+str(out))
            argv=command(root,code,row['condition'],row['seed'],out)
            log=(code/(name+'.log')).open('xb')
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,PYTHONHASHSEED=str(row['seed']))
            # Inherit singleton lock: loss of scheduler cannot make duplicate jobs.
            proc=subprocess.Popen(argv,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,
                                  start_new_session=True,pass_fds=(lock.fileno(),))
            row.update(phase='starting',pid=proc.pid,gpu_uuid=gpu,argv=argv,output=str(out))
            active[index]=(proc,gpu,log)
            atomic(state_path,state)
        for index,(proc,gpu,log) in active.items():
            row=jobs[index];metrics=Path(row['output'])/'metrics.jsonl'
            if metrics.exists():
                try:
                    last=json.loads(metrics.read_text().splitlines()[-1]);row['last_metric']=last
                    row['phase']=last.get('phase','starting')
                except (ValueError,IndexError): pass  # concurrent JSONL append
        state['updated_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
        atomic(state_path,state)
        if not active and all(r['phase'] not in ('pending','starting','critic','actor') for r in jobs):break
        time.sleep(20)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--pilot',choices=list('ABC'));p.add_argument('--output',type=Path)
    p.add_argument('--updates',type=int,default=20);p.add_argument('--seed',type=int,default=11)
    a=p.parse_args();code=Path(__file__).resolve().parent;root=a.root.resolve()
    if a.pilot:
        if not a.output: p.error('--output required for pilot')
        raise SystemExit(subprocess.call(command(root,code,a.pilot,a.seed,a.output.resolve(),True,a.updates),cwd=root))
    run_queue(root,code)
