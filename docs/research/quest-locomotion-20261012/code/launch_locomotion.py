"""Launch eight separate single-GPU experiments, only on currently idle GPUs."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

root=Path('/home/jovyan/vlad/FURRY/twist2-distributed-20261010')
code=Path(__file__).resolve().parent
dest=code/'submitted.json'
if dest.exists():raise RuntimeError('Already submitted; inspect registered processes instead of duplicating')
smoke=code/'smoke-combined/final.pt'
if not smoke.exists():raise RuntimeError('Successful combined GPU smoke checkpoint required')
last=json.loads((code/'smoke-combined/metrics.jsonl').read_text().splitlines()[-1])
if last.get('phase')!='complete' or last.get('actor_updates')!=1:raise RuntimeError('GPU smoke incomplete')
if not (code/'evaluation-smoke.json').exists():raise RuntimeError('Upper-body evaluation smoke required')
data=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
idle=[]
for line in data.splitlines():
    index,memory,util=map(int,line.split(','))
    if memory<100 and util==0:idle.append(index)
if len(idle)<8:raise RuntimeError('Need8 currently idle GPUs; available '+str(idle))
if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
    raise RuntimeError('Compute process appeared; recheck GPU assignment')
uv='/home/jovyan/.local/bin/uv'
conditions=[('control',1.,False),('relax',.25,False),('yaw',1.,True),('relax-yaw',.25,True)]
submitted=[]
for label,factor,yaw in conditions:
    for seed in (11,22):
        gpu=idle[len(submitted)];name=f'locomotion-v1-{label}-s{seed}'
        if (root/'runs'/name).exists():raise RuntimeError('Run already exists '+name)
        cmd=[uv,'run','--no-project','--python',str(root/'env/bin/python'),str(code/'run_locomotion_pipeline.py'),'--root',str(root),'--run-id',name,'--num-envs','4096','--iterations','1000','--save-every','100','--seed',str(seed),'--control-hz','50','--fast-critic','--nonlooping','--window-reward','--sole-urdf',str(root/'assets/g1_main_spheres.urdf'),'--eval-assets',str(root/'runtime-assets'),'--eval-dataset',str(root/'clamp-mirrored'),'--init-checkpoint',str(root/'comparison-baseline.pt'),'--init-manifest',str(root/'locomotion-ancestor-manifest.json'),'--comparison-baseline',str(root/'comparison-baseline.pt'),'--lower-body-coefficient',str(factor)]
        if yaw:cmd.append('--yaw-airtime')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONHASHSEED=str(seed))
        logfile=code/(name+'.log')
        with logfile.open('xb') as log:
            process=subprocess.Popen(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        row={'run_id':name,'condition':label,'training_seed':seed,'gpu':gpu,'pid':process.pid,'lower_body_coefficient':factor,'yaw_airtime':yaw,'actor_updates_target':1000,'critic_warmup':'500..1000 new critic updates, existing readiness check','resume_optimizer':False,'argv':cmd,'log':str(logfile),'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
        submitted.append(row)
        dest.write_text(json.dumps({'runs':submitted,'deployment':False,'shared_initial_checkpoint_sha256':hashlib.sha256((root/'comparison-baseline.pt').read_bytes()).hexdigest(),'initialization':'111260/1000 actor/std/frozen normalization; new critic and Adam for every condition','shared_changes':['nonlooping','sphere soles','window displacement/turn reward'],'source_ppo_unchanged':True},indent=2))
print(json.dumps({'submitted':[{k:r[k] for k in ('run_id','gpu','pid')} for r in submitted]}))
