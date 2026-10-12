"""Finite five-job queue: one verified smoke, then four fixed experiment runs.

Linux/stdlib only. No SSH, Slurm changes, process termination, or modification
of frozen training code. STOP prevents further scheduling; active jobs finish.
Run one queue with an OS lock. Restart adopts its own exact process identities;
an ambiguous/lost launch fails closed and is never automatically retried.
"""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

SOURCE_SHA='a70f6e9031ebcaaecb29e01e152455dab8e3aa1b01f265530315ba52ced71999'
INIT_SHA='5bbfbddf34b19d8be8c7b37ecfbe61cdef83f03c759f22305fb64d0ee0e06a7f'
DATA_SHA='f6cde2c4f205831ac38c832ab84c1b83d5b3fb5ed6e8ce427cdf7fdeac1803b4'
SOLE_SHA='5ce11a1df5da7a40636c494e5a0d257d0b259018dd31cce599b5098df7ef1659'
SMOKE='smoke-clearance-v2'
RUNS=[('walking-clearance-v2-%s-s%d'%(condition,seed),condition,seed)
      for condition in ('control','clearance') for seed in (11,22)]
MAX_ACTIVE=4
class QueueBusy(RuntimeError):pass


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def utc():return time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
def read(path):return json.loads(Path(path).read_text())
def dump(path,value):
    path=Path(path);tmp=path.with_name(path.name+'.tmp-'+str(os.getpid()))
    tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)
def event(state,kind,**fields):
    raw=(json.dumps(dict(utc=utc(),event=kind,**fields))+'\n').encode()
    fd=os.open(str(state/'events.jsonl'),os.O_CREAT|os.O_WRONLY|os.O_APPEND,0o600)
    try:os.write(fd,raw)
    finally:os.close(fd)


def process_info(pid,proc=Path('/proc')):
    """Kernel birth time and complete argv prevent accidental PID reuse."""
    try:
        base=proc/str(pid);stat=(base/'stat').read_text();tail=stat[stat.rfind(')')+2:].split()
        if tail[0]=='Z':return None
        argv=[x.decode(errors='replace') for x in (base/'cmdline').read_bytes().split(b'\0') if x]
        if not argv:return None
        return dict(pid=int(pid),start_ticks=int(tail[19]),argv=argv)
    except (FileNotFoundError,ProcessLookupError,PermissionError):return None


def exact_identity(expected,observed):
    return bool(expected and observed and all(expected.get(k)==observed.get(k) for k in ('pid','start_ticks','argv')))


def process_snapshot():
    if not Path('/proc').is_dir():raise RuntimeError('Queue requires Linux /proc process identity')
    return [p for directory in Path('/proc').iterdir() if directory.name.isdigit()
            for p in [process_info(int(directory.name))] if p]


def flag_value(argv,flag):
    try:return argv[argv.index(flag)+1]
    except (ValueError,IndexError):return None


def own_processes(spec,processes):
    """Include a surviving trainer if its pipeline/wrapper unexpectedly exits."""
    result=[]
    trainer=str(Path(spec['code'])/'clearance_finetune.py')
    for p in processes:
        args=p['argv']
        direct=args==spec['argv'] or args==spec['worker_argv']
        child=trainer in args and flag_value(args,'--output')==spec['training_output'] and flag_value(args,'--root')==spec['root']
        if direct or child:result.append(p)
    return result


def job_status(spec,running,finished,processes,now=None):
    alive=own_processes(spec,processes)
    if running:
        known=[running.get(k) for k in ('wrapper','child') if running.get(k)]
        for observed in alive:
            for original in known:
                if original['pid']==observed['pid'] and not exact_identity(original,observed):
                    raise RuntimeError('PID reused for a different process birth/argv')
    # Never free the device merely because the wrapper or pipeline wrote exit.
    if alive:return dict(status='active',gpu=spec['gpu'],processes=alive)
    if finished:
        if finished.get('spec_sha256')!=spec['identity_sha256']:
            raise RuntimeError('Exit receipt belongs to another job spec')
        if (finished.get('returncode')==75 and finished.get('deferred_without_child') is True
                and finished.get('child_started') is False and finished.get('reason')=='gpu_became_busy' and not running):
            return dict(status='deferred',gpu=spec['gpu'],exit=finished)
        return dict(status='complete' if finished.get('returncode')==0 else 'failed',gpu=spec['gpu'],exit=finished)
    # Popen/receipt is not a filesystem transaction. A crash in that tiny window
    # is ambiguous: adopt exact matching argv if present, otherwise never retry.
    if (now or time.time())-spec['scheduled_epoch']<30:
        return dict(status='starting',gpu=spec['gpu'],processes=[])
    return dict(status='lost',gpu=spec['gpu'],reason='No matching process or terminal receipt; manual inspection required')


def parse_gpu_snapshot(gpu_text,apps_text):
    rows=[]
    for line in gpu_text.splitlines():
        if not line.strip():continue
        parts=[x.strip() for x in line.split(',')]
        if len(parts)!=4:raise RuntimeError('Unexpected nvidia-smi GPU row')
        index,uuid,memory,util=parts
        rows.append(dict(index=int(index),uuid=uuid,memory_mib=int(memory),util_percent=int(util),compute_pids=[]))
    by_uuid={g['uuid']:g for g in rows}
    for line in apps_text.splitlines():
        if not line.strip():continue
        parts=[x.strip() for x in line.split(',')]
        if len(parts)!=2 or parts[0] not in by_uuid:raise RuntimeError('Unknown compute-app GPU; fail closed')
        by_uuid[parts[0]]['compute_pids'].append(int(parts[1]))
    if not rows:raise RuntimeError('No GPUs in snapshot')
    return rows


def gpu_snapshot():
    def query(fields,kind):
        return subprocess.check_output(['nvidia-smi','--query-'+kind+'='+fields,'--format=csv,noheader,nounits'],
                                       universal_newlines=True,timeout=20)
    return parse_gpu_snapshot(query('index,uuid,memory.used,utilization.gpu','gpu'),query('gpu_uuid,pid','compute-apps'))


def select_gpu(snapshot,allowed,reserved,active):
    if active>=MAX_ACTIVE:return None
    for row in sorted(snapshot,key=lambda r:r['index']):
        if row['index'] in allowed and row['index'] not in reserved and row['memory_mib']<100 and row['util_percent']==0 and not row['compute_pids']:
            return row['index']
    return None


def plans(root,code):
    init=['--init-checkpoint',str(root/'comparison-baseline.pt'),'--init-manifest',str(root/'locomotion-ancestor-manifest.json')]
    common=['--root',str(root),'--fast-critic','--nonlooping','--window-reward','--sole-urdf',str(root/'assets/g1_main_spheres.urdf')]+init
    smoke=code/SMOKE
    rows=[dict(label=SMOKE,seed=11,training_output=str(smoke),
        argv=[str(root/'env/bin/python'),str(code/'clearance_finetune.py'),'--output',str(smoke),'--headless','--num_envs','4096','--sim_device','cuda:0','--rl_device','cuda:0','--device','cuda:0','--num_threads','1','--seed','11','--smoke','--velocity-clearance']+common)]
    for label,condition,seed in RUNS:
        argv=[str(root/'env/bin/python'),str(code/'clearance_pipeline.py'),'--run-id',label,'--num-envs','4096','--iterations','1000','--save-every','250','--seed',str(seed),'--control-hz','50','--eval-assets',str(root/'runtime-assets'),'--eval-dataset',str(root/'clamp-mirrored'),'--comparison-baseline',str(root/'comparison-baseline.pt'),'--baseline-report',str(code/'baseline-v2.json')]+common
        if condition=='clearance':argv+=['--velocity-clearance']
        rows.append(dict(label=label,seed=seed,training_output=str(root/'runs'/label/'training'),argv=argv))
    return rows


def verify_inputs(root,code,verify_clips=False):
    root,code=root.resolve(),code.resolve()
    if digest(code/'transfer-manifest.json')!=SOURCE_SHA:raise RuntimeError('Frozen source manifest changed')
    manifest=read(code/'transfer-manifest.json')
    for name,sha in manifest['files'].items():
        path=(code/name).resolve()
        if code not in path.parents or digest(path)!=sha:raise RuntimeError('Frozen source changed: '+name)
    expected={'comparison-baseline.pt':INIT_SHA,'dataset/manifest.json':DATA_SHA,'assets/g1_main_spheres.urdf':SOLE_SHA}
    result={name:digest(root/name) for name in expected}
    if result!=expected:raise RuntimeError('Pinned checkpoint/dataset/sole input changed')
    result['locomotion-ancestor-manifest.json']=digest(root/'locomotion-ancestor-manifest.json')
    result['actor_source']=digest(root/'TWIST2/rsl_rl/rsl_rl/modules/actor_critic_future.py')
    result['source_manifest']=SOURCE_SHA
    if verify_clips:
        clips=read(root/'dataset/manifest.json')['clips']
        for clip in clips:
            relative=Path(clip['file'])
            if relative.is_absolute() or '..' in relative.parts:raise RuntimeError('Invalid dataset clip path')
            if digest(root/'dataset'/relative)!=clip['sha256']:raise RuntimeError('Dataset clip changed: '+clip['id'])
    return result


def verify_smoke(folder,exit_receipt):
    if not exit_receipt or exit_receipt.get('returncode')!=0:raise RuntimeError('Smoke has no successful exit receipt')
    rows=[json.loads(line) for line in (folder/'metrics.jsonl').read_text().splitlines() if line.strip()]
    cfg=read(folder/'config.json');check=read(folder/'gpu-contract-checks.json')
    if not rows or rows[-1].get('phase')!='complete' or rows[-1].get('actor_updates')!=1:raise RuntimeError('Smoke did not finish exactly one actor update')
    if not check.get('success') or check.get('num_envs')!=4096 or check.get('actor_inputs')!=1432:raise RuntimeError('Smoke GPU geometry contract failed')
    if not cfg.get('ppo_config_matches_upstream') or cfg['velocity_clearance'].get('critic_inputs')!=1743 or cfg['velocity_clearance'].get('critic_features_added')!=0:raise RuntimeError('Smoke PPO/critic contract failed')
    if not cfg['velocity_clearance'].get('enabled') or cfg.get('initialization',{}).get('sha256')!=INIT_SHA or cfg.get('dataset_manifest_sha256')!=DATA_SHA:raise RuntimeError('Smoke used different treatment/input')
    if cfg.get('sole_urdf_sha256')!=SOLE_SHA or not (folder/'final.pt').is_file():raise RuntimeError('Smoke output/sole mismatch')
    return dict(exit=exit_receipt,last=rows[-1],gpu=check,config_sha256=digest(folder/'config.json'))


def baseline_ready(root,code):
    exit_file=code/'baseline-clearance-v2-process.exit.json'
    if not exit_file.exists():return False
    if read(exit_file).get('returncode')!=0:raise RuntimeError('Fresh CPU baseline failed')
    module_spec=importlib.util.spec_from_file_location('frozen_evaluation_contract',code/'evaluation_contract.py')
    module=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(module)
    expected=module.contract(root/'runtime-assets',root/'clamp-mirrored',root/'TWIST2/rsl_rl/rsl_rl/modules/actor_critic_future.py',50,[0,1],'validation')
    if not module.baseline_matches(code/'baseline-v2.json',root/'comparison-baseline.pt',expected):raise RuntimeError('Fresh CPU baseline fingerprint differs')
    report=read(code/'baseline-v2.json')
    if report.get('total_trials')!=24:raise RuntimeError('Fresh baseline incomplete')
    return True


def verify_training_complete(root,label):
    folder=root/'runs'/label;training=folder/'training'
    rows=[json.loads(line) for line in (training/'metrics.jsonl').read_text().splitlines() if line.strip()]
    if not rows or rows[-1].get('phase')!='complete' or rows[-1].get('actor_updates')!=1000:
        raise RuntimeError('Process exited without training1000 completion: '+label+' '+json.dumps(rows[-1] if rows else {}))
    if not (training/'final.pt').is_file():raise RuntimeError('Missing final trained checkpoint: '+label)
    result=read(folder/'result.json')
    expected={'checkpoint_%06d.pt'%n for n in (250,500,750,1000)}
    evaluated=[r['checkpoint'] for r in result.get('results',[])]
    if result.get('phase')!='pipeline_complete' or set(evaluated)!=expected or len(evaluated)!=4:
        raise RuntimeError('Missing four checkpoint evaluations: '+label)
    for name in expected:
        stem=Path(name).stem
        for suffix in ('.json','.npz','.comparison.json','.clearance-verdict.json','.fingerprint.json'):
            if not (folder/(stem+suffix)).is_file():raise RuntimeError('Missing evaluation artifact: '+label+'/'+stem+suffix)
        if not (training/name).is_file() or read(folder/(stem+'.json')).get('total_trials')!=24:
            raise RuntimeError('Incomplete evaluation/checkpoint: '+label+'/'+name)
    return dict(training_complete=True,actor_updates=1000,evaluated_checkpoints=sorted(expected),
                final_checkpoint_sha256=digest(training/'final.pt'),last_metrics=rows[-1])


def seal_spec(spec):
    return hashlib.sha256(json.dumps({k:v for k,v in spec.items() if k!='identity_sha256'},sort_keys=True).encode()).hexdigest()


def worker(args):
    state=args.state.resolve();spec=read(state/(args.label+'.spec.json'))
    if spec['identity_sha256']!=seal_spec(spec):raise RuntimeError('Queue job spec changed')
    if spec['root']!=str(args.root.resolve()) or spec['code']!=str(args.code.resolve()):raise RuntimeError('Worker path/spec mismatch')
    matching=[p for p in plans(args.root.resolve(),args.code.resolve()) if p['label']==args.label]
    if len(matching)!=1 or matching[0]['argv']!=spec['argv']:raise RuntimeError('Job argv is not the fixed experiment')
    # The queue has just verified every PKL before reservation. Repeat sealed
    # source/manifests/weights here, not a second long full-dataset scan.
    if verify_inputs(args.root.resolve(),args.code.resolve())!=read(state/'queue-config.json')['inputs']:
        raise RuntimeError('Worker inputs differ from sealed queue inputs')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=str(spec['gpu']):raise RuntimeError('Worker GPU environment mismatch')
    fresh=gpu_snapshot()
    if select_gpu(fresh,[spec['gpu']],set(),0)!=spec['gpu']:
        receipt=dict(returncode=75,deferred_without_child=True,child_started=False,reason='gpu_became_busy',
                     spec_sha256=spec['identity_sha256'],finished_utc=utc(),wrapper=process_info(os.getpid()),gpu_snapshot=fresh)
        dump(state/(args.label+'.exit.json'),receipt);event(state,'job_deferred_without_child',label=args.label,**receipt)
        return 75
    env=dict(os.environ,PYTHONHASHSEED=str(spec['seed']),CUDA_VISIBLE_DEVICES=str(spec['gpu']))
    # Lock descriptors are noninheritable; each job survives a queue STOP/exit.
    child=subprocess.Popen(spec['argv'],cwd=args.root,env=env)
    running=dict(wrapper=process_info(os.getpid()),child=process_info(child.pid),started_utc=utc(),spec_sha256=spec['identity_sha256'])
    dump(state/(args.label+'.running.json'),running);event(state,'child_started',label=args.label,**running)
    rc=child.wait()
    finished=dict(returncode=rc,finished_utc=utc(),spec_sha256=spec['identity_sha256'],wrapper=running['wrapper'],child=running['child'])
    dump(state/(args.label+'.exit.json'),finished);event(state,'child_exited',label=args.label,**finished)
    return rc


def launch(plan,gpu,snapshot,args):
    root,code,state=args.root,args.code,args.state
    spec_path=state/(plan['label']+'.spec.json')
    if spec_path.exists():raise RuntimeError('Refusing duplicate launch: '+plan['label'])
    if (state/'STOP').exists():return False
    output=Path(plan['training_output']) if plan['label']==SMOKE else root/'runs'/plan['label']
    if output.exists():raise RuntimeError('Unexpected prior job output: '+str(output))
    argv=[sys.executable,str(Path(__file__).resolve()),'worker','--root',str(root),'--code',str(code),'--state',str(state),'--label',plan['label']]
    spec=dict(plan,root=str(root),code=str(code),gpu=gpu,scheduled_epoch=time.time(),scheduled_utc=utc(),worker_argv=argv,gpu_snapshot=snapshot,source_manifest_sha256=SOURCE_SHA)
    spec['identity_sha256']=seal_spec(spec)
    # O_EXCL prevents duplicate reservations even if another caller bypasses lock.
    with spec_path.open('x') as f:f.write(json.dumps(spec,indent=2)+'\n')
    event(state,'job_reserved',label=plan['label'],gpu=gpu,spec=spec)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONHASHSEED=str(plan['seed']))
    with (state/(plan['label']+'.log')).open('xb') as log:
        proc=subprocess.Popen(argv,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    dump(state/(plan['label']+'.launch.json'),dict(wrapper=process_info(proc.pid),wrapper_pid=proc.pid,utc=utc(),spec_sha256=spec['identity_sha256']))
    return True


def snapshot_jobs(all_plans,state,processes):
    result={}
    for plan in all_plans:
        label=plan['label'];path=state/(label+'.spec.json')
        if not path.exists():result[label]=dict(status='pending');continue
        spec=read(path)
        if spec['identity_sha256']!=seal_spec(spec) or spec['argv']!=plan['argv']:raise RuntimeError('Stored job plan changed')
        running=read(state/(label+'.running.json')) if (state/(label+'.running.json')).exists() else None
        finished=read(state/(label+'.exit.json')) if (state/(label+'.exit.json')).exists() else None
        archive=state/'attempts'/label/spec['identity_sha256']
        # Recover a crash during documented deferred-attempt cleanup. This path
        # never applies to an unexplained missing receipt or a launched child.
        if finished is None and (archive/'retry-proof.json').exists():
            proof=read(archive/'retry-proof.json')
            if proof.get('spec_sha256')!=spec['identity_sha256']:raise RuntimeError('Deferred cleanup proof mismatch')
            finished=read(archive/(label+'.exit.json'))
        result[label]=job_status(spec,running,finished,processes)
    return result


def requeue_deferred(plan,state,processes):
    label=plan['label'];spec=read(state/(label+'.spec.json'))
    if own_processes(spec,processes):raise RuntimeError('Deferred wrapper/child still alive')
    root=Path(spec['root']);output=Path(plan['training_output']) if label==SMOKE else root/'runs'/label
    if output.exists() or (state/(label+'.running.json')).exists():raise RuntimeError('Cannot retry deferred attempt with child/output evidence')
    archive=state/'attempts'/label/spec['identity_sha256'];archive.mkdir(parents=True,exist_ok=True)
    receipt_path=state/(label+'.exit.json')
    receipt=read(receipt_path) if receipt_path.exists() else read(archive/(label+'.exit.json'))
    if job_status(spec,None,receipt,processes)['status']!='deferred':raise RuntimeError('Missing explicit no-child deferral proof')
    for suffix in ('.spec.json','.exit.json','.launch.json','.log'):
        path=state/(label+suffix)
        if path.exists():shutil.copy2(path,archive/path.name)
    dump(archive/'retry-proof.json',dict(spec_sha256=spec['identity_sha256'],child_started=False,reason='gpu_became_busy'))
    for suffix in ('.exit.json','.launch.json','.log'):(state/(label+suffix)).unlink(missing_ok=True)
    (state/(label+'.spec.json')).unlink() # Commit cleanup last; next poll may reserve a new attempt.
    event(state,'deferred_attempt_archived',label=label,archive=str(archive),no_child_started=True)


def run(args):
    root,code,state=args.root,args.code,args.state;state.mkdir(parents=True,exist_ok=True)
    with (state/'queue.lock').open('a+') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise QueueBusy('Another queue process holds this state lock')
        pinned=verify_inputs(root,code,verify_clips=True);all_plans=plans(root,code)
        identity=dict(root=str(root),code=str(code),allowed_gpus=args.gpus,inputs=pinned,queue_source_sha256=digest(__file__))
        config=state/'queue-config.json'
        if config.exists() and read(config)!=identity:raise RuntimeError('Queue inputs/code/path changed across restart')
        if not config.exists():dump(config,identity)
        for plan in all_plans:
            if (code/(plan['label']+'-process.json')).exists():raise RuntimeError('Existing dispatch job conflicts with queue: '+plan['label'])
        event(state,'queue_started',pid=os.getpid(),identity=process_info(os.getpid()),poll_seconds=args.poll)
        while True:
            if (state/'STOP').exists():
                event(state,'queue_stopped_by_file',active_jobs_untouched=True)
                jobs=snapshot_jobs(all_plans,state,process_snapshot())
                dump(state/'status.json',dict(utc=utc(),phase='stopped',jobs=jobs,active_jobs_untouched=True));return 0
            jobs=snapshot_jobs(all_plans,state,process_snapshot())
            for plan in all_plans:
                if jobs[plan['label']]['status']=='deferred':
                    requeue_deferred(plan,state,process_snapshot())
                    jobs[plan['label']]=dict(status='pending',previous_attempt='GPU became busy before any child started')
            bad={k:v for k,v in jobs.items() if v['status'] in ('failed','lost')}
            if bad:raise RuntimeError('No new starts after failed/lost job: '+json.dumps(bad))
            for plan in all_plans[1:]:
                if jobs[plan['label']]['status']=='complete':
                    jobs[plan['label']]['outcome']=verify_training_complete(root,plan['label'])
            active=[v for v in jobs.values() if v['status'] in ('starting','active')]
            reserved={v['gpu'] for v in active}
            if len(active)>MAX_ACTIVE or len(reserved)!=len(active):raise RuntimeError('Invalid existing concurrency/GPU assignments')
            smoke=jobs[SMOKE];phase='waiting_smoke'
            pending=[all_plans[0]] if smoke['status']=='pending' else []
            if smoke['status']=='complete':
                verify_smoke(code/SMOKE,read(state/(SMOKE+'.exit.json')))
                phase='waiting_baseline'
                if baseline_ready(root,code):
                    phase='matrix';pending=[p for p in all_plans[1:] if jobs[p['label']]['status']=='pending']
                    if all(jobs[p['label']]['status']=='complete' for p in all_plans[1:]):
                        dump(state/'status.json',dict(utc=utc(),phase='complete',jobs=jobs,deployment=False));event(state,'queue_complete');return 0
            gpus=gpu_snapshot()
            dump(state/'status.json',dict(utc=utc(),phase=phase,jobs=jobs,gpus=gpus,deployment=False))
            # At most one new reservation per15s tick. Re-query immediately so
            # stale observations or a just-started process never select a GPU.
            if pending and len(active)<MAX_ACTIVE and not (state/'STOP').exists():
                fresh=gpu_snapshot();gpu=select_gpu(fresh,args.gpus,reserved,len(active))
                if gpu is not None:
                    if verify_inputs(root,code,verify_clips=True)!=pinned:raise RuntimeError('Inputs changed while queued')
                    # Full dataset hashing may take time; only use a fresh
                    # snapshot of this chosen device after it finishes.
                    fresh=gpu_snapshot()
                    if select_gpu(fresh,[gpu],reserved,len(active))==gpu:launch(pending[0],gpu,fresh,args)
            if args.once:return 0
            time.sleep(args.poll)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=('run','worker'))
    for flag in ('root','code','state'):p.add_argument('--'+flag,type=Path,required=True)
    p.add_argument('--label');p.add_argument('--gpus',nargs='+',type=int,default=[0,1,2,3])
    p.add_argument('--poll',type=int,default=15);p.add_argument('--once',action='store_true')
    args=p.parse_args()
    for field in ('root','code','state'):setattr(args,field,getattr(args,field).resolve())
    if not 10<=args.poll<=20 or len(set(args.gpus))!=len(args.gpus) or any(g<0 for g in args.gpus):p.error('poll10..20 and unique nonnegative GPU ids required')
    if args.action=='worker':
        if not args.label:p.error('Worker label required')
        return worker(args)
    try:return run(args)
    except QueueBusy as error:
        print(str(error),file=sys.stderr);return 3
    except Exception as error:
        args.state.mkdir(parents=True,exist_ok=True)
        event(args.state,'queue_failed',error=str(error),active_jobs_untouched=True)
        status=dict(utc=utc(),phase='failed_closed',error=str(error),active_jobs_untouched=True)
        try:status['jobs']=snapshot_jobs(plans(args.root,args.code),args.state,process_snapshot())
        except Exception:pass
        dump(args.state/'status.json',status);raise


if __name__=='__main__':raise SystemExit(main())
