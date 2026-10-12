"""Launch one bounded run only on a currently idle GPU; never touch other jobs."""
import argparse,json,os,shlex,subprocess,sys,time
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--label',required=True);p.add_argument('--worker',action='store_true')
    p.add_argument('--num-envs',type=int,default=4096)
    p.add_argument('--smoke',action='store_true');p.add_argument('--identity',action='store_true')
    p.add_argument('--evaluate');p.add_argument('--iterations',type=int,default=300)
    p.add_argument('--pipeline',action='store_true')
    p.add_argument('--resume');p.add_argument('--clearance-multiplier',type=float,default=1.)
    p.add_argument('--eval-every',type=int,default=500)
    p.add_argument('--timeout-seconds',type=int,default=10800)
    p.add_argument('--gpu',type=int,help='Require this idle GPU; occupancy checks still apply')
    a=p.parse_args();root=a.root.resolve();src=Path(__file__).resolve().parent
    if Path(a.label).name!=a.label:raise ValueError('Label must be a single path component')
    log=src/(a.label+'.log');receipt=src/(a.label+'.launch.json');end=src/(a.label+'.exit.json')
    if a.worker:
        spec=json.loads(receipt.read_text())
        # Recheck occupancy immediately before child initialization.
        occupied=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],universal_newlines=True)
        if spec['gpu_uuid'] in occupied:
            end.write_text(json.dumps(dict(returncode=75,child_started=False,reason='GPU became busy; no child launched')));return
        with log.open('x') as stream:
            child=subprocess.Popen(['bash','-c',spec['command']],stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
            spec['child_pid']=child.pid;receipt.write_text(json.dumps(spec,indent=2))
            code=child.wait()
        end.write_text(json.dumps(dict(returncode=code,finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())),indent=2));return
    if receipt.exists() or log.exists() or (src/a.label).exists():raise ValueError('Refusing to reuse existing run')
    snapshot=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],universal_newlines=True)
    occupied=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],universal_newlines=True)
    free=[line.split(', ') for line in snapshot.splitlines() if line.strip()]
    free=[r for r in free if int(r[2])<100 and int(r[3])==0 and r[1] not in occupied]
    if a.gpu is not None:free=[r for r in free if int(r[0])==a.gpu]
    if not free:raise RuntimeError('No free GPU; no run started')
    gpu,uuid=free[0][:2]
    argv=[str(root/'env/bin/python'),str(src/('run_pilot.py' if a.pipeline else 'train_adapter.py')),'--root',str(root),
          '--base-source',str(root/'walking-clearance-v2-r1-20261012/v2-source-r1'),
          '--output',str(src/a.label),'--headless','--num_envs',str(a.num_envs),'--sim_device','cuda:0',
          '--rl_device','cuda:0','--device','cuda:0','--num_threads','1','--seed','31','--iterations',str(a.iterations)]
    argv+=['--clearance-multiplier',str(a.clearance_multiplier)]
    if a.resume:argv+=['--resume',a.resume]
    if a.pipeline:argv+=['--eval-every',str(a.eval_every)]
    if a.smoke:argv+=['--smoke']
    if a.identity:argv+=['--identity']
    if a.evaluate:argv+=['--evaluate',a.evaluate]
    command='source '+shlex.quote(str(root/'remote_environment.sh'))+' && CUDA_VISIBLE_DEVICES='+gpu+' timeout '+str(a.timeout_seconds)+' '+shlex.join(argv)
    spec=dict(gpu=int(gpu),gpu_uuid=uuid,command=command,argv=argv,started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    with receipt.open('x') as f:json.dump(spec,f,indent=2)
    worker=subprocess.Popen([sys.executable,str(src/'dispatch.py'),'--worker','--root',str(root),'--label',a.label],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
    print(json.dumps(dict(worker_pid=worker.pid,**spec)))


if __name__=='__main__':main()
