"""Server-side serial queue. Uses only a GPU with no compute processes.

The GPU smoke must succeed before any 12k-iteration training run. No other
process is stopped or modified. Independent of the submitting Mac/SSH shell.
"""
import csv,fcntl,hashlib,json,os,signal,subprocess,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parent
os.chdir(ROOT)
lock=(ROOT/'queue.lock').open('w')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
lock.write(str(os.getpid()));lock.flush()
child=None
def status(state,**kw):
    data=dict(status=state,pid=os.getpid(),updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kw)
    p=ROOT/'queue-status.tmp';p.write_text(json.dumps(data,indent=2));p.replace(ROOT/'queue-status.json')
def free_gpus():
    out=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
    busy=set(x.strip() for x in apps.splitlines())
    rows=list(csv.reader(out.splitlines()))
    return [(i.strip(),u.strip()) for i,u,mem,util in rows if u.strip() not in busy and int(mem)<1000 and int(util)<5]
def wait_gpu():
    while True:
        candidates=free_gpus()
        if candidates:
            time.sleep(5)
            again=free_gpus()
            for gpu in candidates:
                if gpu in again:return gpu
        status('waiting_for_free_gpu',required='one H100 with no compute processes, memory < 1000 MiB',references=4)
        time.sleep(30)
def run(reference,out,num_envs,iterations,gpu,smoke=False):
    global child
    args=['bash','run.sh','--reference','data/'+reference,'--output',str(out),'--device','cuda:0','--num-envs',str(num_envs),'--iterations',str(iterations)]
    if smoke:args+=['--smoke']
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=gpu[1]
    status('gpu_smoke' if smoke else 'training',reference=reference,output=str(out),gpu_index=gpu[0],gpu_uuid=gpu[1],num_envs=num_envs,iterations=iterations)
    with open(str(out)+'.log','w') as log:
        child=subprocess.Popen(args,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        code=child.wait();child=None
    result=json.loads((out/'status.json').read_text()) if (out/'status.json').exists() else {}
    if code or result.get('status')!='completed' or not result.get('actor_changed'):
        raise RuntimeError(f'{out} failed: exit={code}, result={result}')
    native=json.loads((out/'native-contract.json').read_text())
    if not native['gpu_dynamics_enabled']:raise RuntimeError('GPU run did not use GPU physics')
def stop(sig,frame):
    if child is not None:os.killpg(child.pid,signal.SIGTERM)
    status('stopped',signal=sig);raise SystemExit(128+sig)
signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
try:
    evidence=json.loads((ROOT/'smoke-cpu-final/status.json').read_text())
    assert evidence['status']=='completed' and evidence['actor_changed']
    frozen=json.loads((ROOT/'launch-manifest.json').read_text())
    for filename,digest in frozen['sha256'].items():
        assert hashlib.sha256((ROOT/filename).read_bytes()).hexdigest()==digest,filename
    manifest=json.loads((ROOT/'data/manifest.json').read_text())
    assert len(manifest['accepted'])==4
    gpu=wait_gpu()
    run(manifest['accepted'][0]['file'],ROOT/'smoke-gpu-4096',4096,2,gpu,True)
    for clip in manifest['accepted']:
        # Recheck before each subsequent policy; do not share an occupied GPU.
        gpu=wait_gpu()
        run(clip['file'],ROOT/('train-'+clip['id']),4096,12000,gpu)
    status('completed',references=4)
except Exception as e:
    status('failed',error=str(e),traceback=traceback.format_exc());raise
