"""Adopt the existing trainer and schedule remaining references on idle GPUs."""
import csv, fcntl, hashlib, json, os, subprocess, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
lock = open('queue.lock', 'w')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
lock.write(str(os.getpid())); lock.flush()
manifest = json.loads(Path('launch-manifest.json').read_text())
for name, digest in manifest['sha256'].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest, name
assert json.loads(Path('smoke-gpu-4096/status.json').read_text())['status'] == 'completed'
refs = json.loads(Path('data/manifest.json').read_text())['accepted']
active, completed, failed = {}, [], []
first = json.loads(Path('queue-status.json').read_text())
pid = int(json.loads((Path(first['output'])/'status.json').read_text())['pid'])
cmd = Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode()
assert '-m dexcup.train' in cmd and refs[0]['file'] in cmd
active[refs[0]['file']] = dict(pid=pid, gpu_index=first['gpu_index'], gpu_uuid=first['gpu_uuid'], output=first['output'], adopted=True)
pending = refs[1:]
children = {}

def free_gpus():
    rows = csv.reader(subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True).splitlines())
    busy = set(subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid', '--format=csv,noheader'], text=True).splitlines())
    reserved = {x['gpu_uuid'] for x in active.values()}
    return [(i.strip(), u.strip()) for i,u,m,v in rows if u.strip() not in busy | reserved and int(m)<1000 and int(v)<5]

while active or pending:
    for name, row in list(active.items()):
        p = Path(f"/proc/{row['pid']}/stat")
        alive = p.exists() and p.read_text().split(') ')[1][0] != 'Z'
        if name in children: alive = children[name].poll() is None
        if not alive:
            result = json.loads((Path(row['output'])/'status.json').read_text())
            valid = result.get('status') == 'completed' and result.get('iteration') == 12000
            (completed if valid else failed).append(dict(reference=name, **row, result=result))
            del active[name]
    if pending:
        candidates = free_gpus()
        if candidates:
            time.sleep(5)
            confirmed = free_gpus()
            for gpu in candidates:
                if not pending: break
                if gpu not in confirmed: continue
                ref = pending.pop(0); out = ROOT/('train-'+ref['id'])
                assert not out.exists(), f'Refuse to overwrite {out}'
                env = os.environ.copy(); env['CUDA_VISIBLE_DEVICES'] = gpu[1]
                with open(str(out)+'.log', 'x') as log:
                    child = subprocess.Popen(['bash','run.sh','--reference','data/'+ref['file'],'--output',str(out),'--device','cuda:0','--num-envs','4096','--iterations','12000'], env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                children[ref['file']] = child
                active[ref['file']] = dict(pid=child.pid,gpu_index=gpu[0],gpu_uuid=gpu[1],output=str(out),adopted=False)
    state = dict(status='running' if active or pending else ('failed' if failed else 'completed'), pid=os.getpid(), updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),active=active,pending=[r['file'] for r in pending],completed=completed,failed=failed)
    Path('parallel-status.tmp').write_text(json.dumps(state,indent=2)); Path('parallel-status.tmp').replace('parallel-status.json')
    if active or pending: time.sleep(20)
