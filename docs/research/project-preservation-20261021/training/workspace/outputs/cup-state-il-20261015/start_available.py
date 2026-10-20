"""Start the two requested bounded experiments on currently idle H100s."""
from pathlib import Path
import hashlib,json,os,signal,subprocess,sys,time
O=Path(__file__).resolve().parent
assert not (O/'available-launch.json').exists(), 'Already launched'
m=json.loads((O/'transfer-manifest.json').read_text())
for f,h in m.items():assert hashlib.sha256((O/f).read_bytes()).hexdigest()==h,f
s=json.loads((O/'dispatch-status.json').read_text())
assert not s['jobs'], 'Existing dispatcher already started training; do not duplicate'
pid=json.loads((O/'dispatch-launch.json').read_text())['pid']
proc=Path(f'/proc/{pid}/cmdline')
if proc.exists():
    cmd=proc.read_bytes().replace(b'\0',b' ').decode()
    assert str(O/'dispatch.py') in cmd and '--worker' in cmd,cmd
    os.kill(pid,signal.SIGTERM)
    time.sleep(1)
s=json.loads((O/'dispatch-status.json').read_text());assert not s['jobs']
s.update(phase='replaced_by_available_launcher',updated=time.time());(O/'dispatch-status.json').write_text(json.dumps(s,indent=2))
def free():
    busy=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    return [(int(v[0]),v[1]) for line in rows.splitlines() if (v:=[x.strip() for x in line.split(',')]) and 'H100' in v[2] and v[1] not in busy and int(v[3])<256 and int(v[4])==0]
slots=free();assert len(slots)>=2,slots
receipt={'started':time.time(),'max_training_s_each':1800,'max_steps_each':12000,'jobs':[]}
for kind,(index,uuid) in zip(['bc-rnn','diffusion'],slots[:2]):
    assert (index,uuid) in free(), 'GPU no longer idle'
    assert not (O/('run-'+kind)).exists()
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=uuid,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1')
    cmd=[sys.executable,str(O/'pipeline.py'),'--kind',kind]
    with (O/(kind+'.log')).open('x') as log:
        p=subprocess.Popen(cmd,cwd=O,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,stdin=subprocess.DEVNULL)
    receipt['jobs'].append({'kind':kind,'pid':p.pid,'gpu_index':index,'gpu_uuid':uuid,'command':cmd})
    (O/'available-launch.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt))
