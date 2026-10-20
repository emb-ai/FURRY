"""One finite run per architecture; never stop or reuse an occupied GPU."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,sys,time
O=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--gpus',nargs='+',required=True);args=p.parse_args()
assert not (O/'launch.json').exists()
manifest=json.loads((O/'transfer-manifest.json').read_text())
for file,expected in manifest.items():assert hashlib.sha256((O/file).read_bytes()).hexdigest()==expected,file
def free():
    busy=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    result=[]
    for row in rows.splitlines():
        idx,uid,name,memory,util=[x.strip() for x in row.split(',')]
        if idx in args.gpus and 'H100' in name and uid not in busy and int(memory)<100 and int(util)==0:result.append((idx,uid))
    return result
available=free();assert len(available)>=2,available
selected=available[:2];jobs=[]
for kind,(index,uid) in zip(['bc-rnn','diffusion'],selected):
    assert (index,uid) in free(),f'GPU {index} became occupied'
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=uid,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1')
    cmd=[sys.executable,str(O/'pipeline.py'),'--kind',kind]
    with (O/(kind+'.log')).open('x') as log:
        proc=subprocess.Popen(cmd,cwd=O,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    jobs.append({'kind':kind,'pid':proc.pid,'gpu_index':index,'gpu_uuid':uid,'command':cmd,'time':time.time()})
    (O/'launch.json').write_text(json.dumps(jobs,indent=2))
print(json.dumps(jobs),flush=True)
