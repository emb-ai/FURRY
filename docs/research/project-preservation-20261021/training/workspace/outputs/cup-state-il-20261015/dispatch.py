"""Finite two-run dispatcher, only explicitly permitted GPU3/4/6.

Uses two GPUs when possible, otherwise the second starts after the first.
Never stops another process. Does not retry failed trainings.
"""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,sys,time,traceback
O=Path(__file__).resolve().parent

def write(path,obj):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(obj,indent=2));temp.replace(path)

def free_gpus():
    busy=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
    output=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    found=[]
    for row in output.splitlines():
        index,uuid,name,memory,util=[x.strip() for x in row.split(',')]
        if index in ['3','4','6'] and 'H100' in name and uuid not in busy and int(memory)<100 and int(util)==0:found.append((index,uuid))
    return found

def main():
    p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true');a=p.parse_args()
    if not a.worker:
        assert not (O/'dispatch-launch.json').exists()
        manifest=json.loads((O/'transfer-manifest.json').read_text())
        for file,h in manifest.items():assert hashlib.sha256((O/file).read_bytes()).hexdigest()==h,file
        with (O/'dispatch.log').open('x') as log:
            proc=subprocess.Popen([sys.executable,str(O/'dispatch.py'),'--worker'],cwd=O,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        receipt={'pid':proc.pid,'started':time.time(),'allowed_gpu_indices':[3,4,6],'methods':['bc-rnn','diffusion'],'max_training_s_each':1800,'max_steps_each':12000,'max_idle_wait_s':900}
        write(O/'dispatch-launch.json',receipt);print(json.dumps(receipt));return
    state={'phase':'running','started':time.time(),'jobs':[],'pending':['bc-rnn','diffusion']};processes={};waiting_since=time.monotonic()
    try:
        while state['pending'] or processes:
            for kind,proc in list(processes.items()):
                code=proc.poll()
                if code is not None:
                    entry=next(j for j in state['jobs'] if j['kind']==kind);entry.update(exit_code=code,finished=time.time());del processes[kind];waiting_since=time.monotonic()
            for index,uuid in free_gpus():
                if not state['pending']:break
                if any(j['gpu_uuid']==uuid and j['kind'] in processes for j in state['jobs']):continue
                assert (index,uuid) in free_gpus()
                kind=state['pending'].pop(0);env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=uuid,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1')
                cmd=[sys.executable,str(O/'pipeline.py'),'--kind',kind]
                with (O/(kind+'.log')).open('x') as log:
                    proc=subprocess.Popen(cmd,cwd=O,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                processes[kind]=proc;state['jobs'].append({'kind':kind,'pid':proc.pid,'gpu_index':int(index),'gpu_uuid':uuid,'started':time.time(),'command':cmd});waiting_since=time.monotonic()
            state['updated']=time.time();write(O/'dispatch-status.json',state)
            if state['pending'] and not processes and time.monotonic()-waiting_since>900:
                state['phase']='waiting_expired';write(O/'dispatch-status.json',state);return
            if state['pending'] or processes:time.sleep(20)
        state['phase']='completed' if all(j.get('exit_code')==0 for j in state['jobs']) else 'failed'
        state['finished']=time.time();write(O/'dispatch-status.json',state)
    except Exception:
        state.update(phase='dispatch_failed',error=traceback.format_exc());write(O/'dispatch-status.json',state);raise
if __name__=='__main__':main()
