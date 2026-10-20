"""Single explicit recovery after pre-PPO PhysX failure; trainer/data unchanged."""
from pathlib import Path
import sys,os,json,time,subprocess,fcntl,shutil
P=Path('/home/jovyan/vlad/FURRY/twist2-distributed-20261010/stance-finetune-20261013');R=P.parent;C=P/'code'
sys.path.insert(0,str(C))
from stance_queue import verify,gpu_snapshot,idle,identity,dump

def alive(receipt):
 try:return identity(receipt['pid'])==receipt
 except FileNotFoundError:return False

def main():
 lock=(P/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 verify()
 old=json.loads((P/'queue-state.json').read_text());assert old['phase']=='failed' and old['reason'] in ['Failed B-s11', "AssertionError('Never discard partial training')"]
 launch=json.loads((P/'B-s11.launch.json').read_text());assert not alive(launch['process'])
 assert not list((P/'runs/B-s11/training').rglob('*.pt')) and not (P/'runs/B-s11/training/metrics.jsonl').exists(),'Never discard partial training'
 assert 'illegal memory access' in (P/'runs/B-s11/training.log').read_text()
 for name in ['smoke-A-s11','smoke-B-s11']:assert json.loads((P/'runs'/name/'smoke-check.json').read_text())['passed']
 archive=P/'recovery-1';archive.mkdir(exist_ok=False)
 for name in ['queue-state.json','queue-start.json','launch.json','queue.log']:
  shutil.copy2(P/name,archive/name)
 (P/'runs/B-s11').rename(archive/'B-s11')
 for name in ['B-s11.launch.json','B-s11.exit.json','B-s11.log']:(P/name).rename(archive/name)
 dump(archive/'reason.json',dict(reason='PhysX CUDA illegal memory access during make_env, before checkpoint load/PPO; one explicit retry on GPU4; skip GPU3 during recovery, hardware cause unproven',trainer_changed=False,data_changed=False,utc=time.time()))
 receipts=json.loads((P/'launch.json').read_text());receipts.append(dict(name='recovery-queue',pid=os.getpid(),command=[sys.executable,__file__],utc=time.time()));dump(P/'launch.json',receipts)
 ext=json.loads((P/'A-s11.launch.json').read_text())['process'];jobs=[('B',11),('A',22),('B',22)];active={};done=['smoke-A-s11','smoke-B-s11'];deadline=json.loads((P/'queue-start.json').read_text())['utc']+86400
 while jobs or active or ext:
  if ext:
   status=json.loads((P/'runs/A-s11/status.json').read_text())
   if status['phase']=='complete':done.append('A-s11');ext=None
   elif status['phase']=='failed' or not alive(ext):raise RuntimeError('Existing A-s11 evaluation failed or process exited')
  for label,(child,gpu,log) in list(active.items()):
   rc=child.poll()
   if rc is None:continue
   log.close();del active[label];status=json.loads((P/'runs'/label/'status.json').read_text());dump(P/(label+'.exit.json'),dict(returncode=rc,status=status,utc=time.time()))
   if rc or status['phase']!='complete':raise RuntimeError('Failed '+label+'; no automatic retry')
   done.append(label)
  if time.time()>deadline or (P/'STOP_SCHEDULING').exists():raise RuntimeError('Scheduling deadline/stop reached')
  snapshot=gpu_snapshot();reserved={v[1] for v in active.values()}
  if jobs and len(active)<4:
   b,s=jobs[0];allowed=[4] if (b,s)==('B',11) else [0,1,2,4]
   choices=[g for g in snapshot if g['index'] in allowed and g['index'] not in reserved and idle(g)]
   if choices:
    verify();g=choices[0];label=b+'-s'+str(s);cmd=[str(R/'env/bin/python'),str(C/'stance_worker.py'),'--branch',b,'--seed',str(s),'--gpu',str(g['index'])];log=(P/(label+'.log')).open('x')
    child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONPATH=str(C)+':'+os.environ.get('PYTHONPATH','')),start_new_session=True)
    active[label]=(child,g['index'],log);jobs.pop(0);dump(P/(label+'.launch.json'),dict(command=cmd,gpu=g,process=identity(child.pid),utc=time.time(),recovery=1))
  dump(P/'queue-state.json',dict(phase='running' if active or ext else 'waiting_free_gpu',active=list(active)+(['A-s11'] if ext else []),done=done,pending=jobs,gpus=snapshot,recovery=1,utc=time.time()))
  if jobs or active or ext:time.sleep(30)
 subprocess.run([str(R/'eval-env/bin/python'),str(C/'stance_summary.py'),str(P/'evaluation')],check=True,stdout=(P/'summary.log').open('w'))
 dump(P/'queue-state.json',dict(phase='complete',done=done,recovery=1,utc=time.time()))
if __name__=='__main__':
 try:main()
 except BaseException as e:
  dump(P/'recovery-1-error.json',dict(error=repr(e),utc=time.time()))
  state=json.loads((P/'queue-state.json').read_text());state.update(phase='failed',reason=repr(e),utc=time.time());dump(P/'queue-state.json',state);raise
