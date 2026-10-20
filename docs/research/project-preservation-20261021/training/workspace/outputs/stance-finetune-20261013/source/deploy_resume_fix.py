from pathlib import Path
import os,signal,json,time,subprocess,hashlib
R=Path('/home/jovyan/vlad/FURRY/twist2-distributed-20261010');P=R/'stance-finetune-20261013';state=json.loads((P/'queue-state.json').read_text());assert not state['active'] and not state['done']
launch=json.loads((P/'launch.json').read_text());item=next(x for x in launch if x['name']=='queue');pid=item['pid'];argv=(Path('/proc')/str(pid)/'cmdline').read_bytes().split(b'\0');assert str(P/'code/stance_queue.py').encode() in argv;os.kill(pid,signal.SIGTERM)
for _ in range(50):
 try:
  if (Path('/proc')/str(pid)/'stat').read_text().split(')')[-1].split()[0]=='Z':break
 except FileNotFoundError:break
 time.sleep(.1)
else:raise RuntimeError('Queue did not stop')
archive=P/'preflight-revision1';archive.mkdir()
for n in ['queue-start.json','queue-state.json','queue.log','package-manifest.json','execution-manifest.json','launch.json']:(archive/n).write_bytes((P/n).read_bytes())
(archive/'resume_state.py').write_bytes((P/'code/resume_state.py').read_bytes())
for n in ['queue-start.json','queue-state.json','queue.log']:(P/n).unlink()
(P/'code/resume_state.py').write_bytes((P/'resume_state.pending.py').read_bytes());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();m=json.loads((P/'package-manifest.json').read_text());m['code/resume_state.py']=sha(P/'code/resume_state.py');(P/'package-manifest.json').write_text(json.dumps(m,indent=2));m=json.loads((P/'execution-manifest.json').read_text());m['package-manifest.json']=sha(P/'package-manifest.json');(P/'execution-manifest.json').write_text(json.dumps(m,indent=2))
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(P/'code')+':'+os.environ.get('PYTHONPATH',''))
with (P/'cpu-resume-check.log').open('w') as log:r=subprocess.run([str(R/'env/bin/python'),str(P/'code/check_resume_actual.py')],env=env,stdout=log,stderr=subprocess.STDOUT)
if r.returncode:
 (P/'queue-state.json').write_text(json.dumps(dict(phase='blocked_preflight',reason='CPU restore verification failed',active=[],done=[])));raise RuntimeError((P/'cpu-resume-check.log').read_text()[-5000:])
check=json.loads((P/'cpu-resume-check.json').read_text());assert check['passed'];print(json.dumps(check))
# Restore ordinary GPU visibility for scheduler; worker selects only a verified idle UUID.
env=dict(os.environ,PYTHONPATH=str(P/'code')+':'+os.environ.get('PYTHONPATH',''));log=(P/'queue.log').open('x');cmd=[str(R/'env/bin/python'),str(P/'code/stance_queue.py')];q=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=env);item.update(pid=q.pid,command=cmd,utc=time.time(),replaces_queue_pid=pid);(P/'launch.json').write_text(json.dumps(launch,indent=2));print('QUEUE',q.pid)
