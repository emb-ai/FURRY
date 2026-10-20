from pathlib import Path
import os,subprocess,json,hashlib,tarfile,time
R=Path('/home/jovyan/vlad/FURRY/twist2-distributed-20261010');P=R/'stance-finetune-20261013';archive=R/'stance-finetune-20261013-package.tar.gz';overlay=R/'stance-finetune-20261013-execution.tar.gz'
assert not P.exists(),'Do not overwrite existing experiment'
P.mkdir()
for f in [archive,overlay]:
 with tarfile.open(f) as t:
  for m in t.getmembers():assert not Path(m.name).is_absolute() and '..' not in Path(m.name).parts
  t.extractall(P)
env=dict(os.environ);env['PYTHONPATH']=str(P/'code')+':'+env.get('PYTHONPATH','')
subprocess.run([str(R/'env/bin/python'),str(P/'code/canonicalize_dataset.py')],env=env,check=True)
with (P/'setup.log').open('w') as f:subprocess.run([str(R/'eval-env/bin/python'),str(P/'setup.py')],stdout=f,stderr=subprocess.STDOUT,env=env,check=True)
with (P/'cpu-resume-check.log').open('w') as f:subprocess.run([str(R/'env/bin/python'),str(P/'code/check_resume_actual.py')],stdout=f,stderr=subprocess.STDOUT,env=dict(env,CUDA_VISIBLE_DEVICES=''),check=True)
receipts=[]
for name,script in [('baseline','baseline.py'),('queue','stance_queue.py')]:
 log=(P/(name+'.log')).open('x');cmd=[str(R/'env/bin/python'),str(P/'code'/script)];p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True);receipts.append(dict(name=name,pid=p.pid,command=cmd,utc=time.time()))
(P/'launch.json').write_text(json.dumps(receipts,indent=2));print(json.dumps(receipts,indent=2))
