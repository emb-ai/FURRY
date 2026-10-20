from pathlib import Path
import json,hashlib,os,signal,time,subprocess,sys,pickle
import numpy as np
R=Path('/home/jovyan/vlad/FURRY/twist2-distributed-20261010');P=R/'stance-finetune-20261013';sys.path.insert(0,str(P/'code'))
from author_dataset import load_motion,validate_motion
state=json.loads((P/'queue-state.json').read_text());assert not state['active'] and not state['done']
launch=json.loads((P/'launch.json').read_text());item=next(x for x in launch if x['name']=='queue');pid=item['pid'];argv=(Path('/proc')/str(pid)/'cmdline').read_bytes().split(b'\0');assert str(P/'code/stance_queue.py').encode() in argv
os.kill(pid,signal.SIGTERM)
for _ in range(50):
 try:
  s=(Path('/proc')/str(pid)/'stat').read_text().split(')')[-1].split()[0]
  if s=='Z':break
 except FileNotFoundError:break
 time.sleep(.1)
else:raise RuntimeError('Queue did not stop')
archive=P/'preflight-revision0';archive.mkdir(exist_ok=False)
for n in ['queue-start.json','queue-state.json','queue.log','execution-manifest.json','package-manifest.json','setup-receipt.json','launch.json']:
 (archive/n).write_bytes((P/n).read_bytes())
for n in ['queue-start.json','queue-state.json','queue.log']:(P/n).unlink()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();manifest=json.loads((P/'B-dataset/manifest.json').read_text());report=[]
for c in manifest['clips']:
 if c['source']!='quest':continue
 path=P/'B-dataset'/c['file'];before=sha(path);x=load_motion(path);validate_motion(x)
 with path.open('wb') as f:pickle.dump(x,f,protocol=4)
 with path.open('rb') as f:y=pickle.load(f)
 assert x.keys()==y.keys()
 for key in x:assert np.array_equal(x[key],y[key]),(c['id'],key)
 c['sha256']=sha(path);report.append(dict(id=c['id'],before_sha256=before,after_sha256=c['sha256'],all_arrays_exact=True))
(P/'B-dataset/manifest.json').write_text(json.dumps(manifest,indent=2));(P/'pickle-compatibility.json').write_text(json.dumps(dict(numpy=np.__version__,clips=report,raw_references_unchanged=True),indent=2))
pm=json.loads((P/'package-manifest.json').read_text())
for f in pm:
 if f.startswith('B-dataset/') and (f.endswith('.pkl') or f.endswith('manifest.json')):pm[f]=sha(P/f)
(P/'package-manifest.json').write_text(json.dumps(pm,indent=2));em=json.loads((P/'execution-manifest.json').read_text());em['package-manifest.json']=sha(P/'package-manifest.json');(P/'execution-manifest.json').write_text(json.dumps(em,indent=2))
subprocess.run([str(R/'eval-env/bin/python'),str(P/'setup.py')],check=True)
log=(P/'queue.log').open('x');cmd=[str(R/'env/bin/python'),str(P/'code/stance_queue.py')];q=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);item.update(pid=q.pid,command=cmd,utc=time.time(),replaces_queue_pid=pid);(P/'launch.json').write_text(json.dumps(launch,indent=2));print(json.dumps({'numpy':np.__version__,'converted_clips':len(report),'all_arrays_exact':True,'new_queue_pid':q.pid,'baseline_process_untouched':True}))
