"""Wait for authorized fine-tunes, export weights, evaluate frozen random scenes."""
from pathlib import Path
import hashlib,json,os,shlex,subprocess,sys,time
import numpy as np,torch
O=Path(__file__).resolve().parent;F=O/'fresh-start';REMOTE='/home/jovyan/vlad/FURRY/cup-homie-finetune-20261015';BASE='/home/jovyan/vlad/FURRY/cup-fast-check-20261015';kinds=['act','dp-epsilon','dp-sample'];started=time.time();state={'pid':os.getpid(),'started':started,'completed':[],'baseline_completed':[],'phase':'waiting'}
def status(**kw):
 state.update(kw);p=F/'evaluation-status.tmp';p.write_text(json.dumps(state,indent=2));p.replace(F/'evaluation-status.json')
def ssh(code,python='python3'):
 r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','andrei_gpu',python+' -'],input=code,text=True,capture_output=True,timeout=300)
 if r.returncode:raise RuntimeError(r.stderr[-2000:])
 return r.stdout
def evaluate(kind,label,checkpoint,stats=None):
 device='mps' if kind!='act' and torch.backends.mps.is_available() else 'cpu'
 cmd=[sys.executable,str(O/'fresh_eval.py'),'--kind',kind,'--checkpoint',str(checkpoint),'--label',label,'--device',device]
 if stats:cmd+=['--stats',str(stats)]
 with (F/f'{label}-{kind}.log').open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
 summary=json.loads((F/label/f'{kind}-summary.json').read_text());assert summary['n']==10 and not any(r['error'] for r in summary['results']);return summary
try:
 status(phase='waiting_for_initial_act',active='act')
 while not (F/'finetuned/act-summary.json').exists():time.sleep(10)
 first=json.loads((F/'finetuned/act-summary.json').read_text());assert first['n']==10 and not any(r['error'] for r in first['results'])
 state['completed'].append('act');state['results']={'act':{'successes':first['successes'],'n':first['n']}};status()
 while len(state['completed'])<len(kinds):
  status(phase='waiting_for_training',active=None)
  try:
   raw=ssh("from pathlib import Path\nimport json\nr={}\nfor kind in "+repr(kinds)+":\n p=Path("+repr(REMOTE)+")/(kind+'-run')/'status.json'\n r[kind]=json.loads(p.read_text()) if p.exists() else {}\nprint(json.dumps(r))\n");runs=json.loads(raw);status(remote=runs,last_poll=time.time(),last_connection_error=None)
  except Exception as exc:status(last_connection_error=str(exc));time.sleep(30);continue
  for kind in kinds:
   if kind in state['completed'] or runs[kind].get('status')!='completed':continue
   status(phase='exporting',active=kind);dest=O/'finetuned-weights'/kind;dest.mkdir(parents=True,exist_ok=True);remote_out=REMOTE+'/exports/'+kind;source=REMOTE+'/'+kind+'-run/latest.pt'
   if kind=='act':
    cmd=[BASE+'/env-act/bin/python',REMOTE+'/export_act.py','--source',source,'--out',remote_out]
    subprocess.run(['ssh','-o','BatchMode=yes','andrei_gpu',shlex.join(cmd)],check=True,timeout=300)
   else:
    code="from pathlib import Path\nimport torch\np=Path("+repr(remote_out)+");p.mkdir(parents=True,exist_ok=True)\nc=torch.load("+repr(source)+",map_location='cpu',weights_only=False)\ntorch.save({k:c[k] for k in ['ema','normalizer','metadata','step','epoch']},p/'checkpoint.pt')\n"
    ssh(code,BASE+'/env-dp/bin/python')
   hashes=json.loads(ssh("from pathlib import Path\nimport hashlib,json\np=Path("+repr(remote_out)+")\nprint(json.dumps({f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in p.iterdir() if f.is_file()}))\n"))
   subprocess.run(['scp','-o','BatchMode=yes','andrei_gpu:'+remote_out+'/*',str(dest)+'/'],check=True,timeout=1800)
   assert all(hashlib.sha256((dest/n).read_bytes()).hexdigest()==h for n,h in hashes.items())
   (dest/'transfer-verification.json').write_text(json.dumps(hashes,indent=2))
   if kind=='act':
    model=torch.jit.load(str(dest/'act-inference.pt')).eval();test=np.load(dest/'parity.npz')
    with torch.inference_mode():err=float(np.max(np.abs(model(torch.from_numpy(test['image']),torch.from_numpy(test['state'])).numpy()-test['expected'])))
    assert err<1e-4;del model
   status(phase='evaluating_finetuned',active=kind)
   result=evaluate(kind,'finetuned',dest/('act-inference.pt' if kind=='act' else 'checkpoint.pt'),dest/'stats.json' if kind=='act' else None);state['completed'].append(kind);state.setdefault('results',{})[kind]={'successes':result['successes'],'n':result['n']};status()
  if len(state['completed'])<len(kinds):time.sleep(30)
 report={'protocol':json.loads((F/'scenes.json').read_text()),'models':{kind:json.loads((F/'finetuned'/f'{kind}-summary.json').read_text()) for kind in kinds},'limitations':['Physics pauses during inference.','Old snapshot-start and teacher-forced continuation results are different protocols and not directly comparable.']}
 (F/'results.json').write_text(json.dumps(report,indent=2));status(phase='completed',elapsed_s=time.time()-started,active=None)
except Exception as exc:status(phase='failed',error=repr(exc));raise
