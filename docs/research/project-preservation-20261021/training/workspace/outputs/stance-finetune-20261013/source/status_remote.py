from pathlib import Path
import json,time,subprocess,hashlib
R=Path('/home/jovyan/vlad/FURRY/twist2-distributed-20261010');P=R/'stance-finetune-20261013'
def read(p):
 try:return json.loads(p.read_text())
 except FileNotFoundError:return None
out={'utc':time.time(),'queue':read(P/'queue-state.json'),'launch':read(P/'launch.json'),'setup':read(P/'setup-receipt.json'),'baselines':{},'logs':{},'runs':{}}
for mode in ['A','B']:
 p=P/'evaluation'/('initial-'+mode+'.json');x=read(p)
 if x:out['baselines'][mode]={k:x[k] for k in ['falls','total_trials','survived_seconds','reference_seconds','policy_sha256','dataset_manifest_sha256','initial_state_hashes']}
for name in ['setup.log','queue.log','baseline-A.log','baseline-B.log']:
 p=P/name
 if p.exists():out['logs'][name]=p.read_text()[-1800:]
for p in (P/'runs').glob('*/status.json') if (P/'runs').exists() else []:out['runs'][p.parent.name]=read(p)
for item in out['launch'] or []:
 d=Path('/proc')/str(item['pid']);item['actual_argv']=(d/'cmdline').read_bytes().replace(b'\0',b' ').decode() if d.exists() else None
out['source_hashes']={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for area in ['TWIST2/rsl_rl','TWIST2/legged_gym','TWIST2/pose','g1_sim'] for p in (R/area).rglob('*.py')}
print(json.dumps(out,indent=2))
