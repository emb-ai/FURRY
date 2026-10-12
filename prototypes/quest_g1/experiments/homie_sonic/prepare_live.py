"""Convert native source-only GMR exports, excluding their teacher-forced warmup."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('csv',type=Path,nargs='+');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);receipt=[]
for src in a.csv:
 x=np.genfromtxt(src,delimiter=',',names=True);x=x[x['warmup']==0]
 assert np.all(x['ref_available']==1)
 assert np.allclose(np.diff(x['elapsed']),.01,atol=1e-7)
 q=np.column_stack([x[f'ref{i}'] for i in range(36)]);cmd=np.column_stack([x[f'cmd_{i}'] for i in range(35)]);goal=np.column_stack([x[k] for k in ['goal_x','goal_y','goal_yaw']])
 assert np.isfinite(q).all() and np.isfinite(cmd).all() and np.isfinite(goal).all()
 name=src.stem.removeprefix('probes-').removesuffix('-trajectory-off');dest=a.output/(name+'.npz')
 np.savez_compressed(dest,qpos=q,fps=100.,commands=cmd,camera_goal=goal,source_time=x['elapsed'])
 receipt.append(dict(source_name=src.name,source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),output_name=dest.name,frames=len(x),all_applying=bool(np.all(x['applying']==1)),semantics='native source-only GMR; saved physical qpos excluded; catch-up recomputed per rollout'))
(a.output/'manifest.json').write_text(json.dumps(receipt,indent=2)+'\n')
