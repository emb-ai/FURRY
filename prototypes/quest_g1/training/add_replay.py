"""Add trusted pinned TWIST2 public walking fixtures as a 20% replay mixture."""
import argparse,pickle,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation,Slerp
from prepare import sha
p=argparse.ArgumentParser();p.add_argument('dataset',type=Path);p.add_argument('upstream',type=Path);a=p.parse_args();manifest=json.loads((a.dataset/'manifest.json').read_text());out=a.dataset/'replay';out.mkdir(exist_ok=True)
for f in sorted((a.upstream/'assets/example_motions').glob('0807_yanjie_walk_*.pkl')):
 if f.stem.endswith('002'):continue # pre-existing public fixture exclusion, independent of Quest validation
 with f.open('rb') as stream:d=pickle.load(stream)
 q=np.c_[d['root_pos'],np.asarray(d['root_rot'])[:,[3,0,1,2]],d['dof_pos']];t=np.arange(len(q))/d['fps'];grid=np.arange(0,t[-1],.01);new=np.empty((len(grid),36));cols=np.r_[0:3,7:36];new[:,cols]=np.column_stack([np.interp(grid,t,q[:,j]) for j in cols]);new[:,3:7]=Slerp(t,Rotation.from_quat(q[:,[4,5,6,3]]))(grid).as_quat()[:,[3,0,1,2]];new[:,:2]-=new[0,:2]
 dest=out/(f.stem+'.npz');np.savez_compressed(dest,qpos=new,fps=100);manifest['clips'].append({'id':f.stem,'file':str(dest.relative_to(a.dataset)),'stage':0,'split':'replay','seconds':(len(new)-1)/100,'sha256':sha(dest),'source_sha256':sha(f)})
manifest['replay_sampling_fraction']=.2;manifest['replay_source_revision']='b06178f19a22f2138cbd31f60c6d494bc263f67d';(a.dataset/'manifest.json').write_text(json.dumps(manifest,indent=2))
