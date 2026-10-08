"""Exclude infeasible velocity bursts without mixing train and validation."""
import argparse,json,pickle
from pathlib import Path
import numpy as np
from scipy.ndimage import maximum_filter1d
from prepare import regions,sha

def main():
 p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('output',type=Path);a=p.parse_args();a.output.mkdir(exist_ok=True,parents=True);r=json.loads((a.source/'manifest.json').read_text());clips=r.pop('clips');r['pre_curation_duration_seconds']=r['duration_seconds'];r['curation']={'leg_torso_speed_ceiling_rad_s':12,'arm_speed_ceiling_rad_s':16,'exclude_margin_s':.25,'min_clip_s':2,'note':'Conservative curation thresholds, not hardware specifications'};r['clips']=[];r['dropped_bursts']=[]
 for c in clips:
  x=np.load(a.source/c['file']);q=x['qpos'];v=abs(np.diff(q[:,7:],axis=0))*100;bad=np.r_[False,(v[:,:15]>12).any(1)|(v[:,15:]>16).any(1)];expanded=maximum_filter1d(bad.astype(int),51,mode='constant')>0
  r['dropped_bursts'].append({'id':c['id'],'flagged_frames':int(bad.sum()),'excluded_frames':int(expanded.sum())})
  for n,ids in enumerate(regions(~expanded)):
   if len(ids)<201:continue
   s,e=int(ids[0]),int(ids[-1])+1;qq=q[s:e].copy();qq[:,:2]-=qq[0,:2];local=x['local_body_pos'][s:e];names=x['link_body_list'];name=f"{c['id']}-part{n:02d}";dest=a.output/c['split']/f'{name}.npz';dest.parent.mkdir(exist_ok=True)
   np.savez_compressed(dest,qpos=qq,fps=100,local_body_pos=local,link_body_list=names)
   d={'fps':100,'root_pos':qq[:,:3],'root_rot':qq[:,[4,5,6,3]],'dof_pos':qq[:,7:],'local_body_pos':local,'link_body_list':names.tolist()}
   with dest.with_suffix('.pkl').open('wb') as f:pickle.dump(d,f,protocol=4)
   cc=dict(c,id=name,file=str(dest.relative_to(a.output)),seconds=(e-s-1)/100,source_start_wall_s=c['source_start_wall_s']+s/100,source_end_wall_s=c['source_start_wall_s']+(e-1)/100,sha256=sha(dest),max_joint_speed_rad_s=float(abs(np.diff(qq[:,7:],axis=0)).max()*100));r['clips'].append(cc)
 r['duration_seconds']={s:sum(c['seconds'] for c in r['clips'] if c['split']==s) for s in ['train','validation']};r['validation_fraction']=r['duration_seconds']['validation']/sum(r['duration_seconds'].values());(a.output/'manifest.json').write_text(json.dumps(r,indent=2));print(r['duration_seconds'],r['validation_fraction'],len(r['clips']))
 for split in ['train','validation']:
  lines=['root_path: '+str(a.output.resolve()),'motions:']
  for c in r['clips']:
   if c['split']==split:lines+=['- file: '+str(Path(c['file']).with_suffix('.pkl')),'  weight: '+str(c['seconds'])]
  (a.output/f'{split}.yaml').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
