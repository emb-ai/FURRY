"""Recompute phase from cached TRAIN FK; preserve auditable geometric evidence."""
import argparse
import json
from pathlib import Path
import numpy as np
from build_phase import infer
from contact_clearance import sha


def rebuild(cache, output, dataset):
    from author_dataset import load_motion
    from scipy.spatial.transform import Rotation
    meta=json.loads(cache.with_suffix('.json').read_text())
    geometry=np.load(cache.with_name(cache.stem+'-kinematics.npz'),allow_pickle=False)
    old=np.load(cache,allow_pickle=False);arrays={};coverage={};audit=[]
    for filename,row in meta['files'].items():
        key=row['key'];n=len(old[key]);fps=row['fps']
        result=np.zeros((n,2,3),np.float32);intervals=[]
        motion=load_motion(dataset/filename)
        family=filename.split('/')[1] if row['trusted_author'] else 'quest'
        if key in geometry:
            x=geometry[key];h=x[:,:2];pos=x[:,2:].reshape(-1,2,3)
            result,intervals=infer(h,pos,fps)
            speed=np.linalg.norm(np.gradient(pos,1/fps,axis=0),axis=-1)
            slow=speed<.15
            row['phase_floor_offset_m']=float(np.quantile(h[slow],.2)) if slow.any() else None
            if intervals and sum(a['family']==family for a in audit)<12:
                audit.append(dict(file=filename,family=family,interval=intervals[len(intervals)//2],
                                  floor_offset_m=row['phase_floor_offset_m']))
        row['swing_intervals']=intervals;arrays[key]=result
        rot=Rotation.from_quat(motion['root_rot'])
        vel=rot.inv().apply(np.gradient(motion['root_pos'],1/fps,axis=0))
        yaw=np.gradient(np.unwrap(rot.as_euler('xyz')[:,2]),1/fps)
        moving=np.linalg.norm(vel[:,:2],axis=-1)>.05
        groups=dict(all=np.ones(n,bool),walking=moving,backward=vel[:,0]<-.05,
                    yaw_only=(~moving)&(abs(yaw)>.15),idle=(~moving)&(abs(yaw)<=.15),
                    transition=abs(np.gradient(moving.astype(float)))>0)
        for group,mask in groups.items():
            c=coverage.setdefault(family+'/'+group,dict(foot_frames=0,valid=0,swing=0))
            c['foot_frames']+=int(mask.sum())*2;c['valid']+=int(result[mask,:,1].sum())
            c['swing']+=int(((result[mask,:,0]>.005)&(result[mask,:,1]>0)).sum())
    np.savez_compressed(output,**arrays)
    for c in coverage.values():
        c['valid_fraction']=c['valid']/max(1,c['foot_frames']);c['swing_fraction']=c['swing']/max(1,c['foot_frames'])
    meta.update(schema='train-kinematic-phase-v2',coverage=coverage,npz_sha256=sha(output),approved=False,
                phase_floor_calibration='20th percentile of authored reference sole height at speed < .15m/s; |offset| <= 6.5cm; phase classification only',
                audit_intervals=audit)
    output.with_suffix('.json').write_text(json.dumps(meta,indent=2))
    print(json.dumps(dict(coverage=coverage,audit_intervals=audit)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('cache','output','dataset'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();rebuild(a.cache,a.output,a.dataset)
