"""Offline sagittal reflection of TRAIN references; preserve validation bytes.

This is a G1-specific joint mapping, verified against MuJoCo FK before export.
It is deliberately not a generic augmentation for arbitrary robot models.
"""
import argparse, hashlib, json, shutil
from pathlib import Path
import mujoco
import numpy as np
from env import JOINTS


def opposite(name):
    if name.startswith('left_'):return 'right_'+name[5:]
    if name.startswith('right_'):return 'left_'+name[6:]
    return name


def reflect(q):
    q=np.asarray(q);out=q.copy()
    out[...,1]*=-1
    out[...,4]*=-1;out[...,6]*=-1  # S R S, S=diag(1,-1,1), quaternion wxyz
    for i,name in enumerate(JOINTS):
        sign=-1 if ('roll' in name or 'yaw' in name) else 1
        out[...,7+i]=sign*q[...,7+JOINTS.index(opposite(name))]
    return out


def augment(source, destination, assets):
    source=Path(source);destination=Path(destination);assets=Path(assets)
    if destination.exists():raise ValueError('Use a new output directory')
    manifest=json.loads((source/'manifest.json').read_text())
    if any(c.get('augmentation') for c in manifest['clips']):raise ValueError('Already augmented')
    model=mujoco.MjModel.from_xml_path(str(assets/'gmr_model.xml'))
    left=mujoco.MjData(model);right=mujoco.MjData(model)
    assert model.nq==36
    # Confirm dataset joint ordering before touching angles.
    assert [model.joint(i).name for i in range(1,model.njnt)]==list(JOINTS)
    # The fixed IMU mount is intentionally offset laterally in the model.
    checked_bodies=[i for i in range(1,model.nbody) if model.body(i).name!='imu_in_torso']
    mirror_bodies=[model.body(opposite(model.body(i).name)).id for i in checked_bodies]
    signs=np.array([1.,-1.,1.]);diagnostics=[];additions=[]
    destination.mkdir(parents=True)
    for c in list(manifest['clips']):
        src=source/c['file'];dst=destination/c['file'];dst.parent.mkdir(parents=True,exist_ok=True)
        if hashlib.sha256(src.read_bytes()).hexdigest()!=c['sha256']:raise ValueError('Source checksum mismatch: '+str(src))
        shutil.copy2(src,dst)
        if c['split']!='train':continue
        with np.load(src) as d:
            q=d['qpos'];mirrored=reflect(q);names=d['link_body_list'];fps=float(d['fps'])
        np.testing.assert_allclose(reflect(mirrored),q,atol=1e-12)
        maximum=0.;rotation_error=0.
        for i in range(0,len(q),25):
            left.qpos[:]=q[i];right.qpos[:]=mirrored[i]
            mujoco.mj_kinematics(model,left);mujoco.mj_kinematics(model,right)
            maximum=max(maximum,float(abs(right.xpos[checked_bodies]-left.xpos[mirror_bodies]*signs).max()))
        rotation_error=max(rotation_error,float(abs(right.xmat[checked_bodies].reshape(-1,3,3)-left.xmat[mirror_bodies].reshape(-1,3,3)*signs[None,:,None]*signs[None,None,:]).max()))
        if rotation_error>2e-5:raise ValueError(f'Reflection orientation failed {c["id"]}: {rotation_error}')
        if maximum>2e-5:raise ValueError(f'Reflection FK failed {c["id"]}: {maximum}')
        for i in range(1,model.njnt):
            if model.jnt_limited[i]:
                j=model.jnt_qposadr[i];lo,hi=model.jnt_range[i]
                if (mirrored[:,j]<lo-1e-5).any() or (mirrored[:,j]>hi+1e-5).any():raise ValueError('Mirrored joint limit: '+model.joint(i).name)
        bodies=[model.body(str(n)).id for n in names];local=[]
        for row in mirrored:
            right.qpos[:]=row;mujoco.mj_kinematics(model,right);local.append(right.xpos[bodies]-row[:3])
        path=destination/'train'/(c['id']+'-mirror.npz')
        np.savez_compressed(path,qpos=mirrored,fps=fps,local_body_pos=np.asarray(local),link_body_list=names)
        additions.append(dict(c,id=c['id']+'-mirror',file=str(path.relative_to(destination)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),augmentation='sagittal_reflection',parent_id=c['id']))
        diagnostics.append({'parent_id':c['id'],'max_fk_position_error_m':maximum,'max_fk_rotation_matrix_error':rotation_error})
    manifest['clips']+=additions
    manifest['augmentation']={'type':'G1 sagittal reflection, train only','source_manifest_sha256':hashlib.sha256((source/'manifest.json').read_bytes()).hexdigest(),'independent_demonstrations_added':0,'augmented_clips':len(additions),'excluded_fixed_sensor_frame':'imu_in_torso (asymmetric fixed mount); all moving links checked','checks':diagnostics}
    manifest['duration_seconds']={s:sum(c['seconds'] for c in manifest['clips'] if c['split']==s) for s in ['train','validation','replay']}
    manifest['validation_fraction_note']='Original source-group split unchanged; augmented seconds are not independent data.'
    (destination/'manifest.json').write_text(json.dumps(manifest,indent=2))
    for c in manifest['clips']:
        if c['split']=='validation':assert (source/c['file']).read_bytes()==(destination/c['file']).read_bytes()
    return manifest['augmentation']


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('destination');p.add_argument('--assets',required=True);a=p.parse_args()
    print(json.dumps(augment(a.source,a.destination,a.assets),indent=2))
