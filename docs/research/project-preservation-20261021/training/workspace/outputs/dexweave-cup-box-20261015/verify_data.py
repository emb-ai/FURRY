"""Independent FK and archive integrity checks for every exported demonstration."""
import json,hashlib,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import mujoco as mj
from scipy.spatial.transform import Rotation as R

root=Path(__file__).resolve().parent;data=root/'data'
manifest=json.loads((data/'manifest.json').read_text())
m=mj.MjModel.from_binary_path(str(data/'assets/reference.mjb'));d=mj.MjData(m)
joints=list(ET.parse(data/'assets/robot.urdf').getroot().iter('joint'))
max_pos=max_rot=0.;checked=0
for clip in manifest['accepted']:
    path=data/clip['file'];assert hashlib.sha256(path.read_bytes()).hexdigest()==clip['sha256']
    a=np.load(path);assert a['joint_pos'].shape==(clip['frames'],43)
    assert np.max(np.abs(np.diff(a['source_time'])-.02))<1e-10
    assert all(np.isfinite(a[key]).all() for key in a.files if a[key].dtype.kind in 'fi')
    for index in np.unique(np.linspace(0,clip['frames']-1,20).astype(int)):
        d.qpos[:]=a['qpos'][index];mj.mj_forward(m,d)
        transforms={'pelvis':(d.xpos[1].copy(),R.from_quat(np.roll(d.xquat[1],-1)))}
        for j in joints:
            parent=j.find('parent').get('link');child=j.find('child').get('link');o=j.find('origin')
            pos=np.fromstring(o.get('xyz'),sep=' ');rot=R.from_euler('xyz',np.fromstring(o.get('rpy'),sep=' '));pp,pr=transforms[parent]
            if j.get('type')!='fixed':
                ji=mj.mj_name2id(m,mj.mjtObj.mjOBJ_JOINT,j.get('name'));axis=np.fromstring(j.find('axis').get('xyz'),sep=' ')
                rot=rot*R.from_rotvec(axis*d.qpos[m.jnt_qposadr[ji]])
            transforms[child]=(pp+pr.apply(pos),pr*rot);bi=mj.mj_name2id(m,mj.mjtObj.mjOBJ_BODY,child)
            max_pos=max(max_pos,float(np.linalg.norm(transforms[child][0]-d.xpos[bi])))
            max_rot=max(max_rot,float((transforms[child][1].inv()*R.from_quat(np.roll(d.xquat[bi],-1))).magnitude()))
        checked+=1
assert max_pos<1e-8 and max_rot<1e-8
report=dict(accepted_episodes=len(manifest['accepted']),independent_poses_checked=checked,all_links_checked=53,max_urdf_fk_position_error_m=max_pos,max_urdf_fk_rotation_error_rad=max_rot,hashes_verified=True,resampling_hz=50,simulation_import_verified=False)
(root/'data-verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
