"""Frozen Quest GMR references -> named, 50 Hz body-only reference package.

Virtual legs remain the source. Ankle stationarity is a geometric support
proposal, not a measurement of load or a fabricated SDK confidence value.
The original continuous GMR solve/cuts are retained; this is not a claim to
reproduce DexWeave's unpublished source-to-robot retargeter.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import mujoco as mj
from scipy.spatial.transform import Rotation
from .spec import JOINTS, LINKS

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read_csv(path):
    with Path(path).open() as f:
        cols = f.readline().strip().split(',')
    return np.loadtxt(path, delimiter=',', skiprows=1), {k:i for i,k in enumerate(cols)}

def retained_runs(mask, minimum=6):
    """Offline minimum duration filter; never bridge an invalid sample."""
    out=np.zeros_like(mask, dtype=bool)
    for side in range(mask.shape[1]):
        ids=np.flatnonzero(mask[:,side])
        for run in np.split(ids, np.flatnonzero(np.diff(ids)>1)+1):
            if len(run)>=minimum:out[run,side]=True
    return out

def ankle_support(pos, quat, valid, dt=.02):
    # Meta positions are Y-up. Absolute floor offset is unknown; use the
    # low ankle-height plateau of this clip, independently for each foot.
    height=pos[:,:,1]
    floor=np.array([np.quantile(height[valid[:,s],s],.05) if valid[:,s].any() else np.nan for s in range(2)])
    velocity=np.diff(pos, axis=0, prepend=pos[:1])/dt
    speed=np.linalg.norm(velocity[:,:,[0,2]],axis=2)
    vz=np.abs(velocity[:,:,1])
    # No assumed local "forward" axis: obtain world angular velocity and
    # use its component along Meta's vertical Y axis.
    rot=Rotation.from_quat(quat[:,:,[1,2,3,0]].reshape(-1,4))
    mats=rot.as_matrix().reshape(-1,2,3,3)
    delta=mats[1:]@mats[:-1].transpose(0,1,3,2)
    omega=np.zeros_like(pos)
    omega[1:]=Rotation.from_matrix(delta.reshape(-1,3,3)).as_rotvec().reshape(-1,2,3)/dt
    good=valid.copy();good[1:] &= valid[:-1];good[0]=False
    candidate=good & (height-floor<=.025) & (speed<=.15) & (vz<=.15) & (abs(omega[:,:,1])<=np.pi/6)
    return {'support':retained_runs(candidate), 'ankle_floor_m':floor,
            'ankle_height_m':height-floor, 'ankle_horizontal_speed_m_s':speed,
            'ankle_vertical_speed_m_s':vz, 'ankle_yaw_rate_rad_s':omega[:,:,1]}

class Geometry:
    def __init__(self, model_path):
        self.model=mj.MjModel.from_xml_path(str(model_path));self.data=mj.MjData(self.model)
        m=self.model
        self.qadr=np.array([m.joint(n).qposadr[0] for n in JOINTS])
        self.dadr=np.array([m.joint(n).dofadr[0] for n in JOINTS])
        self.ranges=np.array([m.joint(n).range.copy() for n in JOINTS])
        self.bodyids=np.array([m.body(n).id for n in LINKS])
        self.feet=[m.body(s+'_ankle_roll_link').id for s in ('left','right')]
        floor=m.geom('floor').id
        if not np.allclose(m.geom_pos[floor],0):raise ValueError('Expected z=0 floor')
        self.pads=[]
        for body in self.feet:
            pads=[]
            for g in range(m.ngeom):
                if m.geom_bodyid[g]!=body:continue
                if not ((m.geom_contype[g]&m.geom_conaffinity[floor]) or (m.geom_contype[floor]&m.geom_conaffinity[g])):continue
                kind=m.geom_type[g];size=m.geom_size[g]
                if kind==mj.mjtGeom.mjGEOM_CAPSULE:points=[(np.array([0,0,z*size[1]]),size[0]) for z in (-1,1)]
                elif kind==mj.mjtGeom.mjGEOM_SPHERE:points=[(np.zeros(3),size[0])]
                elif kind==mj.mjtGeom.mjGEOM_BOX:points=[(np.array([x,y,z])*size,0.) for x in (-1,1) for y in (-1,1) for z in (-1,1)]
                else:raise ValueError('Unsupported foot geometry')
                pads.extend((g,p,r) for p,r in points)
            if not pads:raise ValueError('No floor-contact geometry')
            self.pads.append(pads)

    def set(self,q):
        self.data.qpos[:]=self.model.qpos0
        self.data.qpos[:7]=q[:7];self.data.qpos[self.qadr]=q[7:]
        mj.mj_kinematics(self.model,self.data)

    def clearance(self):
        d=self.data
        return np.array([min((d.geom_xpos[g]+d.geom_xmat[g].reshape(3,3)@p)[2]-r for g,p,r in pads) for pads in self.pads])

    def fk(self,q):
        p=[];r=[];h=[];toe=[]
        for row in q:
            self.set(row);d=self.data
            p.append(d.xpos[self.bodyids].copy());r.append(d.xquat[self.bodyids].copy());h.append(self.clearance())
            toe.append([d.xpos[b]+d.xmat[b].reshape(3,3)@np.array([.14,0,-.03]) for b in self.feet])
        return np.asarray(p),np.asarray(r),np.asarray(h),np.asarray(toe)

def angular_velocity(quat,dt):
    shape=quat.shape
    mats=Rotation.from_quat(quat[...,[1,2,3,0]].reshape(-1,4)).as_matrix().reshape(shape[:-1]+(3,3))
    delta=mats[1:]@mats[:-1].swapaxes(-2,-1)
    vel=Rotation.from_matrix(delta.reshape(-1,3,3)).as_rotvec().reshape(shape[0]-1,-1,3)/dt
    return np.concatenate([vel[:1],vel],axis=0)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',type=Path,required=True);ap.add_argument('--episode',type=Path,required=True);ap.add_argument('--model',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    manifest=json.loads((args.dataset/'manifest.json').read_text())
    for name,digest in manifest['source_sha256'].items():
        if sha(args.episode/name)!=digest:raise ValueError('Source hash mismatch: '+name)
    inp,ic=read_csv(args.episode/'input.csv');body,bc=read_csv(args.episode/'body.csv')
    if not np.array_equal(inp[:,ic['sequence']],body[:,bc['sequence']]):raise ValueError('Unmatched source sequences')
    wall=(inp[:,ic['receive_ns']]-inp[0,ic['receive_ns']])*1e-9
    geometry=Geometry(args.model)
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    report={'schema':'furry-dexweave-reference-v1','leg_source':'meta_virtual_legs','fps':50,
            'source_manifest_sha256':sha(args.dataset/'manifest.json'),'source_sha256':manifest['source_sha256'],
            'model_path':str(args.model.resolve()),'model_sha256':sha(args.model),'validation_stages':[2,9,14],
            'contact_method':{'source':'raw Meta ankle poses; no toe observation','stationary_speed_m_s':.15,'yaw_rate_rad_s':float(np.pi/6),'height_band_m':.025,'vertical_speed_m_s':.15,'min_run_s':.12,'confidence':None,'status':'geometric support proposal, not measured contact'},
            'retarget':'frozen continuous FURRY GMR + existing floor guard; canonical FK export, no new contact optimization',
            'clips':[]}
    for clip in manifest['clips']:
        src=args.dataset/clip['file']
        if sha(src)!=clip['sha256']:raise ValueError('Clip hash mismatch '+clip['id'])
        if (clip['stage'] in [2,9,14])!=(clip['split']=='validation'):raise ValueError('Split leakage')
        with np.load(src) as n:
            if int(n['fps'])!=100:raise ValueError('Expected 100 Hz original')
            q=n['qpos'][::2].astype(np.float64).copy()
        if not np.isfinite(q).all() or not np.allclose(np.linalg.norm(q[:,3:7],axis=1),1,atol=1e-5):raise ValueError('Invalid qpos')
        times=clip['source_start_wall_s']+np.arange(len(q))*.02
        ids=np.searchsorted(wall,times).clip(1,len(wall)-1)
        ids-=abs(wall[ids-1]-times)<abs(wall[ids]-times)
        poses=np.stack([body[ids][:,[bc[f'pose_{j}_{k}'] for k in range(7)]] for j in (6,7)],axis=1)
        flags=body[ids][:,[bc['flags_6'],bc['flags_7']]].astype(np.int64)
        valid=((flags&3)==3)&(body[ids,bc['valid'],None]==1)&(abs(wall[ids]-times)[:,None]<=.04)
        support=ankle_support(poses[:,:,:3],poses[:,:,3:],valid)
        p,r,h,toe=geometry.fk(q)
        dq=np.gradient(q[:,7:],.02,axis=0)
        violation=np.maximum(geometry.ranges[:,0]-q[:,7:],q[:,7:]-geometry.ranges[:,1]).clip(0)
        speed=np.linalg.norm(np.diff(toe[:,:,:2],axis=0),axis=2)/.02
        planted=support['support'][1:]&support['support'][:-1]
        item=dict(clip);item.update(frames=len(q),fps=50,source_npz_sha256=sha(src),
                seconds=(len(q)-1)*.02,joint_limit_violation_max_rad=float(violation.max()),
                floor_clearance_min_m=float(h.min()),support_fraction=support['support'].mean(0).tolist(),
                source_timing_error_max_s=float(abs(wall[ids]-times).max()),
                support_toe_speed_mean_m_s=float(speed[planted].mean()) if planted.any() else None,
                ready_kinematic=bool(violation.max()<1e-6 and h.min()>-1e-5))
        dst=out/clip['file'];dst.parent.mkdir(exist_ok=True)
        np.savez_compressed(dst,qpos=q,joint_pos=q[:,7:],joint_vel=dq,
            body_pos_w=p,body_quat_w=r,body_lin_vel_w=np.gradient(p,.02,axis=0),body_ang_vel_w=angular_velocity(r,.02),
            joint_names=np.array(JOINTS),body_names=np.array(LINKS),fps=np.array(50),
            source_sequence=inp[ids,ic['sequence']].astype(np.int64),
            source_measurement_ns=body[ids,bc['time_ns']].astype(np.int64),source_receive_ns=inp[ids,ic['receive_ns']].astype(np.int64),
            source_body_flags=flags,source_skeleton_version=body[ids,bc['skeleton_version']].astype(np.int64),
            source_ankle_pose=poses,source_valid=valid,floor_clearance=h,toe_pos_w=toe,**support)
        item['sha256']=sha(dst);report['clips'].append(item)
        print(clip['id'], 'ready' if item['ready_kinematic'] else 'FAILED',flush=True)
    report['pilot_ids']=[next(c['id'] for c in report['clips'] if c['split']=='train' and c['stage']==s and c['seconds']>=10) for s in (1,8,13)]
    report['ready_kinematic_count']=sum(c['ready_kinematic'] for c in report['clips'])
    report['counts']={s:sum(c['split']==s for c in report['clips']) for s in ('train','validation')}
    (out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
