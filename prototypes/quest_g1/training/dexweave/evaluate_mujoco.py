"""Matched 50 Hz body-policy/TWIST2 evaluation on the same frozen MuJoCo model.

The model, reference, seeds and horizon are identical. Each backend keeps
its own observation/PD contract. This measures a controller stack, not an
isolated Transformer-vs-MLP ablation. Five noisy first-frame starts, 10 s
maximum. Reference-end/time-limit completion is distinct from physical fall.
"""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import mujoco as mj
import onnxruntime as ort
from scipy.spatial.transform import Rotation
from .spec import JOINTS,LINKS
from .observations import build,process_action,matrices

DEFAULT=np.array([-.312,0,0,.669,-.363,0]*2+[0]*3+[.2,.2,0,.6,0,0,0]+[.2,-.2,0,.6,0,0,0])
ARMATURE=np.array([.010177520,.025101925,.010177520,.025101925,2*.003609725,2*.003609725]*2+
    [.010177520,2*.003609725,2*.003609725]+[.003609725]*5+[.00425]*2+[.003609725]*5+[.00425]*2)
KP=ARMATURE*(20*np.pi)**2;KD=4*ARMATURE*20*np.pi
EFFORT=np.array([88,139,88,139,50,50]*2+[88,50,50]+[25]*5+[5]*2+[25]*5+[5]*2)
SCALE=.25*EFFORT/KP

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

class Trial:
    def __init__(self,model_path,reference,policy,backend):
        self.m=mj.MjModel.from_binary_path(str(model_path));self.m.opt.timestep=.001
        self.d=mj.MjData(self.m);m=self.m
        self.qids=np.array([m.joint(n).qposadr[0] for n in JOINTS]);self.vids=np.array([m.joint(n).dofadr[0] for n in JOINTS]);self.aids=np.array([m.actuator(n).id for n in JOINTS])
        self.bids=np.array([m.body(n).id for n in LINKS]);self.floor=m.geom('floor').id
        self.footids=[m.body(s+'_ankle_roll_link').id for s in ('left','right')]
        with np.load(reference) as f:self.ref={k:f[k].copy() for k in f.files}
        assert self.ref['joint_names'].tolist()==list(JOINTS) and self.ref['body_names'].tolist()==list(LINKS) and int(self.ref['fps'])==50
        self.backend=backend
        if backend=='twist2':
            from .twist_controller import Controller
            self.controller=Controller(m,policy=policy,frequency=50)
        else:
            opts=ort.SessionOptions();opts.intra_op_num_threads=1;opts.inter_op_num_threads=1
            self.session=ort.InferenceSession(str(policy),sess_options=opts,providers=['CPUExecutionProvider'])
            if self.session.get_inputs()[0].shape[-1]!=157:raise ValueError('Wrong DexWeave input size')
        rootr=matrices(self.ref['qpos'][:,3:7]);v=np.gradient(self.ref['qpos'][:,:3],.02,axis=0)
        local=np.einsum('nji,nj->ni',rootr,v)
        rp=Rotation.from_matrix(rootr).as_euler('xyz')
        yawrate=np.gradient(np.unwrap(rp[:,2]),.02)
        self.commands=np.column_stack([local[:,:2],self.ref['qpos'][:,2],rp[:,:2],yawrate,self.ref['qpos'][:,7:]])

    def reset(self,seed):
        m,d=self.m,self.d;mj.mj_resetData(m,d);q=self.ref['qpos'];d.qpos[:7]=q[0,:7];d.qpos[self.qids]=q[0,7:]
        nextq=d.qpos.copy();nextq[:7]=q[1,:7];nextq[self.qids]=q[1,7:]
        mj.mj_differentiatePos(m,d.qvel,.02,d.qpos,nextq)
        if seed:d.qpos[self.qids]+=np.random.default_rng(seed).normal(0,.005,29)
        mj.mj_forward(m,d)
        self.previous_target=q[0,7:].copy();self.applied=np.zeros(29)
        if self.backend=='twist2':self.controller.reset()

    def observation(self,frame):
        d=self.d;r=self.ref
        omega=matrices(d.qpos[3:7])@d.qvel[3:6]
        return build(r['joint_pos'][frame],r['joint_vel'][frame],r['body_pos_w'][frame,7],r['body_quat_w'][frame,7],
                     d.xpos[self.bids],d.xquat[self.bids],d.qvel[:3],omega,d.qpos[self.qids],d.qvel[self.vids],DEFAULT,self.applied)[0]

    def metrics(self,frame):
        m,d=self.m,self.d;r=self.ref;p=d.xpos[self.bids];quat=d.xquat[self.bids]
        target=r['body_pos_w'][frame];rot=matrices(quat);rr=matrices(r['body_quat_w'][frame]);relative=rot[7]@rr[7].T
        yaw=np.arctan2(relative[1,0],relative[0,0]);align=Rotation.from_euler('z',yaw).as_matrix()
        anchor=p[7].copy();anchor[2]=target[7,2];aligned=(target-target[7])@align.T+anchor
        slip=[];force=np.zeros(6)
        for k in range(d.ncon):
            con=d.contact[k];g1,g2=int(con.geom1),int(con.geom2)
            other=g2 if g1==self.floor else g1 if g2==self.floor else None
            if other is None or m.geom_bodyid[other] not in self.footids:continue
            mj.mj_contactForce(m,d,k,force)
            if abs(force[0])<=1:continue
            jac=np.zeros((3,m.nv));mj.mj_jac(m,d,jac,None,con.pos,int(m.geom_bodyid[other]))
            slip.append(float(np.linalg.norm((jac@d.qvel)[:2])))
        height=np.abs(p[[3,6,7,10,13],2]-target[[3,6,7,10,13],2]).max()
        gravity=abs(rot[7,2,2]-rr[7,2,2])
        tilt=np.arccos(np.clip(rot[0,2,2],-1,1))
        return {'anchor_error_m':float(np.linalg.norm(p[7]-target[7])),
            'root_xy_error_m':float(np.linalg.norm(p[0,:2]-target[0,:2])),
            'body_aligned_error_m':float(np.linalg.norm(p-aligned,axis=1).mean()),
            'wrist_world_error_m':float(np.linalg.norm(p[[10,13]]-target[[10,13]],axis=1).mean()),
            'contact_slip_m_s':float(np.mean(slip)) if slip else None,
            'height_error_max_m':float(height),'gravity_vertical_error':float(gravity),
            'fell':bool(p[0,2]<.45 or tilt>1.2),'terminated':bool(height>.25 or gravity>.8),
            'pelvis_x':float(p[0,0]),'pelvis_y':float(p[0,1])}

    def run(self,seed,max_steps=500):
        self.reset(seed);rows=[];available=len(self.ref['qpos'])-1
        horizon=available if max_steps is None else min(max_steps,available)
        if horizon<1:raise ValueError('Evaluation requires at least two reference frames')
        for frame in range(horizon):
            if self.backend!='twist2':
                obs=self.observation(frame)
                action=self.session.run(None,{self.session.get_inputs()[0].name:obs[None]})[0][0]
                target,self.applied=process_action(action,self.previous_target,DEFAULT,SCALE)
                self.previous_target=target
            for _ in range(20):
                if self.backend=='twist2':self.controller.step(self.d,self.commands[frame],grip=0.)
                else:
                    self.d.ctrl[self.aids]=np.clip(KP*(target-self.d.qpos[self.qids])-KD*self.d.qvel[self.vids],-EFFORT,EFFORT)
                    mj.mj_step(self.m,self.d)
            if not np.isfinite(self.d.qpos).all():raise RuntimeError('Nonfinite simulation')
            # Refresh kinematics after mj_step before evaluating post-step pose.
            mj.mj_forward(self.m,self.d)
            row=self.metrics(frame+1);rows.append(row)
            if row['terminated']:break
        summary={'seed':seed,'steps':len(rows),'seconds':len(rows)*.02,'horizon_steps':horizon,
                 'completed':not rows[-1]['terminated'],'fell':any(r['fell'] for r in rows)}
        for name in ('anchor_error_m','root_xy_error_m','body_aligned_error_m','wrist_world_error_m','contact_slip_m_s'):
            vals=[r[name] for r in rows if r[name] is not None]
            summary[name]=float(np.mean(vals)) if vals else None
        actual=np.array([[r['pelvis_x'],r['pelvis_y']] for r in rows]);ref=self.ref['qpos'][:len(rows)+1,:2]
        summary['actual_path_m']=float(np.linalg.norm(np.diff(actual,axis=0),axis=1).sum())
        summary['reference_path_m']=float(np.linalg.norm(np.diff(ref,axis=0),axis=1).sum())
        return summary,rows

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--references',type=Path,required=True)
    p.add_argument('--policy',type=Path,required=True);p.add_argument('--backend',choices=['twist2','dexweave'],required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--clip-id',help='Evaluate only the matching reference for a per-reference policy')
    p.add_argument('--full-clip',action='store_true',help='Evaluate until the reference ends, instead of the default 10 seconds')
    args=p.parse_args();manifest=json.loads((args.references/'report.json').read_text())
    trials=[];traces={}
    report={'backend':args.backend,'policy_sha256':sha(args.policy),'model_sha256':sha(args.model),'mujoco_version':mj.__version__,
            'control_hz':50,'physics_hz':1000,'horizon_steps':None if args.full_clip else 500,'seeds':[0,1,2,3,4],
            'initialization':'reference frame zero with reference velocity; seeds1–4 joint noise std .005 rad; zero actor history',
            'trials':trials,'status':'running'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    for clip in manifest['clips']:
        if args.clip_id and clip['id']!=args.clip_id:continue
        env=Trial(args.model,args.references/clip['file'],args.policy,args.backend)
        for seed in range(5):
            result,rows=env.run(seed,max_steps=None if args.full_clip else 500);result.update(clip=clip['id'],reference_sha256=clip['sha256']);trials.append(result)
            traces[f"{clip['id']}__seed{seed}"]=rows;print(json.dumps(result),flush=True)
            args.output.write_text(json.dumps(report,indent=2)+'\n')
    if not trials:raise ValueError('No matching evaluation reference')
    report['status']='completed';report['completion_fraction']=sum(t['completed'] for t in trials)/len(trials);report['falls']=sum(t['fell'] for t in trials)
    report['tracking_failures']=sum(not t['completed'] for t in trials)
    args.output.write_text(json.dumps(report,indent=2)+'\n');args.output.with_suffix('.traces.json').write_text(json.dumps(traces)+'\n')

if __name__=='__main__':main()
