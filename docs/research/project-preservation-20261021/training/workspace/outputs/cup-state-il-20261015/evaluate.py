"""Autonomy from complete native reset; never restore demonstrated robot states."""
from pathlib import Path
import argparse,ctypes,hashlib,json,time
from collections import deque
import numpy as np
import mujoco
import torch
from state import Encoder,ACTION_DIM,HISTORY,EXECUTE
from models import make

O=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

class World:
    def __init__(self):
        suffix='dylib' if __import__('sys').platform=='darwin' else 'so'
        self.lib=ctypes.CDLL(str(O/('bridge.'+suffix)));l=self.lib
        ptr=ctypes.c_void_p;dp=np.ctypeslib.ndpointer(np.float64,flags='C_CONTIGUOUS');fp=np.ctypeslib.ndpointer(np.float32,flags='C_CONTIGUOUS')
        l.create.argtypes=[ctypes.c_char_p];l.create.restype=ptr;l.destroy.argtypes=[ptr];l.last_error.restype=ctypes.c_char_p
        l.reset_random.argtypes=[ptr,ctypes.c_uint];l.fresh_reset_flags.argtypes=[ptr];l.is_standing.argtypes=[ptr]
        l.read_state.argtypes=[ptr,dp,dp,dp,dp,dp];l.step.argtypes=[ptr,fp,ctypes.c_int];l.contacts.argtypes=[ptr]
        p=O/'evaluation-episode';meta=json.loads((O/'dataset.json').read_text());self.meta=meta
        assert sha(p/'learning-model.mjb')==meta['family']['model_sha256']
        assert sha(p/'policy.onnx')==meta['family']['policy_sha256'];assert sha(p/'policy-secondary.onnx')==meta['family']['policy_secondary_sha256']
        self.m=mujoco.MjModel.from_binary_path(str(p/'learning-model.mjb'));self.d=mujoco.MjData(self.m);self.enc=Encoder(self.m)
        self.p=l.create(str(p).encode());assert self.p,l.last_error().decode();assert l.is_standing(self.p)==0
        self.q=np.zeros(self.m.nq);self.v=np.zeros(self.m.nv);self.u=np.zeros(self.m.nuserdata);self.mp=np.zeros(self.m.nmocap*3);self.mq=np.zeros(self.m.nmocap*4)
        self.cq=int(self.m.joint('cup_free').qposadr[0])
    def read(self):
        self.lib.read_state(self.p,self.q,self.v,self.u,self.mp,self.mq)
        self.d.qpos[:]=self.q;self.d.qvel[:]=self.v;self.d.userdata[:]=self.u;self.d.mocap_pos.flat[:]=self.mp;self.d.mocap_quat.flat[:]=self.mq
        mujoco.mj_forward(self.m,self.d)
    def reset(self,seed):
        assert self.lib.reset_random(self.p,seed)==0;assert self.lib.fresh_reset_flags(self.p)==63;self.read()
    def close(self):self.lib.destroy(self.p)

def run(checkpoint,out,device='cpu',count=10,limit=90):
    out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False);stats=ck['stats'];model=make(ck['kind'],len(stats['obs_mean']));model.load_state_dict(ck['ema']);model.eval().to(device)
    meta=json.loads((O/'dataset.json').read_text());assert ck['metadata']['dataset_sha256']==meta['sha256']
    cases=json.loads((O/'evaluation-scenes.json').read_text())['seeds'][:count];rows=[]
    for seed in cases:
        torch.manual_seed(seed);w=World();w.reset(seed);start=time.monotonic();previous=np.zeros(ACTION_DIM,np.float32);history=deque(maxlen=HISTORY)
        first=w.enc(w.d,previous);history.extend([first.copy() for _ in range(HISTORY)])
        initial={k:getattr(w,k).copy() for k in ['q','v','u','mp','mq']};cup0=w.q[w.cq:w.cq+3].copy();states=[];vels=[];actions=[];trace=[];infer=[]
        maxlift=0.;maxtilt=0.;rootpath=0.;lastroot=w.q[:2].copy();minheight=w.q[2];outcome=0;error=None
        for k in range(round(limit*50)):
            if k%EXECUTE==0:
                x=(np.stack(history)-stats['obs_mean'])/stats['obs_std'];t=time.monotonic()
                with torch.inference_mode():chunk=model.predict(torch.from_numpy(x)[None].to(device)).cpu().numpy()[0]*stats['action_std']+stats['action_mean']
                infer.append(time.monotonic()-t)
            a=np.ascontiguousarray(chunk[k%EXECUTE],np.float32)
            if not np.isfinite(a).all():error='Nonfinite policy action';outcome=-1;break
            a[35:37]=np.clip(a[35:37],0,1)
            n=20
            outcome=w.lib.step(w.p,a,n);w.read();previous=a.copy()
            if not np.isfinite(w.q).all() or not np.isfinite(w.v).all():
                error=w.lib.last_error().decode() or 'Nonfinite simulator state';outcome=-1;break
            history.append(w.enc(w.d,previous))
            cup=w.q[w.cq:w.cq+3];quat=w.q[w.cq+3:w.cq+7];tilt=float(np.degrees(np.arccos(np.clip(1-2*(quat[1]**2+quat[2]**2),-1,1))))
            maxlift=max(maxlift,float(cup[2]-cup0[2]));maxtilt=max(maxtilt,tilt);rootpath+=float(np.linalg.norm(w.q[:2]-lastroot));lastroot=w.q[:2].copy();minheight=min(minheight,w.q[2])
            goal=w.d.xpos[w.enc.goal];err=float(np.linalg.norm(cup[:2]-goal[:2]))
            states.append(w.q.copy());vels.append(w.v.copy());actions.append(a.copy());trace.append([(k+1)/50,*cup,err,tilt,w.lib.contacts(w.p),*w.u])
            if outcome:
                if outcome<0:error=w.lib.last_error().decode()
                break
        row={'seed':seed,'success':outcome==1,'outcome':outcome,'sim_s':(k+1)/50,'wall_s':time.monotonic()-start,'lift_m':maxlift,'max_tilt_deg':maxtilt,'min_root_z':float(minheight),'root_path_m':rootpath,'goal_xy_error_m':err if states else None,'inference_mean_s':float(np.mean(infer)),'error':error}
        np.savez_compressed(out/f'{seed}.npz',qpos=states,qvel=vels,actions=actions,trace=trace,**{'initial_'+k:v for k,v in initial.items()})
        rows.append(row);(out/'results.json').write_text(json.dumps(rows,indent=2));print(json.dumps(row),flush=True);w.close()
    summary={'status':'completed','kind':ck['kind'],'checkpoint_step':ck['step'],'checkpoint_sha256':sha(checkpoint),'successes':sum(r['success'] for r in rows),'trials':len(rows),'results':rows,
             'initialization':'Native NewSceneTrial(seed), complete reset/history=0/grips=0/velocities=0; first predicted action from first step; no demonstration pose/prefix',
             'state_input':'Exact MuJoCo numeric states; no images','execution':EXECUTE,'fps':50,'limit_s':limit,'latency':'Simulation pauses during inference; not live Quest timing','single_training_seed':True}
    (out/'summary.json').write_text(json.dumps(summary,indent=2));return summary

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--device',default='cpu');p.add_argument('--count',type=int,default=10);p.add_argument('--limit',type=float,default=90);a=p.parse_args();run(a.checkpoint,a.out,a.device,a.count,a.limit)
