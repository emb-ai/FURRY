"""Desktop collision microbenchmark + policy checks; not a Quest FPS measurement."""
import argparse
import json
from pathlib import Path
import sys
import time
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from g1_sim.scene import make_model
from g1_sim.collisions import primitive_collisions
from g1_sim.controller import Controller, standing_command, JOINTS


def run(clip_path=None):
    original, xml = make_model('empty')
    proxy_xml, report = primitive_collisions(xml)
    proxy = mujoco.MjModel.from_xml_string(proxy_xml)
    result = dict(collision_model=report, timing_platform='desktop CPU, not Quest', models={})
    # Same deterministic poses for both models, including closed fingers and
    # knees folded near the torso. No policy/GMR overhead in these timings.
    rng = np.random.default_rng(42)
    poses = []
    for _ in range(160):
        q = original.qpos0.copy()
        q[2] = rng.uniform(.4, .8)
        for side, sign in [('left',1),('right',-1)]:
            for part, value in [('hip_pitch',rng.uniform(-1.6,-.2)),('knee',rng.uniform(.4,2.3)),
                                ('ankle_pitch',rng.uniform(-.8,.2)),('shoulder_pitch',rng.uniform(-1.4,.3)),
                                ('shoulder_roll',sign*rng.uniform(-.2,.4)),('elbow',rng.uniform(1.2,2.0))]:
                q[original.joint(f'{side}_{part}_joint').qposadr[0]] = value
        for j in range(original.njnt):
            if 'hand_' in original.joint(j).name:
                lo,hi=original.jnt_range[j];q[original.jnt_qposadr[j]]=rng.choice([lo,hi])
        poses.append(q)
    walk = None
    if clip_path:
        clip=np.load(clip_path);fps=float(clip['fps']);rot=Rotation.from_quat(clip['root_rot']);p=clip['root_pos']
        v=rot.inv().apply(np.gradient(p,1/fps,axis=0));w=np.zeros_like(v);w[1:]=(rot[:-1].inv()*rot[1:]).as_rotvec()*fps
        walk=np.column_stack([v[:,:2],p[:,2],rot.as_euler('xyz')[:,:2],w[:,2],clip['dof_pos']])
    for name, model in [('mesh', original), ('primitive', proxy)]:
        d=mujoco.MjData(model);cost=[];counts=[];warnings=0
        for q in poses:
            d.qpos[:]=q;d.qvel[:]=0
            for _ in range(3):
                start=time.perf_counter();mujoco.mj_forward(model,d);cost.append((time.perf_counter()-start)*1000)
            counts.append(d.ncon)
        r=dict(forward_ms_p50=float(np.percentile(cost,50)),forward_ms_p95=float(np.percentile(cost,95)),
               forward_ms_max=max(cost),max_contacts=max(counts),warnings=sum(w.number for w in d.warning),policy={})
        modes=['stand','squat','fold_arms']+(['walk'] if walk is not None else [])
        ctrl=Controller(model)
        for mode in modes:
            ctrl.initialize(d);minz=1.;max_tilt=0.;max_contacts=0;distance=0.;xy=d.qpos[:2].copy();start=time.perf_counter()
            for n in range(1200):
                t=n*.01;cmd=standing_command();u=np.clip((t-2)/3,0,1)
                if mode=='squat':
                    cmd[2]=.8-.28*u
                    for offset in [6,12]:cmd[offset:offset+6]=[-.2-.7*u,0,0,.4+1.4*u,-.2-.7*u,0]
                if mode=='fold_arms':
                    cmd[21]=-1.1*u;cmd[22]=.4-.55*u;cmd[24]=1.2+.7*u
                    cmd[28]=-1.1*u;cmd[29]=-.4+.55*u;cmd[31]=1.2+.7*u
                if mode=='walk':
                    index=np.clip((t-2)*fps,0,len(walk)-1);a=int(index);b=min(a+1,len(walk)-1)
                    cmd=walk[a]*(1-index+a)+walk[b]*(index-a)
                    if t<2:cmd=standing_command()*(1-t/2)+cmd*t/2
                for _ in range(10):ctrl.step(d,cmd,grip=float(mode=='fold_arms'))
                minz=min(minz,float(d.qpos[2]));max_tilt=max(max_tilt,float(np.arccos(np.clip(d.xmat[model.body('pelvis').id,8],-1,1))))
                max_contacts=max(max_contacts,d.ncon);distance+=float(np.linalg.norm(d.qpos[:2]-xy));xy=d.qpos[:2].copy()
                if d.qpos[2]<.35:break
            r['policy'][mode]=dict(sim_seconds=(n+1)*.01,wall_seconds=time.perf_counter()-start,min_height=minz,
                max_tilt_deg=float(np.rad2deg(max_tilt)),max_contacts=max_contacts,path_m=distance,final_xy=xy.tolist(),fell=minz<.35,
                warnings=sum(w.number for w in d.warning))
            print(name,mode,r['policy'][mode],flush=True)
        result['models'][name]=r
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--walk-clip');parser.add_argument('--output',default='outputs/collision-check.json');args=parser.parse_args()
    result=run(args.walk_clip);path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(result,indent=2)+'\n')
