from pathlib import Path
import hashlib,json,sys,time
import numpy as np
import mujoco
from state import Encoder,SCHEMA,ACTION_DIM
O=Path(__file__).resolve().parent
B=O.parent/'cup-homie-finetune-20261015'
sys.path.insert(0,str(O.parent/'cup-dp-training-20261015'))
from capture_snapshot import inspect_episode,columns,digest

def main():
    names=json.loads((B/'quality-audit.json').read_text())['accepted']
    assert len(names)==59 and not (O/'dataset.npz').exists()
    # Fixed whole-episode holdout distributed across the recording session.
    valid=set(names[i] for i in [4,14,24,34,44,54])
    arrays={};rows=[];seen=set();family=None;started=time.time()
    for j,name in enumerate(names):
        e=inspect_episode(B/'recordings'/name)
        assert e['meta']['policy_id']=='homie-v2'
        assert e['identity'] not in seen;seen.add(e['identity'])
        if family is None:family=e['family']
        assert e['family']==family
        for key,file in [('policy_sha256','policy.onnx'),('policy_secondary_sha256','policy-secondary.onnx')]:
            assert digest(e['path']/file)==e['meta'][key]
        m=mujoco.MjModel.from_binary_path(str(e['path']/'learning-model.mjb'));d=mujoco.MjData(m);enc=Encoder(m)
        q=columns(e['frames'],'qpos',m.nq);v=columns(e['frames'],'qvel',m.nv)
        steps=np.rint(e['frames']['step']).astype(np.int64);ids=np.arange(e['start'],e['end']+1);ids=ids[steps[ids]%20==0];assert np.all(np.diff(steps[ids])==20)
        actions=e['actions'][ids,:ACTION_DIM].astype(np.float32);obs=[]
        for k,i in enumerate(ids):
            d.qpos[:]=q[i];d.qvel[:]=v[i]
            for key,n in [('mocap_pos',m.nmocap*3),('mocap_quat',m.nmocap*4)]:
                getattr(d,key).flat[:]=[e['scene'][f'{key}_{z}'][i] for z in range(n)]
            previous=actions[k-1] if k else np.zeros(ACTION_DIM,np.float32)
            obs.append(enc(d,previous))
        key=f'e{j:03d}';arrays[key+'_obs']=np.stack(obs);arrays[key+'_action']=actions
        arrays[key+'_time']=e['frames']['sim_time'][ids];arrays[key+'_source_index']=ids;arrays[key+'_source_step']=steps[ids]
        rows.append(dict(key=key,episode=name,identity=e['identity'],split='valid' if name in valid else 'train',frames=len(ids),duration_s=float(e['frames']['sim_time'][ids[-1]]-e['frames']['sim_time'][ids[0]]),root_path_m=float(np.linalg.norm(np.diff(q[ids,:2],axis=0),axis=1).sum())))
        print(j+1,name,len(ids),flush=True)
    np.savez_compressed(O/'dataset.npz',**arrays)
    report=dict(schema=SCHEMA,family=family,episodes=rows,observation_names=enc.names,action_dim=ACTION_DIM,fps=50,
                source=str(B/'recordings'),train=53,valid=6,frames=sum(x['frames'] for x in rows),
                independent_session=False,observation='pre-integration robot and object state; previous sampled command only; no task outcomes/timers/human skeleton/images',
                source_timing='native Frame/Mimic captured before the next 10 physics steps; exact native control phase: step modulo20=0; 50Hz',
                sha256=digest(O/'dataset.npz'),seconds=time.time()-started)
    (O/'dataset.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:report[k] for k in ['train','valid','frames','sha256','seconds']}),flush=True)
if __name__=='__main__':main()
