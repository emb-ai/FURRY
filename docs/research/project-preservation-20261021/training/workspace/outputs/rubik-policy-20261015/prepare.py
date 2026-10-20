"""One source chain, split by complete macro actions, never by adjacent frames."""
from pathlib import Path
import hashlib
import json
import numpy as np
from scipy.spatial.transform import Rotation as R
from native_env import Geometry, turn_target, mujoco, ASSETS, ROOT, TRANSLATION_STEP, ROTATION_STEP

source = ROOT.parent/'rubik-feedback-20261015/second-other-face.npz'
z = dict(np.load(source))
model = mujoco.MjModel.from_xml_path(str(ASSETS/'scene-rubik.xml'))
geometry = Geometry(model,z['qpos'][4049])
bounds = [40.5,59.32,78.17,93.02,117.01,131.90,148.16,164.62,180.98,197.29,278.31]
observations, actions, macro_ids, indices = [], [], [], []
raw_max = 0.
clipped = 0
segments = []
for macro, (start,end) in enumerate(zip(bounds,bounds[1:])):
    a,b=round(start*100),round(end*100)
    command=(3,-1) if macro<9 else (4,-1)
    target=turn_target(z['userdata'][a-1,8:64].reshape(8,7),*command)
    start_distance=np.linalg.norm(z['trace'][a-1,14:20].reshape(2,3)-z['trace'][a-1,2:5],axis=1)
    released=False
    release_ticks=np.zeros(2,dtype=int)
    count=0
    for k in range(a,b-4):
        ix=k-1
        old=z['commands'][ix,:14].reshape(2,7)
        grips=z['commands'][ix,-2:]
        masks=z['trace'][ix,12:14].astype(int)
        release_ticks=np.where((masks==0)&(grips<.1),release_ticks+1,0)
        dist=np.linalg.norm(z['trace'][ix,14:20].reshape(2,3)-z['trace'][ix,2:5],axis=1)
        released|=bool(np.any((release_ticks>=20)&(dist>start_distance+.02)))
        if (k-a)%5:continue
        geometry.update(z['qpos'][ix])
        observations.append(geometry.observation(z['qvel'][ix],old,grips,masks,target,command,released))
        next_goals=z['commands'][k+4,:14].reshape(2,7)
        dp=(next_goals[:,:3]-old[:,:3])/TRANSLATION_STEP
        dr=(R.from_quat(next_goals[:,[4,5,6,3]])*R.from_quat(old[:,[4,5,6,3]]).inv()).as_rotvec()/ROTATION_STEP
        action=np.r_[np.c_[dp,dr].ravel(),2*z['commands'][k+4,-2:]-1]
        raw_max=max(raw_max,float(abs(action).max()))
        clipped+=int(np.any(abs(action)>1.00001))
        actions.append(np.clip(action,-1,1));macro_ids.append(macro);indices.append(ix);count+=1
    segments.append(dict(macro=macro,start=start,end=end,command=command,samples=count,
                         complete_canonical_demonstration=macro<9,
                         split='validation' if macro in (3,7) else 'train'))
macro_ids=np.array(macro_ids)
train=~np.isin(macro_ids,[3,7])
np.savez_compressed(ROOT/'demonstrations.npz',observations=np.asarray(observations,dtype=np.float32),
                    actions=np.asarray(actions,dtype=np.float32),macro_ids=macro_ids,
                    source_indices=np.asarray(indices),train=train)
report=dict(source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            unique_source_chains=1,segments=segments,observation_shape=list(observations[0].shape),
            action_dim=14,control_hz=20,training_samples=int(train.sum()),validation_samples=int((~train).sum()),
            raw_action_absolute_max=raw_max,clipped_samples=clipped,
            limitations=['Correlated macro holdout from one deterministic simulation, not an independent test',
                         'Nine complete same-face cycles and one provisional adjacent-face macro',
                         'No demonstrations of all faces or all inverse commands',
                         '20 Hz interpolated actions need native rollout validation; saved source was 100 Hz'])
(ROOT/'dataset.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
