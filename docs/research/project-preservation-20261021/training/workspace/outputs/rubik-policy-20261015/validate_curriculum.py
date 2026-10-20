import json,time
import numpy as np
from scipy.spatial.transform import Rotation as R
from curriculum_env import CurriculumEnv,STAGES,BOUNDS,ROOT,lib,VALID_MACROS
from native_env import TRANSLATION_STEP,ROTATION_STEP
start=time.monotonic();env=CurriculumEnv(seed=9)
print('BANK',json.dumps(env.bank_report),flush=True)
z=dict(np.load(ROOT.parent/'rubik-feedback-20261015/verified-nine.npz'))
results=[]
for stage in range(4):
 env.set_stage(stage)
 for evaluation,macro in [(False,0),(True,3)]:
  env.evaluation=evaluation
  offsets=STAGES[stage]['offsets']
  for offset in offsets:
   index=next(i for i,e in enumerate(env.bank) if e['macro']==macro and e['offset']==offset)
   obs,reset_info=env.reset(seed=123,options={'bank_index':index})
   entry=env.bank[index];target=env.target.copy()
   assert np.max(abs(env.s.q-z['qpos'][entry['frame']-1]))<1e-9
   # A warm state must target the already performed quarter, never another one.
   assert env.geometry.metrics(target)['corner_angle']<np.deg2rad(3)
   if offset=='return_start':assert not env.released
   outcome=None
   for k in range(entry['frame'],round(BOUNDS[macro+1]*100)+100,5):
    cc=z['commands'][min(k+4,round(BOUNDS[macro+1]*100)-1)]
    goal=cc[:14].reshape(2,7)
    dp=(goal[:,:3]-env.goals[:,:3])/TRANSLATION_STEP
    dr=(R.from_quat(goal[:,[4,5,6,3]])*R.from_quat(env.goals[:,[4,5,6,3]]).inv()).as_rotvec()/ROTATION_STEP
    action=np.r_[np.c_[dp,dr].ravel(),2*cc[-2:]-1]
    obs,reward,done,truncated,info=env.step(action)
    assert np.isfinite(obs).all() and np.isfinite(reward)
    if done or truncated:outcome=info;break
   result=dict(stage=stage,macro=macro,offset=offset,outcome=outcome)
   results.append(result);print('ORACLE',json.dumps(result),flush=True)
   if not outcome or not outcome['success']:raise RuntimeError('Warm oracle failed')
# Draw inverse-expanded, mixed-face sequences with deterministic seeds.
env.set_stage(8);_,a=env.reset(seed=88);sequence=env.sequence.copy()
_,b=env.reset(seed=88);assert sequence==env.sequence and len(a['logical_end_indices'])==10
logicalfaces=[sequence[end-1][0] for end in a['logical_end_indices']]
assert all(x!=y for x,y in zip(logicalfaces,logicalfaces[1:]))
assert len(set(logicalfaces))==3
# Cold reset is precisely the original airborne state, unaffected by warm starts.
assert np.max(abs(env.s.q-z['qpos'][4049]))<1e-9
result=dict(passed=True,bank=env.bank_report,oracle_cases=results,random_sequence=sequence,
            logical_faces=logicalfaces,wall_seconds=time.monotonic()-start,
            action_dim=14,no_finger_command_api=not hasattr(lib,'fingers'),
            native_warnings=int(lib.warnings(env.s.p)),policy_tested=False)
(ROOT/'curriculum-validation.json').write_text(json.dumps(result,indent=2))
env.close();print('VALIDATED',json.dumps({k:v for k,v in result.items() if k!='oracle_cases'}),flush=True)
