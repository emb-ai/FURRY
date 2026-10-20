"""Check physical goal semantics and the resampled native action interface."""
import json
import time
import hashlib
import numpy as np
from scipy.spatial.transform import Rotation as R
from native_env import ROOT, Geometry, turn_target, mujoco, ASSETS, NativeRubikEnv, TRANSLATION_STEP, ROTATION_STEP, lib
source_sha256=hashlib.sha256((ROOT/'native_env.py').read_bytes()).hexdigest()

z=dict(np.load(ROOT.parent/'rubik-feedback-20261015/verified-nine.npz'))
m=mujoco.MjModel.from_xml_path(str(ASSETS/'scene-rubik.xml'))
g=Geometry(m,z['qpos'][4049])
logical=z['userdata'][4049,8:64].reshape(8,7)
target=turn_target(logical,3,-1)
before=g.metrics(target)
assert before['corner_angle']>1
g.update(z['qpos'][5931]);after=g.metrics(target)
assert after['corner_position']<.001 and after['corner_angle']<np.deg2rad(3)
wrong=g.metrics(turn_target(logical,4,-1))
assert wrong['corner_angle']>1
assert np.max(abs(turn_target(turn_target(logical,3,-1),3,1)[:,:3]-logical[:,:3]))<1e-12
whole=dict(np.load(ROOT.parent/'rubik-feedback-20261015/whole-x.npz'))
q=whole['qpos'][19728];g.update(q)
whole_goal=turn_target(whole['userdata'][19728,8:64].reshape(8,7),4,-1)
g.update(whole['qpos'][20328]);whole_only=g.metrics(whole_goal)
assert whole_only['corner_angle']>1, 'Whole rotation alone must not count as a face turn'
start=time.monotonic()
env=NativeRubikEnv(sequence_length=2)
obs,info=env.reset(seed=22,options={'sequence':[(3,-1),(4,-1)]})
initial_q=env.s.q.copy();completed=0;fail=None;info={}
for k in range(4050,5930,5):
    goal=z['commands'][k+4,:14].reshape(2,7)
    dp=(goal[:,:3]-env.goals[:,:3])/TRANSLATION_STEP
    dr=(R.from_quat(goal[:,[4,5,6,3]])*R.from_quat(env.goals[:,[4,5,6,3]]).inv()).as_rotvec()/ROTATION_STEP
    action=np.r_[np.c_[dp,dr].ravel(),2*z['commands'][k+4,-2:]-1]
    obs,reward,done,truncated,info=env.step(action)
    assert np.all(np.isfinite(obs)) and np.isfinite(reward)
    completed=info['completed_moves']
    if completed:
        assert env.command==(4,-1)
        assert np.max(abs(env.s.q-initial_q))>.01, 'Physical state must not reset between moves'
        break
    if done or truncated:fail=info;break
result=dict(geometry_tests_passed=True,whole_cube_only_rejected=True,wrong_face_rejected=True,
            resampled_oracle_completed_moves=completed,oracle_info=info,failure=fail,
            native_warnings=int(lib.warnings(env.s.p)),wall_seconds=time.monotonic()-start,
            action_dim=14,observation_dim=len(obs),native_resets_between_commands=0,
            policy_tested=False,native_env_sha256=source_sha256,
            translation_step_m=TRANSLATION_STEP,rotation_step_rad=ROTATION_STEP)
(ROOT/'environment-validation.json').write_text(json.dumps(result,indent=2))
env.close();print(json.dumps(result,indent=2))
if completed!=1:raise SystemExit('20 Hz resampled oracle failed; do not assume demonstration parity')
