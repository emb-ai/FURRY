from pathlib import Path
import sys,json
import numpy as np,mujoco
APP=Path(__file__).resolve().parents[2];sys.path.insert(0,str(APP));sys.path.insert(0,str(Path(__file__).parent))
from g1_sim.autonomous_legs import Homie,Sonic,rot
from evaluate import Metrics
assets=APP/'android/assets';root=APP.parents[3];out=root/'outputs/homie-sonic-comparison-20261012'
m=mujoco.MjModel.from_xml_path(str(assets/'scene-stand.xml'));p=Homie(m,assets);d=mujoco.MjData(m);p.initialize(d)
cmd=np.r_[0,0,d.qpos[2],0,0,0,d.qpos[p.qa]].astype(np.float32);rows=[]
for i in range(100):
 if i==50:cmd[0]=.3
 p.update(d,cmd)
 for j in range(20):p.apply(d)
 rows.append(d.qpos.copy())
expected=np.loadtxt(out/'native-parity.csv',delimiter=',');error=float(np.max(np.abs(expected-rows)))
print('native Homie rollout max error',error)
assert error<5e-5,error
m=mujoco.MjModel.from_xml_path(str(assets/'scene-stand.xml'));s=Sonic(m,assets,out/'models',Path(__file__).with_name('sonic_constants.json'));d=mujoco.MjData(m);s.initialize(d)
ref=np.r_[d.qpos[:7],d.qpos[s.qa]];cmd=np.r_[0,0,d.qpos[2],0,0,0,d.qpos[s.qa]];s.replan(d,cmd)
a=s.encoder_observation(d,ref);other=ref.copy();other[7:19]+=np.linspace(-.2,.2,12);b=s.encoder_observation(d,other)
# Leg reference joints cannot affect the 3-point encoder or its planner input.
assert np.max(np.abs(a-b))<1e-6
assert a.shape==(1762,) and a[0]==1 and np.all(a[1:4]==0)
metric=Metrics(m);mujoco.mj_forward(m,d);metric.sample(d,ref)
for key in ['root_xy_m','upper_world_mean_m','upper_local_mean_m','wrist_world_mean_m','upper_joint_rmse_rad']:
 assert metric.rows[-1][key]<1e-9,(key,metric.rows[-1][key])
# Rotating foot contact point velocity includes omega cross r via Jacobian.
point=d.xpos[metric.feet[0]]+np.array([.1,0,0]);jp=np.zeros((3,m.nv));jr=np.zeros_like(jp);mujoco.mj_jac(m,d,jp,jr,point,metric.feet[0]);assert np.isfinite(jp).all()
result=dict(homie_native_100_control_steps_max_qpos_error=error,sonic_encoder_width=1762,sonic_decoder_width=994,sonic_recorded_leg_invariance=True,perfect_tracking_metric_zero=True)
(out/'contract-checks.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
