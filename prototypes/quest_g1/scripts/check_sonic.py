"""Native SONIC parity and packaged mode integration checks."""
from pathlib import Path
import os,sys,subprocess,json
import numpy as np,mujoco,onnxruntime
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from g1_sim.autonomous_legs import Sonic
out=Path(sys.argv[1]).resolve();out.mkdir(parents=True,exist_ok=True)
mj=Path(mujoco.__file__).parent;ort=Path(onnxruntime.__file__).parent/'capi'
env=dict(os.environ);env['DYLD_LIBRARY_PATH']=str(mj)+os.pathsep+str(ort)
binary=out/'check_sonic'
subprocess.run(['c++','-O1','-std=c++17','-pthread','-I'+str(ROOT/'android/native'),'-I'+str(mj/'include'),'-I'+str(ROOT/'vendor/onnxruntime-android/headers'),str(ROOT/'tests/check_sonic.cpp'),str(ROOT/'android/native/simulation.cpp'),str(next(mj.glob('libmujoco*.dylib'))),str(next(ort.glob('libonnxruntime*.dylib'))),'-o',str(binary)],check=True)
subprocess.run([str(binary),str(ROOT/'android/assets'),str(out)],env=env,check=True)
m=mujoco.MjModel.from_xml_path(str(ROOT/'android/assets/scene-stand.xml'));p=Sonic(m,ROOT/'android/assets',ROOT/'policies/sonic-vr3',ROOT/'experiments/homie_sonic/sonic_constants.json');d=mujoco.MjData(m);p.initialize(d)
cmd=np.r_[0,0,.793,0,0,0,p.home].astype(np.float32);cmd[24]=cmd[31]=1.2;ref=np.r_[0,0,.793,1,0,0,0,cmd[6:]].astype(float);rows=[]
for frame in range(100):
 if frame==50:cmd[0]=.3
 p.update(d,cmd,ref)
 for i in range(20):p.apply(d)
 rows.append(d.qpos.copy())
native=np.loadtxt(out/'native-sonic.csv',delimiter=',');error=float(np.abs(native-rows).max())
print('SONIC Python/native 100-step max qpos error:',error)
assert error<5e-5,error
(out/'native-checks.json').write_text(json.dumps(dict(sonic_native_100_steps_max_qpos_error=error,transactional_load=True,reset=True,mode_switches=True,recording=True,async_reset=True,virtual_leg_invariance=True,task_scenes=["stand","cup","push_t"]),indent=2)+'\n')
