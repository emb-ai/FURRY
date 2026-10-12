"""Native source-patch export and paired fresh physical rollouts on a Quest episode."""
from pathlib import Path
import sys,os,subprocess,json,hashlib
import numpy as np,mujoco,onnxruntime
APP=Path(__file__).resolve().parents[2];sys.path.insert(0,str(APP));sys.path.insert(0,str(APP/'experiments/homie_sonic'))
from evaluate import run
from argparse import Namespace
out=Path(sys.argv[1]).resolve();episode=Path(sys.argv[2]).resolve();out.mkdir(parents=True,exist_ok=True)
mj=Path(mujoco.__file__).parent;ort=Path(onnxruntime.__file__).parent/'capi';env=dict(os.environ,DYLD_LIBRARY_PATH=str(mj)+os.pathsep+str(ort))
flags=['c++','-O2','-std=c++17','-pthread','-I'+str(APP/'android/native'),'-I'+str(mj/'include'),'-I'+str(APP/'vendor/onnxruntime-android/headers')]
libs=[str(next(mj.glob('libmujoco*.dylib'))),str(next(ort.glob('libonnxruntime*.dylib')))]
subprocess.run(flags+[str(APP/'tests/check_virtual_leg_lift.cpp')]+libs+['-o',str(out/'check-patch')],check=True)
subprocess.run([str(out/'check-patch')],check=True,env=env)
subprocess.run(flags+[str(Path(__file__).with_name('export.cpp'))]+[str(APP/'android/native'/f) for f in ['simulation.cpp','meta_retarget.cpp','gmr.cpp']]+libs+['-o',str(out/'export')],check=True)
inputs=[]
for variant in ['off','on']:
 path=out/f'{variant}.csv'
 subprocess.run([str(out/'export'),str(APP/'android/assets'),str(episode),str(path),variant],check=True,env=env)
 a=np.loadtxt(path,delimiter=',',skiprows=1);assert np.max(np.abs(np.diff(a[:,0])-.01))<1e-7
 inputs.append(a)
# Headers: 8 diagnostics, 35 cmd, 36 reference, 7 camera, 98 raw, 98 patched.
# Keep every input frame after the first second; same activation/initialization.
for variant,a in zip(['off','on'],inputs):
 a=a[100:];common=inputs[0][100:];q=a[:,43:79];goal=common[:,79:86];from scipy.spatial.transform import Rotation
 mat=Rotation.from_quat(goal[:,3:][:,[1,2,3,0]]).as_matrix();yaw=np.arctan2(-mat[:,1,2],-mat[:,0,2]);camera=np.c_[goal[:,:2],yaw]
 np.savez_compressed(out/f'{variant}.npz',qpos=q,fps=100.,commands=a[:,8:43],camera_goal=camera,source=a)
 args=Namespace(assets=APP/'android/assets',models=None,seconds=1e9,seed=11,twist_policy='policy-quest-111586-500.onnx',no_catchup=False,output=out/'rollouts')
 run(args,out/f'{variant}.npz','twist2')
proof=dict(episode=episode.name,policy='quest-111586-500',policy_hz=50,physics_hz=1000,catch_up=True,trimmed_start_seconds=1,source_hashes={})
for f in ['body.csv','input.csv','frames.csv','manifest.json']:
 with (episode/f).open('rb') as stream:proof['source_hashes'][f]=hashlib.file_digest(stream,'sha256').hexdigest()
(out/'provenance.json').write_text(json.dumps(proof,indent=2)+'\n')
