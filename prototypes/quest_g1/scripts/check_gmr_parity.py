"""Differential test of native GMR against unmodified pinned Python GMR/Mink."""
from pathlib import Path
import importlib.util,sys,types,json,subprocess,os
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R
ROOT=Path(__file__).resolve().parents[1];out=ROOT/'outputs/gmr-parity';out.mkdir(parents=True,exist_ok=True)
# Import only the upstream solver; optional viewer/streamer packages aren't needed.
package=types.ModuleType('gmr_oracle_upstream');package.__path__=[str(ROOT/'vendor/GMR/general_motion_retargeting')];sys.modules[package.__name__]=package
spec=importlib.util.spec_from_file_location(package.__name__+'.motion_retarget',Path(package.__path__[0])/'motion_retarget.py');mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
oracle_env=dict(os.environ)
if sys.platform=='darwin':
 oracle_env['DYLD_LIBRARY_PATH']=str(Path(mujoco.__file__).parent)+os.pathsep+oracle_env.get('DYLD_LIBRARY_PATH','')
config=json.loads((ROOT/'android/config/gmr_xrobot_g1.json').read_text());names=list(config['human_scale_table']);entries={v[0]:(k,v) for k,v in config['ik_match_table1'].items()}
for height in (1.30,1.75,1.95):
 gmr=mod.GeneralMotionRetargeting('xrobot','unitree_g1',actual_human_height=height,verbose=False)
 m=gmr.model;d=mujoco.MjData(m);all_inputs=[];expected=[]
 rng=np.random.default_rng(7)
 for n in range(70):
  mujoco.mj_resetData(m,d);d.qpos[2]=.8
  # Smooth bilateral arm/leg poses, root translation/rotation, joint-limit cases.
  u=n/69.;d.qpos[:2]=[.3*u,.1*np.sin(3*u)]
  d.qpos[3:7]=R.from_euler('xyz',[.1*np.sin(4*u),.15*np.sin(3*u),3.13*u]).as_quat(scalar_first=True)
  for j in range(1,m.njnt):
   a=m.jnt_qposadr[j];lo,hi=m.jnt_range[j]
   d.qpos[a]=np.clip(.4*np.sin(5*u+j*.4),lo+.001,hi-.001)
  mujoco.mj_forward(m,d)
  human={};root=d.body('pelvis').xpos.copy();human_root=root/(config['human_scale_table']['Pelvis']*height/config['human_height_assumption'])
  for name in names:
   body,entry=entries[name];b=d.body(body);robotrot=R.from_matrix(b.xmat.reshape(3,3));rawrot=robotrot*R.from_quat(entry[4],scalar_first=True).inv()
   rawpos=(b.xpos-robotrot.apply(entry[3])-root)/(config['human_scale_table'][name]*height/config['human_height_assumption'])+human_root
   human[name]=[rawpos,rawrot.as_quat(scalar_first=True)]
  all_inputs.append(np.concatenate([np.r_[human[n][0],human[n][1]] for n in names]))
  expected.append(gmr.retarget(human,offset_to_ground=True))
 input_path=out/f'input-{height}.txt';native_path=out/f'native-{height}.csv'
 np.savetxt(input_path,all_inputs,header=f'{len(all_inputs)} {len(names)} {height}',comments='')
 np.savetxt(out/f'python-{height}.csv',expected,delimiter=',')
 subprocess.run([str(ROOT/'android/build/gmr-oracle'),str(ROOT/'android/assets'),str(input_path),str(native_path)],check=True,env=oracle_env)
 native=np.loadtxt(native_path,delimiter=',');errors=[]
 for a,b in zip(expected,native):
  dq=np.empty(m.nv);mujoco.mj_differentiatePos(m,dq,1,a,b);errors.append(np.max(abs(dq)))
 print(f'height={height} frames={len(errors)} max_tangent_difference={max(errors):.9g}',flush=True)
 assert max(errors)<2e-4, max(errors)
