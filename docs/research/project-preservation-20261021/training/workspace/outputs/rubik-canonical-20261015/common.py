from pathlib import Path
import ctypes,subprocess,sys,os,json
import numpy as np,mujoco,onnxruntime
O=Path(__file__).resolve().parent;H=O;P=Path('/Users/vlad/Documents/ChatGPT/VR-телеуправление гуманоидом/outputs.nosync/rubik-sequences-20261015/project');ASSETS=Path(os.environ.get('SCENE_ASSETS',str(P/'android/assets')));sys.path.insert(0,str(P))
from g1_sim.controller import JOINTS
mj=Path(mujoco.__file__).parent;ort=Path(onnxruntime.__file__).parent/'capi'
if '--build' in sys.argv:
 subprocess.run(['c++','-O2','-std=c++17','-dynamiclib','-I'+str(P/'android/native'),'-I'+str(mj/'include'),'-I'+str(P/'vendor/onnxruntime-android/headers'),str(O/'bridge.cpp'),str(P/'android/native/simulation.cpp'),str(next(mj.glob('libmujoco*.dylib'))),str(next(ort.glob('libonnxruntime*.dylib'))),'-Wl,-rpath,'+str(ort),'-o',str(Path('/Users/vlad/Documents/ChatGPT/VR-телеуправление гуманоидом/outputs.nosync/rubik-sequences-20261015')/os.environ.get('BRIDGE','bridge-home.dylib'))],check=True)
lib=ctypes.CDLL(str(Path('/Users/vlad/Documents/ChatGPT/VR-телеуправление гуманоидом/outputs.nosync/rubik-sequences-20261015')/os.environ.get('BRIDGE','bridge-home.dylib')));ptr=ctypes.c_void_p;fp=np.ctypeslib.ndpointer(np.float32,flags='C_CONTIGUOUS');dp=np.ctypeslib.ndpointer(np.float64,flags='C_CONTIGUOUS')
for name,args,ret in [('create_policy',[ctypes.c_char_p]*3,ptr),('destroy',[ptr],None),('state',[ptr,dp],ctypes.c_int),('step',[ptr,fp,ctypes.c_int],ctypes.c_int),('step_grip',[ptr,fp,ctypes.c_int,ctypes.c_double,ctypes.c_double],ctypes.c_int),('correct',[ptr,fp,dp,fp],None),('seed',[ptr,ctypes.c_uint],None),('task',[ptr,dp],None),('warnings',[ptr],ctypes.c_int),('supports',[ptr],ctypes.c_int),('rubik_contacts',[ptr,ctypes.c_int],ctypes.c_int),('tune',[ptr,ctypes.c_int,ctypes.c_double],None),('velocity',[ptr,dp],None),('contact_min',[ptr,dp],None)]:
 f=getattr(lib,name);f.argtypes=args;f.restype=ret
class Sim:
 def __init__(self,scene,policy='homie-standing',seed=0):
  self.m=mujoco.MjModel.from_xml_path(str(ASSETS/f'scene-{scene}.xml'));self.d=mujoco.MjData(self.m);self.p=lib.create_policy(str(ASSETS).encode(),scene.encode(),policy.encode());assert self.p
  lib.seed(self.p,seed);lib.tune(self.p,int(os.environ.get("NOSLIP",str(self.m.opt.noslip_iterations))),float(os.environ.get("FORCE","0")));self.qa=np.array([int(self.m.joint(j).qposadr[0]) for j in JOINTS]);self.wr=[self.m.body(s+'_wrist_yaw_link').id for s in ['left','right']];self.q=np.zeros(self.m.nq);self.read();self.cmd=np.r_[0,0,.74,0,0,0,self.q[self.qa]].astype(np.float32)
 def read(self):
  lib.state(self.p,self.q);self.d.qpos[:]=self.q;mujoco.mj_kinematics(self.m,self.d)
 def poses(self):return np.array([np.r_[self.d.xpos[b],self.d.xquat[b]] for b in self.wr])
 def step(self,goals=None,grips=(0,0),n=10):
  cmd=self.cmd.copy()
  if goals is not None:lib.correct(self.p,self.cmd,np.ascontiguousarray(goals.ravel()),cmd)
  self.applied=cmd.copy();fail=lib.step_grip(self.p,cmd,n,*grips);self.read();return bool(fail)
 def state(self):
  state=np.zeros(self.m.nuserdata);lib.task(self.p,state);return state
 def close(self):lib.destroy(self.p)
