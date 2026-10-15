"""Regression tests for floor-placement crouching and passive cup stability."""
from pathlib import Path
import os,subprocess,tempfile,unittest
import mujoco,numpy as np
from g1_sim.scene import make_model
ROOT=Path(__file__).resolve().parents[1]
class WBDexTests(unittest.TestCase):
 def test_crouch_and_fall_semantics(self):
  mj=Path(mujoco.__file__).parent
  with tempfile.TemporaryDirectory() as temp:
   binary=Path(temp)/'wb-test'
   subprocess.run(['c++','-std=c++17','-O2','-Wall','-Wextra','-Werror','-I'+str(ROOT/'android/native'),'-I'+str(mj/'include'),str(ROOT/'tests/check_wb_dex.cpp'),str(next(mj.glob('libmujoco*.dylib'))),'-o',str(binary)],check=True)
   result=subprocess.run([str(binary),str(ROOT/'android/assets')],env=dict(os.environ,DYLD_LIBRARY_PATH=str(mj)),capture_output=True,text=True)
   self.assertEqual(result.returncode,0,result.stdout+result.stderr)
 def test_runtime_does_not_abort_squat(self):
  import onnxruntime
  mj=Path(mujoco.__file__).parent;ort=Path(onnxruntime.__file__).parent/'capi'
  with tempfile.TemporaryDirectory() as temp:
   binary=Path(temp)/'physics-test'
   subprocess.run(['c++','-std=c++17','-O1','-I'+str(ROOT/'android/native'),'-I'+str(mj/'include'),'-I'+str(ROOT/'vendor/onnxruntime-android/headers'),str(ROOT/'tests/check_wb_dex_physics.cpp'),str(ROOT/'android/native/simulation.cpp'),str(next(mj.glob('libmujoco*.dylib'))),str(next(ort.glob('libonnxruntime*.dylib'))),'-o',str(binary)],check=True)
   result=subprocess.run([str(binary),str(ROOT/'android/assets')],env=dict(os.environ,DYLD_LIBRARY_PATH=str(mj)+os.pathsep+str(ort)),capture_output=True,text=True)
   self.assertEqual(result.returncode,0,result.stdout+result.stderr)
 def test_passive_cup_settles_on_shelf_and_in_box(self):
  m,_=make_model('wb_dex');self.assertTrue(m.opt.enableflags & mujoco.mjtEnableBit.mjENBL_MULTICCD)
  # Fixed distant robot isolates the contact fixture; cup integrated freely.
  b=m.body('task_object_0').id;j=m.joint('task_object_0_free');q=int(j.qposadr[0]);v=int(j.dofadr[0])
  for surface,base in [('shelf',(.65,0,.55)),('box',(.58,-.64,.018))]:
   for yaw in [0.,.7,2.1]:
    with self.subTest(surface=surface,yaw=yaw):
     d=mujoco.MjData(m);d.qpos[0]=-2;d.qpos[2]=2;d.qpos[q:q+3]=np.array(base)+[0,0,.025]
     d.qpos[q+3:q+7]=[np.cos(yaw/2),0,0,np.sin(yaw/2)];robot=d.qpos[:q].copy();linear=[];angular=[]
     for tick in range(4000):
      d.qpos[:q]=robot;d.qvel[:v]=0;mujoco.mj_step(m,d)
      if tick>=3000 and tick%10==0:
       vel=np.zeros(6);mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_BODY,b,vel,0);linear.append(np.linalg.norm(vel[3:]));angular.append(np.linalg.norm(vel[:3]))
     self.assertLess(max(linear),.001,'Persistent linear chatter');self.assertLess(max(angular),.03,'Persistent angular chatter')
     self.assertLess(np.linalg.norm(d.qpos[q:q+2]-base[:2]),.01);self.assertLess(abs(d.qpos[q+2]-base[2]),.003)
     self.assertEqual(sum(w.number for w in d.warning),0)
