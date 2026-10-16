"""Verify controller pose authority with unchanged SDK wrist/body timestamps."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import mujoco
import onnxruntime


class NarrowStanceTests(unittest.TestCase):
    def test_native_narrow_stance_contract(self):
        root=Path(__file__).resolve().parents[1];mj=Path(mujoco.__file__).parent;ort=Path(onnxruntime.__file__).parent/'capi'
        pattern='*.dylib' if sys.platform=='darwin' else '*.so*'
        mjlib=next(mj.glob('libmujoco'+pattern));ortlib=next(p for p in ort.glob('libonnxruntime'+pattern) if not p.name.startswith('libonnxruntime_providers_'))
        with tempfile.TemporaryDirectory() as directory:
            binary=Path(directory)/'check_narrow_stance'
            if sys.platform != 'darwin':
                (Path(directory)/'libonnxruntime.so.1').symlink_to(ortlib)
            sources=[root/'tests/check_narrow_stance.cpp',*[root/'android/native'/s for s in ['gmr.cpp','meta_retarget.cpp','simulation.cpp']]]
            subprocess.run([os.environ.get('CXX','c++'),'-O1','-std=c++17','-I'+str(root/'android/native'),
                '-I'+str(mj/'include'),'-I'+str(root/'vendor/onnxruntime-android/headers'),
                *map(str,sources),str(mjlib),str(ortlib),'-o',str(binary)],check=True)
            env=dict(os.environ);env['DYLD_LIBRARY_PATH' if sys.platform=='darwin' else 'LD_LIBRARY_PATH']=directory+os.pathsep+str(mj)+os.pathsep+str(ort)
            result=subprocess.run([str(binary),str(root/'android/assets')],env=env,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertIn('Narrow stance target/frame/limits/switch-off checks passed',result.stdout)


if __name__=='__main__':unittest.main()
