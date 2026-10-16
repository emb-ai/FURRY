import sys
from pathlib import Path
import os,subprocess,tempfile,unittest
import mujoco

class VirtualLegLiftTests(unittest.TestCase):
    def test_native_ik_and_phase_contract(self):
        root=Path(__file__).resolve().parents[1];mj=Path(mujoco.__file__).parent
        with tempfile.TemporaryDirectory() as tmp:
            exe=Path(tmp)/'check'
            subprocess.run(['c++','-std=c++17','-I'+str(root/'android/native'),'-I'+str(mj/'include'),str(root/'tests/check_virtual_leg_lift.cpp'),str(next(mj.glob('libmujoco.*'))),'-o',str(exe)],check=True)
            subprocess.run([str(exe)],env=dict(os.environ, **{('DYLD_LIBRARY_PATH' if sys.platform == 'darwin' else 'LD_LIBRARY_PATH'): str(mj)}),check=True)
