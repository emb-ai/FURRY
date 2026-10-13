"""Run the actual native retaining/magnetic mechanism, without robot fixtures."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
import mujoco
from g1_sim.scene import make_model


class RubikTests(unittest.TestCase):
    def test_passive_mechanism_and_magnetic_alignment(self):
        root=Path(__file__).resolve().parents[1];_,xml=make_model('rubik');tree=ET.fromstring(xml)
        world=tree.findall('worldbody')[-1]
        pelvis=tree.find(".//body[@name='pelvis']")
        next(w for w in tree.findall('worldbody') if pelvis in list(w)).remove(pelvis)
        tree.remove(tree.find('actuator'))
        for tag in ('contact','sensor'):
            element=tree.find(tag)
            if element is not None:tree.remove(element)
        core=tree.find(".//body[@name='rubik_core']");core.remove(core.find('freejoint'));core.set('pos','.36 0 1.2')
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);path=folder/'fixture.xml';path.write_text(ET.tostring(tree,encoding='unicode'))
            mj=Path(mujoco.__file__).parent;lib=next(mj.glob('libmujoco*.dylib' if sys.platform=='darwin' else 'libmujoco.so*'))
            binary=folder/'check_rubik'
            subprocess.run([os.environ.get('CXX','c++'),'-std=c++17','-O2','-Wall','-Wextra','-Werror',
                '-I'+str(root/'android/native'),'-I'+str(mj/'include'),str(root/'tests/check_rubik_mechanism.cpp'),str(lib),
                '-Wl,-rpath,'+str(mj),'-o',str(binary)],check=True)
            env=dict(os.environ);env['DYLD_LIBRARY_PATH' if sys.platform=='darwin' else 'LD_LIBRARY_PATH']=str(mj)
            result=subprocess.run([str(binary),str(path)],env=env,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertIn('snapshots_and_no_teleport_and_inverse_scramble_passed',result.stdout)
            print(result.stdout.strip())


if __name__=='__main__':unittest.main()
