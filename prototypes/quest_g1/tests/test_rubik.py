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
    def test_command_intent_and_grip_limits(self):
        import numpy as np
        root=Path(__file__).resolve().parents[1];m,xml=make_model('rubik')
        self.assertEqual(m.opt.noslip_iterations,2)
        self.assertTrue(m.opt.enableflags & int(mujoco.mjtEnableBit.mjENBL_MULTICCD))
        for a in range(m.nu):
            name=m.actuator(a).name
            if 'hand_' not in name or 'thumb_0' in name:continue
            sign=1 if name.startswith('left') else -1
            target=sign*(1. if 'thumb_1' in name else 1.74) if 'thumb' in name else -sign*(1.57 if '_0_' in name else 1.74)
            self.assertEqual(m.actuator_gainprm[a,0],2.)
            self.assertTrue(m.actuator_ctrllimited[a])
            np.testing.assert_allclose(m.actuator_ctrlrange[a], sorted([0,target*.8]))
            np.testing.assert_allclose(m.actuator_forcerange[a],[-.35,.35])
        mj=Path(mujoco.__file__).parent;lib=next(mj.glob('libmujoco*.dylib' if sys.platform=='darwin' else 'libmujoco.so*'))
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);scene=folder/'scene.xml';scene.write_text(xml);binary=folder/'intent'
            subprocess.run(['c++','-O2','-std=c++17','-I'+str(root/'android/native'),'-I'+str(mj/'include'),str(root/'tests/check_rubik_intent.cpp'),str(lib),'-o',str(binary)],check=True)
            env=dict(os.environ);env['DYLD_LIBRARY_PATH' if sys.platform=='darwin' else 'LD_LIBRARY_PATH']=str(mj)
            subprocess.run([str(binary),str(scene)],env=env,check=True)

    def test_grip_guard_physics_and_archived_replay(self):
        import numpy as np
        root=Path(__file__).resolve().parents[1];model,xml=make_model('rubik');tree=ET.fromstring(xml)
        pelvis=tree.find(".//body[@name='pelvis']")
        next(w for w in tree.findall('worldbody') if pelvis in list(w)).remove(pelvis)
        tree.remove(tree.find('actuator'))
        for tag in ('contact','sensor'):
            element=tree.find(tag)
            if element is not None:tree.remove(element)
        core=tree.find(".//body[@name='rubik_core']");core.remove(core.find('freejoint'));core.set('pos','.36 0 1.2')
        poses=model.numeric('rubik_initial').data.reshape(8,7)
        for i,pose in enumerate(poses):
            body=tree.find(f".//body[@name='rubik_cubie_{i}']")
            body.set('pos',' '.join(map(str,np.array([.36,0,1.2])+pose[:3])))
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);guarded=folder/'guarded.xml';guarded.write_text(ET.tostring(tree,encoding='unicode'))
            tree.find('custom').remove(tree.find(".//numeric[@name='rubik_grip_guard_v2']"));tree.find('size').set('nuserdata','64')
            for joint in tree.findall('.//joint'):
                if joint.get('name','').startswith('rubik_hinge_'):joint.set('frictionloss','0')
            legacy=folder/'legacy.xml';legacy.write_text(ET.tostring(tree,encoding='unicode'))
            mj=Path(mujoco.__file__).parent;lib=next(mj.glob('libmujoco*.dylib' if sys.platform=='darwin' else 'libmujoco.so*'));binary=folder/'guard'
            subprocess.run(['c++','-O2','-std=c++17','-Wall','-Wextra','-Werror','-I'+str(root/'android/native'),'-I'+str(mj/'include'),str(root/'tests/check_rubik_grip_guard.cpp'),str(lib),'-o',str(binary)],check=True)
            env=dict(os.environ);env['DYLD_LIBRARY_PATH' if sys.platform=='darwin' else 'LD_LIBRARY_PATH']=str(mj)
            subprocess.run([str(binary),str(guarded),str(legacy)],env=env,check=True)

    def test_passive_mechanism_and_magnetic_alignment(self):
        root=Path(__file__).resolve().parents[1];_,xml=make_model('rubik');tree=ET.fromstring(xml)
        # Archived v1 scenes retain their original freely moving layers.
        tree.find('custom').remove(tree.find(".//numeric[@name='rubik_grip_guard_v2']"))
        tree.find('size').set('nuserdata','64')
        for joint in tree.findall('.//joint'):
            if joint.get('name','').startswith('rubik_hinge_'):joint.set('frictionloss','0')
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
