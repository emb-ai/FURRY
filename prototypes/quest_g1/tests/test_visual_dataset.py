"""Dataset boundary/causality tests against the native recorder fixture."""
import csv
import json
from pathlib import Path
import shutil
import tempfile
import unittest
import subprocess
import os
import sys
import mujoco
import numpy as np
from g1_sim.visual_dataset import inspect_episode, unpack_archives, table

ROOT=Path(__file__).resolve().parents[1]

@unittest.skipUnless((ROOT/'android/assets/scene-cup.xml').exists(),'Prepare Android assets first')
class VisualCaptureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory(prefix='furry-learning-test-')
        cls.folder=Path(cls.temp.name)
        mj=Path(mujoco.__file__).parent
        extension='*.dylib' if sys.platform=='darwin' else '*.so*'
        library=next(mj.glob('libmujoco'+extension))
        binary=cls.folder/'fixture'
        subprocess.run(['c++','-O1','-std=c++17','-I'+str(ROOT/'android/native'),'-I'+str(mj/'include'),
                        str(ROOT/'tests/check_learning_capture.cpp'),str(library),'-o',str(binary)],check=True)
        env=dict(os.environ);env['DYLD_LIBRARY_PATH' if sys.platform=='darwin' else 'LD_LIBRARY_PATH']=str(mj)
        subprocess.run([str(binary),str(ROOT/'android/assets'),str(cls.folder/'raw')],env=env,check=True,capture_output=True)
        cls.paths=sorted((cls.folder/'raw/recordings').glob('episode-*'))
    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()
    def test_success_causal_alignment(self):
        e=inspect_episode(self.paths[0],True);ids=e['indices']
        self.assertEqual(e['actions'].shape[1],39)
        self.assertTrue(np.all(e['frames']['sim_time'][ids] <= e['grid']+1e-9))
        self.assertTrue(np.all(np.diff(ids)>0))
        self.assertEqual(e['camera']['outcome'][e['end']],1)
        self.assertLessEqual(e['grid'][-1],e['frames']['sim_time'][e['end']])
    def test_synthetic_excluded_by_default(self):
        with self.assertRaisesRegex(ValueError,'Synthetic'):inspect_episode(self.paths[0])
    def test_random_goal_recorded_without_changing_model(self):
        e=inspect_episode(self.paths[0],True)
        m=mujoco.MjModel.from_binary_path(str(self.paths[0]/'learning-model.mjb'))
        marker=m.body('cup_goal_marker');mid=int(marker.mocapid[0])
        goals=[]
        for path in self.paths:
            capture=json.loads((path/'learning.json').read_text())
            self.assertEqual(capture['model_sha256'],e['capture']['model_sha256'])
            scene=table(path/'scene_state.csv')
            target=np.stack([scene[f'mocap_pos_{3*mid+k}'] for k in range(3)],1)
            np.testing.assert_array_equal(target,np.broadcast_to(target[0],target.shape))
            goals.append(target[0])
        self.assertEqual(len({tuple(g) for g in goals}),len(self.paths))
        q=int(m.joint('cup_free').qposadr[0])
        endpoint=np.array([e['frames'][f'qpos_{q+k}'][e['end']] for k in range(3)])
        np.testing.assert_allclose(endpoint,goals[0],atol=1e-9)
    def test_failure_excluded(self):
        with self.assertRaisesRegex(ValueError,'did not succeed'):inspect_episode(self.paths[1],True)
    def test_pause_excluded(self):
        with self.assertRaisesRegex(ValueError,'pause'):inspect_episode(self.paths[2],True)
    def test_gap_excluded(self):
        with self.assertRaisesRegex(ValueError,'gap'):inspect_episode(self.paths[3],True)
    def test_model_tampering(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'episode';shutil.copytree(self.paths[0],p)
            with (p/'learning-model.mjb').open('ab') as f:f.write(b'x')
            with self.assertRaisesRegex(ValueError,'checksum'):inspect_episode(p,True)
    def test_unfinalized(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'episode';shutil.copytree(self.paths[0],p);(p/'complete.json').unlink()
            with self.assertRaises(FileNotFoundError):inspect_episode(p,True)
    def test_action_desync(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'episode';shutil.copytree(self.paths[0],p)
            lines=(p/'mimic.csv').read_text().splitlines();fields=lines[5].split(',');fields[1]='999';lines[5]=','.join(fields)
            (p/'mimic.csv').write_text('\n'.join(lines)+'\n')
            with self.assertRaisesRegex(ValueError,'Unsynchronized'):inspect_episode(p,True)
    def test_zip_traversal(self):
        import zipfile
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'out').mkdir()
            with zipfile.ZipFile(root/'bad.zip','w') as z:z.writestr('../bad','evil')
            with self.assertRaisesRegex(ValueError,'Unsafe'):unpack_archives(root,root/'out')

if __name__=='__main__':unittest.main()
