"""Exercise native tennis dynamics against analytic physics and real MuJoCo contacts."""
from pathlib import Path
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import mujoco
from scripts.scene_assets import export_scenes
import json


class TennisTests(unittest.TestCase):
    def test_native_physics_and_task(self):
        root = Path(__file__).resolve().parents[1]
        compiler = os.environ.get('CXX') or shutil.which('c++')
        if not compiler:
            self.skipTest('A C++17 compiler is required')
        mj = Path(mujoco.__file__).parent
        candidates = list(mj.glob('libmujoco*.dylib' if sys.platform == 'darwin' else 'libmujoco.so*'))
        if not candidates:
            self.skipTest('MuJoCo shared library is required')
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            assets = folder / 'assets'
            assets.mkdir()
            (assets / 'recording_metadata.json').write_text(json.dumps({'schema_version': 1, 'sha256': {}}))
            export_scenes(assets)
            binary = folder / 'check_tennis'
            subprocess.run([compiler, '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror',
                '-I'+str(root/'android/native'), '-I'+str(mj/'include'),
                str(root/'tests/check_tennis.cpp'), str(candidates[0]), '-o', str(binary)], check=True)
            env = dict(os.environ)
            env['DYLD_LIBRARY_PATH' if sys.platform == 'darwin' else 'LD_LIBRARY_PATH'] = str(mj)
            result = subprocess.run([str(binary), str(assets/'scene-tennis.xml'), str(folder/'recordings')],
                env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            self.assertIn('recording passed', result.stdout)


if __name__ == '__main__':
    unittest.main()
