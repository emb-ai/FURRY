"""Compile and exercise scene selection and pause parsing without runtime libs."""
from pathlib import Path
import os
import shutil
import subprocess
import tempfile
import unittest


class EpisodeManifestTests(unittest.TestCase):
    def test_native_manifest_fixtures(self):
        root = Path(__file__).resolve().parents[1]
        compiler = os.environ.get('CXX') or shutil.which('c++')
        if not compiler:
            self.skipTest('A C++17 compiler is required')
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / 'check_episode_manifest'
            compiled = subprocess.run([compiler, '-std=c++17', '-Wall', '-Wextra', '-Werror',
                                       '-I'+str(root/'android/native'), str(root/'tests/check_episode_manifest.cpp'),
                                       '-o', str(binary)], capture_output=True, text=True)
            self.assertEqual(compiled.returncode, 0, compiled.stderr)
            result = subprocess.run([str(binary)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('fixtures passed', result.stdout)


if __name__ == '__main__':
    unittest.main()
