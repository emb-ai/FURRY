"""Native render-only camera clearance math; no robot physics changes."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

class CameraFadeTests(unittest.TestCase):
    def test_native_camera_clearance(self):
        root=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            binary=Path(directory)/'check_camera_fade'
            subprocess.run([os.environ.get('CXX','c++'),'-std=c++17','-O2','-Wall','-Wextra','-Werror',
                            '-I'+str(root/'android/native'),str(root/'tests/check_camera_fade.cpp'),'-o',str(binary)],check=True)
            subprocess.run([str(binary)],check=True)
