"""Exercise native capture coverage and the unchanged idle projection."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


class CaptureFovTests(unittest.TestCase):
    def test_native_projection_rays(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "capture-fov"
            subprocess.run([
                os.environ.get("CXX", "c++"), "-std=c++17", "-Wall", "-Wextra", "-Werror",
                "-I" + str(root / "android/native"),
                "-I" + str(root / "vendor/openxr-android/prefab/modules/headers/include"),
                str(root / "tests/check_capture_fov.cpp"), "-o", str(binary),
            ], check=True)
            subprocess.run([str(binary)], check=True)
