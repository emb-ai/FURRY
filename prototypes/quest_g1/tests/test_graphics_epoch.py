import unittest,subprocess,tempfile,os
from pathlib import Path
class GraphicsContextTests(unittest.TestCase):
    def test_recreated_context_drops_all_old_cached_handles(self):
        root=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            binary=Path(tmp)/"graphics-context-check"
            subprocess.run([os.environ.get("CXX","c++"),"-std=c++17","-I"+str(root/"android/native"),str(root/"tests/check_graphics_epoch.cpp"),"-o",str(binary)],check=True)
            subprocess.run([str(binary)],check=True,capture_output=True)
