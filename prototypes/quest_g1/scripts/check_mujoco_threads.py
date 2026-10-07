"""Check single/threaded engine equality through dense contacts and resets."""
from pathlib import Path
import os
import subprocess
import sys
import mujoco

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from g1_sim.scene import make_model
from g1_sim.collisions import primitive_collisions

out = ROOT / 'outputs/thread-check'
out.mkdir(parents=True, exist_ok=True)
_, xml = make_model('lab')
xml, _ = primitive_collisions(xml)
scene = out / 'scene.xml'
scene.write_text(xml)
package = Path(mujoco.__file__).parent
pattern = 'libmujoco*.dylib' if sys.platform == 'darwin' else 'libmujoco.so*'
library = sorted(package.glob(pattern))[0]
binary = out / 'check_threads'
subprocess.run([os.environ.get('CXX', 'c++'), '-O2', '-std=c++17',
                '-I' + str(package / 'include'),
                str(ROOT / 'tests/check_mujoco_threads.cpp'), str(library),
                '-Wl,-rpath,' + str(package), '-o', str(binary)], check=True)
env = os.environ.copy()
if sys.platform == 'darwin':
    env['DYLD_LIBRARY_PATH'] = str(package) + ':' + env.get('DYLD_LIBRARY_PATH', '')
subprocess.run([str(binary), str(scene)], env=env, check=True)
