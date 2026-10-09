"""Check rigid alignment, bounded camera fusion and the outer pose loop."""
from pathlib import Path
import os
import subprocess
import sys
import mujoco
root = Path(__file__).resolve().parents[1]
package = Path(mujoco.__file__).parent
library = sorted(package.glob('libmujoco*.dylib' if sys.platform == 'darwin' else 'libmujoco.so*'))[0]
build = root/'android/build'
build.mkdir(parents=True, exist_ok=True)
env = dict(os.environ)
if sys.platform == 'darwin':
    env['DYLD_LIBRARY_PATH'] = str(package)+':'+env.get('DYLD_LIBRARY_PATH', '')
for name, sources in [('camera_fusion', ['android/native/camera_fusion.cpp']), ('camera_servo', []), ('button_latch', []), ('operator_skeleton', [])]:
    target = build/f'check_{name}'
    subprocess.run([os.environ.get('CXX', 'c++'), '-O2', '-std=c++17',
        '-I'+str(root/'android/native'), '-I'+str(package/'include'),
        str(root/f'tests/check_{name}.cpp'), *[str(root/s) for s in sources],
        str(library), '-Wl,-rpath,'+str(package), '-o', str(target)], check=True)
    subprocess.run([str(target)], check=True, env=env)
