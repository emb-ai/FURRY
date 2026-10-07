"""Build the native GMR oracle against the installed MuJoCo wheel (macOS/Linux)."""
from pathlib import Path
import os
import subprocess
import sys
import mujoco

ROOT = Path(__file__).resolve().parents[1]
package = Path(mujoco.__file__).parent
pattern = 'libmujoco*.dylib' if sys.platform == 'darwin' else 'libmujoco.so*'
libraries = sorted(package.glob(pattern))
if not libraries:
    raise SystemExit(f'No MuJoCo shared library in {package}')
build = ROOT / 'android/build'
build.mkdir(parents=True, exist_ok=True)
subprocess.run([
    os.environ.get('CXX', 'c++'), '-O2', '-std=c++17',
    '-I' + str(package / 'include'),
    str(ROOT / 'android/native/gmr.cpp'), str(ROOT / 'android/native/gmr_oracle.cpp'),
    str(libraries[0]), '-Wl,-rpath,' + str(package), '-o', str(build / 'gmr-oracle'),
], check=True)

# Anatomical ankle-to-toe mapping, independent of the ONNX policy/runtime.
subprocess.run([
    os.environ.get('CXX', 'c++'), '-O2', '-std=c++17',
    '-I' + str(package / 'include'),
    *[str(ROOT / 'android/native' / name) for name in
      ('gmr.cpp', 'meta_retarget.cpp', 'check_leg_mapping.cpp')],
    str(libraries[0]), '-Wl,-rpath,' + str(package), '-o', str(build / 'check_leg_mapping'),
], check=True)
env = os.environ.copy()
if sys.platform == 'darwin':
    env['DYLD_LIBRARY_PATH'] = str(package) + ':' + env.get('DYLD_LIBRARY_PATH', '')
subprocess.run([str(build / 'check_leg_mapping'), str(ROOT / 'android/assets')], env=env, check=True)
