"""Exercise actual packaged ONNX policies through the native Quest controller."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import mujoco
import onnxruntime

ROOT = Path(__file__).resolve().parents[1]


def main():
    mj = Path(mujoco.__file__).parent
    ort = Path(onnxruntime.__file__).parent / 'capi'
    suffix = '*.dylib' if sys.platform == 'darwin' else '*.so*'
    env = dict(os.environ)
    env['DYLD_LIBRARY_PATH' if sys.platform == 'darwin' else 'LD_LIBRARY_PATH'] = str(mj) + os.pathsep + str(ort)
    with tempfile.TemporaryDirectory() as directory:
        binary = Path(directory) / 'check_policy_selection'
        subprocess.run([os.environ.get('CXX', 'c++'), '-O1', '-std=c++17',
                        '-I' + str(ROOT / 'android/native'), '-I' + str(mj / 'include'),
                        '-I' + str(ROOT / 'vendor/onnxruntime-android/headers'),
                        str(ROOT / 'tests/check_policy_selection.cpp'), str(ROOT / 'android/native/simulation.cpp'),
                        str(next(mj.glob('libmujoco' + suffix))), str(next(p for p in ort.glob('libonnxruntime'+suffix) if not p.name.startswith('libonnxruntime_providers_'))),
                        '-o', str(binary)], check=True)
        subprocess.run([str(binary), str(ROOT / 'android/assets')], env=env, check=True)


if __name__ == '__main__':
    main()
