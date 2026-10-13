"""Verify the separate decoupled backend with official weights on the Quest model."""
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
    ortlib = next(p for p in ort.glob('libonnxruntime'+suffix) if not p.name.startswith('libonnxruntime_providers_'))
    env = dict(os.environ)
    env['DYLD_LIBRARY_PATH' if sys.platform == 'darwin' else 'LD_LIBRARY_PATH'] = str(mj) + os.pathsep + str(ort)
    with tempfile.TemporaryDirectory() as directory:
        if sys.platform != "darwin":
            (Path(directory)/"libonnxruntime.so.1").symlink_to(ortlib)
            env["LD_LIBRARY_PATH"] = directory + os.pathsep + env["LD_LIBRARY_PATH"]
        binary = Path(directory) / 'check_decoupled_backend'
        subprocess.run([os.environ.get('CXX', 'c++'), '-O1', '-std=c++17',
                        '-I' + str(ROOT / 'android/native'), '-I' + str(mj / 'include'),
                        '-I' + str(ROOT / 'vendor/onnxruntime-android/headers'),
                        str(ROOT / 'tests/check_decoupled_backend.cpp'), str(ROOT / 'android/native/simulation.cpp'),
                        str(next(mj.glob('libmujoco' + suffix))), str(ortlib),
                        '-o', str(binary)], check=True)
        subprocess.run([str(binary), str(ROOT / 'android/assets')], env=env, check=True)


if __name__ == '__main__':
    main()
