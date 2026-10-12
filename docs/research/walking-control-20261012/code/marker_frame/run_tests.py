"""Portable POSIX runner for an offline target-frame prototype, not runtime code.

Requires Python MuJoCo and a C++17 compiler. Supply a prepared gmr_model.xml.
Build products live in a temporary directory; stdout is a compact JSON receipt.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--cxx', default=os.environ.get('CXX', 'c++'))
    args = parser.parse_args()
    import mujoco
    here = Path(__file__).resolve().parent
    lib = Path(mujoco.__file__).resolve().parent
    pattern = 'libmujoco*.dylib' if sys.platform == 'darwin' else 'libmujoco.so*'
    libraries = sorted(lib.glob(pattern))
    if not libraries:
        raise RuntimeError('MuJoCo shared library not found; macOS/Linux supported')
    model = args.model.resolve(strict=True)
    with tempfile.TemporaryDirectory(prefix='furry-marker-frame-') as folder:
        binary = Path(folder) / 'test-marker-frame'
        result = Path(folder) / 'result.json'
        subprocess.run([args.cxx, '-O2', '-std=c++17', '-Wall', '-Wextra', '-Werror',
                        '-I' + str(lib / 'include'),
                        str(here / 'test_anatomical_marker_offset.cpp'), str(libraries[0]),
                        '-Wl,-rpath,' + str(lib), '-o', str(binary)], check=True)
        env = os.environ.copy()
        if sys.platform == 'darwin':
            # The wheel retains a framework install-name; resolve its basename.
            env['DYLD_LIBRARY_PATH'] = str(lib) + os.pathsep + env.get('DYLD_LIBRARY_PATH', '')
        completed = subprocess.run([str(binary), str(model), str(result)],
                                   capture_output=True, text=True, env=env)
        if completed.returncode:
            raise RuntimeError(completed.stderr or completed.stdout or
                               f'Test exited with {completed.returncode}')
        report = dict(mujoco_version=mujoco.__version__,
                      model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
                      result=json.loads(result.read_text()))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
