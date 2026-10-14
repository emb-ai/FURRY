"""Compile against an explicit native source snapshot; record every input hash."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    import mujoco
    import onnxruntime
    native = a.source.resolve()/'android/native'
    mj = Path(mujoco.__file__).parent
    ort = Path(onnxruntime.__file__).parent/'capi'
    cpp = Path(__file__).with_name('export_native_commands.cpp')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    # macOS wheels ship .dylib; Linux wheels ship versioned .so, found via rpath.
    lib = '*.dylib' if sys.platform == 'darwin' else '*.so*'
    libraries = [next(mj.glob('libmujoco'+lib)), next(ort.glob('libonnxruntime'+lib))]
    sources = [cpp] + [native/n for n in ('gmr.cpp', 'meta_retarget.cpp', 'simulation.cpp')]
    cmd = ['c++', '-O2', '-std=c++17', '-I'+str(native), '-I'+str(mj/'include'),
           '-I'+str(a.source.resolve()/'vendor/onnxruntime-android/headers')]
    if sys.platform != 'darwin':
        # The onnxruntime wheel has libonnxruntime.so.1.x.y but no file named
        # by its SONAME (libonnxruntime.so.1): link the SONAMEs next to the binary.
        libdir = a.output.resolve().parent/(a.output.name+'.lib')
        libdir.mkdir(exist_ok=True)
        for library in libraries:
            dynamic = subprocess.run(['readelf', '-d', str(library)], check=True, capture_output=True, text=True).stdout
            soname = dynamic.split('Library soname: [', 1)[1].split(']', 1)[0]
            link = libdir/soname
            if not link.exists():
                link.symlink_to(library)
        cmd += [f'-Wl,-rpath,{libdir}']
    subprocess.run(cmd + list(map(str, sources+libraries)) + ['-o', str(a.output)], check=True)
    assets = a.source.resolve()/'android/assets'
    inputs = sources + list(native.glob('*.h')) + list(assets.rglob('*.xml')) + list(assets.glob('*.txt'))
    receipt = {'source': str(a.source.resolve()), 'mujoco': mujoco.__version__,
               'onnxruntime': onnxruntime.__version__,
               'dyld_library_path': os.pathsep.join(map(str, (mj, ort))),
               'sha256': {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in inputs+libraries+[a.output]},
               'contract': {'markers': 14, 'full_feet': False, 'neutral_wrists': False,
                            'wrist_frame_v2': False, 'catch_up': False, 'startup_blend': False,
                            'offline_floor_guard': False, 'offline_filter_hz': None,
                            'camera_tracking': 'explicit CLI argument', 'travel_gain': 1.}}
    a.output.with_suffix('.json').write_text(json.dumps(receipt, indent=2)+'\n')


if __name__ == '__main__':
    main()
