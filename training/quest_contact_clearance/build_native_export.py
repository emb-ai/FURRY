"""Compile against an explicit native source snapshot; record every input hash."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


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
    libraries = [next(mj.glob('libmujoco*.dylib')), next(ort.glob('libonnxruntime*.dylib'))]
    sources = [cpp] + [native/n for n in ('gmr.cpp', 'meta_retarget.cpp', 'simulation.cpp')]
    cmd = ['c++', '-O2', '-std=c++17', '-I'+str(native), '-I'+str(mj/'include'),
           '-I'+str(a.source.resolve()/'vendor/onnxruntime-android/headers')]
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
