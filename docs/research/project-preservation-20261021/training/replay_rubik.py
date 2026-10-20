"""Rebuild the archived native runtime and replay ordinary wrist/Grip commands."""
from pathlib import Path
import argparse
import ctypes
import json
import platform
import shutil
import subprocess
import sys
import numpy as np
import mujoco
import onnxruntime
import archive


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace', type=Path, required=True)
    p.add_argument('--frames', type=int, help='Optional prefix check; omit for all 28,367 frames')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.frames is not None and a.frames <= 0:
        p.error('--frames must be positive')
    if mujoco.__version__ != '3.3.7':
        raise RuntimeError('Exact recorded replay requires MuJoCo 3.3.7')
    workspace = a.workspace.resolve()
    original = workspace / 'outputs/rubik-canonical-20261015/original-native'
    sequences = workspace / 'outputs/rubik-sequences-20261015'
    assets = sequences / 'project/android/assets'
    build = workspace / 'replay-build'
    headers = build / 'ort-headers'
    headers.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((archive.HERE / 'manifest.json').read_text())
    for row in manifest['files']:
        if row['workspace_path'].startswith('outputs/cup-native50-20261015/ort-headers/'):
            shutil.copy2(archive.check(row), headers / Path(row['workspace_path']).name)
    mj = Path(mujoco.__file__).parent
    ort = Path(onnxruntime.__file__).parent / 'capi'
    mac = platform.system() == 'Darwin'
    mjlib = next(mj.glob('libmujoco*.dylib' if mac else 'libmujoco.so*'))
    ortlib = next(ort.glob('libonnxruntime*.dylib' if mac else 'libonnxruntime.so*'))
    library = build / ('replay.dylib' if mac else 'replay.so')
    subprocess.run(['c++', '-O2', '-std=c++17', '-dynamiclib' if mac else '-shared', '-fPIC',
                    '-I' + str(original), '-I' + str(mj / 'include'), '-I' + str(headers),
                    str(sequences / 'bridge.cpp'), str(original / 'simulation.cpp'),
                    str(mjlib), str(ortlib), '-Wl,-rpath,' + str(mjlib.parent),
                    '-Wl,-rpath,' + str(ort), '-o', str(library)], check=True)
    lib = ctypes.CDLL(str(library))
    ptr = ctypes.c_void_p
    fp = np.ctypeslib.ndpointer(np.float32, flags='C_CONTIGUOUS')
    dp = np.ctypeslib.ndpointer(np.float64, flags='C_CONTIGUOUS')
    for name, args, result in [
        ('create_policy', [ctypes.c_char_p] * 3, ptr), ('destroy', [ptr], None),
        ('correct', [ptr, fp, dp, fp], None), ('step_grip', [ptr, fp, ctypes.c_int, ctypes.c_double, ctypes.c_double], ctypes.c_int),
        ('state', [ptr, dp], ctypes.c_int), ('task', [ptr, dp], None),
        ('seed', [ptr, ctypes.c_uint], None), ('warnings', [ptr], ctypes.c_int)]:
        fn = getattr(lib, name)
        fn.argtypes, fn.restype = args, result
    assert not hasattr(lib, 'fingers') and not hasattr(lib, 'thumbs')
    sys.path.insert(0, str(sequences / 'project'))
    from g1_sim.controller import JOINTS
    model = mujoco.MjModel.from_xml_path(str(assets / 'scene-rubik.xml'))
    handle = lib.create_policy(str(assets).encode(), b'rubik', b'homie-standing')
    if not handle:
        raise RuntimeError('Native runtime initialization failed')
    try:
        lib.seed(handle, 0)
        q = np.zeros(model.nq)
        userdata = np.zeros(model.nuserdata)
        lib.state(handle, q)
        qa = [int(model.joint(name).qposadr[0]) for name in JOINTS]
        command = np.r_[0, 0, .74, 0, 0, 0, q[qa]].astype(np.float32)
        if lib.step_grip(handle, command, 2500, 0., 0.):
            raise RuntimeError('Warmup failed')
        file = workspace / 'outputs/rubik-feedback-20261015/second-other-face.npz'
        with np.load(file, allow_pickle=False) as z:
            count = len(z['commands']) if a.frames is None else min(a.frames, len(z['commands']))
            maxq = maxu = 0.
            for k, c in enumerate(z['commands'][:count]):
                corrected = command.copy()
                lib.correct(handle, command, np.ascontiguousarray(c[:14], dtype=np.float64), corrected)
                if lib.step_grip(handle, corrected, 10, float(c[-2]), float(c[-1])):
                    raise RuntimeError(f'Physics failed at frame {k}')
                lib.state(handle, q)
                lib.task(handle, userdata)
                maxq = max(maxq, float(np.max(np.abs(q - z['qpos'][k]))))
                maxu = max(maxu, float(np.max(np.abs(userdata - z['userdata'][k]))))
        result = {'frames': count, 'full_rollout': count == 28367, 'qpos_max_error': maxq,
                  'userdata_max_error': maxu, 'warnings': int(lib.warnings(handle)),
                  'mujoco_version': mujoco.__version__, 'platform': platform.platform(),
                  'control': 'two wrist poses and two ordinary scalar Grip values'}
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result, indent=2))
        if maxq > 1e-8 or maxu > 1e-8 or result['warnings']:
            raise SystemExit('Replay differs from archived trajectory; do not claim exact parity')
    finally:
        lib.destroy(handle)


if __name__ == '__main__':
    main()
