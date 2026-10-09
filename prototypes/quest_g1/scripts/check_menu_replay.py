"""Compile scene-aware replay tools and run synthetic pause/snapshot episodes.

Uses the already exported android/assets and the project's MuJoCo/ONNX Python
runtime libraries. No exporter is run and no participant recordings are read.
"""
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]


def main():
    runtime = subprocess.check_output([
        str(ROOT/'.venv/bin/python'), '-c',
        'import json,mujoco,onnxruntime;from pathlib import Path;'
        'print(json.dumps([str(Path(mujoco.__file__).parent),str(Path(onnxruntime.__file__).parent/"capi")]))'
    ], text=True)
    mj, ort = map(Path, json.loads(runtime))
    extension = '*.dylib' if sys.platform == 'darwin' else '*.so*'
    mjlib = next(mj.glob('libmujoco'+extension))
    ortlib = next(ort.glob('libonnxruntime'+extension))
    env = dict(os.environ)
    env['DYLD_LIBRARY_PATH' if sys.platform == 'darwin' else 'LD_LIBRARY_PATH'] = str(mj)+os.pathsep+str(ort)
    with tempfile.TemporaryDirectory(prefix='furry-menu-replay-') as directory:
        folder = Path(directory)
        common = [str(ROOT/'android/native'/name) for name in ('gmr.cpp', 'meta_retarget.cpp', 'simulation.cpp')]
        for name in ('replay_gmr', 'replay_gmr_dynamics', 'check_replay_menu'):
            source = ROOT/('tests/check_replay_menu.cpp' if name == 'check_replay_menu' else 'android/native/'+name+'.cpp')
            subprocess.run([os.environ.get('CXX', 'c++'), '-O1', '-std=c++17', '-I'+str(ROOT/'android/native'),
                            '-I'+str(ROOT/'vendor/mujoco/include'), '-I'+str(ROOT/'vendor/onnxruntime-android/headers'),
                            str(source), *common, str(mjlib), str(ortlib), '-o', str(folder/name)], check=True)
        cases = [(name, None) for name in ('lab', 'stand', 'cup', 'push_t')]
        catalog = json.loads((ROOT/'android/assets/policy_catalog.json').read_text())
        cases += [('cup', row) for row in catalog[1:]]
        for scene, policy in cases:
            case = scene if policy is None else scene+"-"+policy["id"]
            episode_root = folder/case
            episode_root.mkdir()
            metadata = 'recording_metadata.json' if scene == 'lab' else f'recording_metadata-{scene}.json'
            if policy:
                metadata = f'recording_metadata-{scene}-policy-{policy["id"]}.json'
                shutil.copy2(ROOT/'android/assets'/policy['file'], episode_root/policy['file'])
            shutil.copy2(ROOT/'android/assets'/metadata, episode_root/metadata)
            command = [str(folder/'check_replay_menu'), str(ROOT/'android/assets'), str(episode_root), scene]
            if policy:
                command.append(policy['id'])
            episode = subprocess.check_output(command, env=env, text=True).strip()
            if policy:
                manifest = json.loads((Path(episode)/'manifest.json').read_text())
                if manifest['policy_id'] != policy['id'] or manifest['policy_sha256'] != policy['sha256']:
                    raise AssertionError('Recording has the wrong policy identity')
                if not (Path(episode)/'policy.onnx').is_file():
                    raise AssertionError('Recording did not preserve selected weights')
            # lab metadata has no scene field: this also checks legacy fallback.
            subprocess.run([str(folder/'replay_gmr'), str(ROOT/'android/assets'), episode], env=env, check=True)
            for mode in ('saved', 'retarget'):
                output = folder/f'{case}-{mode}.csv'
                subprocess.run([str(folder/'replay_gmr_dynamics'), str(ROOT/'android/assets'), episode,
                                mode, str(output)], env=env, check=True)
                with output.open() as stream:
                    rows = list(csv.DictReader(stream))
                if len(rows) != 162 or abs(float(rows[-1]['sim_s'])-1.59) > 1e-9:
                    raise AssertionError(f'{scene}: snapshots advanced physics or physical frames were lost')
                if max(float(row['q_error']) for row in rows) > 1e-5:
                    raise AssertionError(f'{scene}: dynamic scene/snapshot replay diverged')
                if max(float(row['cmd_error']) for row in rows) > 1e-5:
                    raise AssertionError(f'{scene}: pause blending/reference replay diverged')
            print(f'{case}: policy, scene, dimensions, pauses and calibration snapshots passed', flush=True)


if __name__ == '__main__':
    main()
