"""Compile scene-aware replay tools and run synthetic pause/snapshot episodes.

Uses the already exported android/assets and the project's MuJoCo/ONNX Python
runtime libraries. No exporter is run and no participant recordings are read.
"""
import csv
import hashlib
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
    ortlib = next(p for p in ort.glob('libonnxruntime'+extension) if not p.name.startswith('libonnxruntime_providers_'))
    env = dict(os.environ)
    env['DYLD_LIBRARY_PATH' if sys.platform == 'darwin' else 'LD_LIBRARY_PATH'] = str(mj)+os.pathsep+str(ort)
    with tempfile.TemporaryDirectory(prefix='furry-menu-replay-') as directory:
        folder = Path(directory)
        if sys.platform != "darwin":
            (folder/"libonnxruntime.so.1").symlink_to(ortlib)
            env["LD_LIBRARY_PATH"] = directory + os.pathsep + env["LD_LIBRARY_PATH"]
        common = [str(ROOT/'android/native'/name) for name in ('gmr.cpp', 'meta_retarget.cpp', 'simulation.cpp')]
        for name in ('replay_gmr', 'replay_gmr_dynamics', 'check_replay_menu'):
            source = ROOT/('tests/check_replay_menu.cpp' if name == 'check_replay_menu' else 'android/native/'+name+'.cpp')
            subprocess.run([os.environ.get('CXX', 'c++'), '-O1', '-std=c++17', '-I'+str(ROOT/'android/native'),
                            '-I'+str(ROOT/'vendor/mujoco/include'), '-I'+str(ROOT/'vendor/onnxruntime-android/headers'),
                            str(source), *common, str(mjlib), str(ortlib), '-o', str(folder/name)], check=True)
        cases = [(name, None, False, 'observer') for name in ('lab', 'stand', 'cup', 'push_t', 'tennis', 'rubik', 'wb_dex', 'bimanual', 'basket')]
        catalog = json.loads((ROOT/'android/assets/policy_catalog.json').read_text())
        cases += [('cup', row, False, 'observer') for row in catalog[1:]]
        cases += [('cup', row, True, 'observer') for row in catalog]
        cases += [('rubik', row, catch, 'observer') for row in catalog if row.get('backend') in ('groot_decoupled_v1','sonic_vr3_v1') for catch in (False,True)]
        cases += [('cup', catalog[0], True, view) for view in ('first', 'switch')]
        # Cadence and wrist-frame versions are episode semantics. Exercise both
        # old100 Hz manifests and missing-field fallbacks against the new live50.
        cases = [(*case, None, None) for case in cases]
        cases += [('stand', catalog[0], False, 'observer', hz, wrist)
                  for hz in (50, 100) for wrist in (False, True)]
        cases += [('stand', row, False, 'observer', 0, False)
                  for row in (catalog[0], [row for row in catalog if row.get("backend", "twist2") == "twist2"][-1])]
        if os.environ.get('G1_COLLECTION_REPLAY_ONLY'):
            cases = [case for case in cases if case[0] in ('cup','tennis','wb_dex','bimanual','basket') and case[1] is None]
        if os.environ.get('G1_STANDING_REPLAY_ONLY'):
            standing=next(row for row in catalog if row['id']=='homie-standing')
            cases=[(scene,standing,False,view,None,None) for scene in ('stand','cup','bimanual','rubik') for view in ('observer','first','switch')]
        # This recorder/replayer carries TWIST-style velocity references. DexWeave
        # needs its absolute 36-value reference stream; neither this synthetic
        # fixture nor replay_gmr_dynamics implements that contract yet.
        unsupported = [case for case in cases if case[1] and case[1].get('backend') == 'dexweave_body_v1']
        if unsupported:
            print(f'SKIP {len(unsupported)} DexWeave cases: absolute-reference dynamic replay is unsupported', flush=True)
        cases = [case for case in cases if case not in unsupported]
        for scene, policy, catch_up, view, override_hz, wrist_v2 in cases:
            if policy and policy.get('input_patch')=='homie_stand_v1':catch_up=False
            case = scene if policy is None else scene+"-"+policy["id"]
            if catch_up:
                case += "-catch-up-"+view
            if override_hz is not None:
                case += f'-rate{override_hz}-wrist{int(wrist_v2)}'
            episode_root = folder/case
            episode_root.mkdir()
            metadata = 'recording_metadata.json' if scene == 'lab' else f'recording_metadata-{scene}.json'
            if policy:
                metadata = f'recording_metadata-{scene}-policy-{policy["id"]}.json'
                shutil.copy2(ROOT/'android/assets'/policy['file'], episode_root/policy['file'])
                if policy.get('secondary_file'):
                    shutil.copy2(ROOT/'android/assets'/policy['secondary_file'], episode_root/policy['secondary_file'])
                    profile_name='sonic_motor_profile.txt' if policy.get('backend')=='sonic_vr3_v1' else 'body_motor_profile.txt'
                    shutil.copy2(ROOT/'android/assets'/profile_name, episode_root/profile_name)
                if policy.get('planner_file'):
                    shutil.copy2(ROOT/'android/assets'/policy['planner_file'], episode_root/policy['planner_file'])
            fixture = json.loads((ROOT/'android/assets'/metadata).read_text())
            if override_hz is not None:
                fixture.pop('controller_stage_targets',None)
                fixture.pop('controller_anatomical_frame',None)
                if override_hz:
                    fixture['policy_hz'] = override_hz
                else:
                    fixture.pop('policy_hz', None)
                fixture['wrist_frame_v2'] = wrist_v2
            (episode_root/metadata).write_text(json.dumps(fixture))
            command = [str(folder/'check_replay_menu'), str(ROOT/'android/assets'), str(episode_root), scene]
            if policy:
                command.append(policy['id'])
                command += ['on' if catch_up else 'off', view]
            episode = subprocess.check_output(command, env=env, text=True).strip()
            if policy:
                manifest = json.loads((Path(episode)/'manifest.json').read_text())
                if manifest['catch_up'] != catch_up or manifest['first_person'] != (view == 'first'):
                    raise AssertionError('Wrong catch-up/view settings in recording')
                if manifest['policy_id'] != policy['id'] or manifest['policy_sha256'] != policy['sha256']:
                    raise AssertionError('Recording has the wrong policy identity')
                if not (Path(episode)/'policy.onnx').is_file():
                    raise AssertionError('Recording did not preserve selected weights')
                if policy.get('secondary_file'):
                    if manifest['policy_backend'] != policy['backend'] or manifest['policy_secondary_sha256'] != policy['secondary_sha256']:
                        raise AssertionError('Wrong decoupled backend identity')
                    if not (Path(episode)/'policy-secondary.onnx').is_file():
                        raise AssertionError('Missing archived walking policy')
                    profile_digest = hashlib.sha256((Path(episode)/profile_name).read_bytes()).hexdigest()
                    if profile_digest != manifest['body_motor_profile_sha256'] or manifest['body_motor_profile'] != fixture['body_motor_profile']:
                        raise AssertionError('Missing or different archived body motor profile')
                if policy.get('planner_file'):
                    if hashlib.sha256((Path(episode)/'policy-planner.onnx').read_bytes()).hexdigest()!=policy['planner_sha256']:
                        raise AssertionError('Missing archived SONIC planner')
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
            # Each case has independent recorded model copies. Drop only this
            # completed synthetic fixture so large planners do not accumulate.
            shutil.rmtree(episode_root)


if __name__ == '__main__':
    main()
