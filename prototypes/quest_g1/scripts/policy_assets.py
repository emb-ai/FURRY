"""Package author policies and optional local ONNX candidates with provenance."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np
import onnxruntime as ort

DEFAULT_POLICY_CONFIG = Path(__file__).resolve().parents[1] / 'android/config/policies.json'


def checked_model(source, ident, backend, expected_digest, opts):
    source = Path(source)
    data = source.read_bytes()
    if data.startswith(b'version https://git-lfs.github.com/spec/v1\n'):
        raise ValueError(f'Policy {ident} is a Git LFS pointer; run git lfs pull')
    digest = hashlib.sha256(data).hexdigest()
    if expected_digest and expected_digest != digest:
        raise ValueError(f'Policy hash mismatch: {ident}')
    session = ort.InferenceSession(str(source), sess_options=opts, providers=['CPUExecutionProvider'])
    input_width, output_width = {'dexweave_body_v1': (157,29), 'twist2': (1432,29), 'groot_decoupled_v1': (516,15), 'sonic_encoder': (1762,64), 'sonic_decoder': (994,29)}[backend]
    rate = session.get_modelmeta().custom_metadata_map.get('g1_policy_hz', '100' if backend == 'twist2' else '50')
    if rate not in ('50', '100') or (backend != 'twist2' and rate != '50'):
        raise ValueError(f'Unsupported policy frequency: {ident}: {rate}')
    inputs, outputs = session.get_inputs(), session.get_outputs()
    if (len(inputs) != 1 or len(outputs) != 1 or inputs[0].type != 'tensor(float)'
            or outputs[0].type != 'tensor(float)' or len(inputs[0].shape) != 2
            or len(outputs[0].shape) != 2 or inputs[0].shape[1] != input_width or outputs[0].shape[1] != output_width
            or not (inputs[0].shape[0] in (1, None) or isinstance(inputs[0].shape[0], str))
            or not (outputs[0].shape[0] in (1, None) or isinstance(outputs[0].shape[0], str))):
        raise ValueError(f'Incompatible Quest policy: {ident}')
    action = session.run(None, {inputs[0].name: np.zeros((1, input_width), np.float32)})[0]
    if action.shape != (1, output_width) or not np.isfinite(action).all():
        raise ValueError(f'Invalid policy output: {ident}')
    return digest, int(rate)


def export_policies(assets, upstream, config=None):
    assets, upstream = Path(assets), Path(upstream)
    rows = [dict(id='twist2-20k', title='TWIST2: авторская 20k', note='Основная',
                 path=str(upstream / 'assets/ckpts/twist2_1017_20k.onnx')),
            dict(id='twist2-25k', title='TWIST2: авторская 25k', note='Авторский checkpoint',
                 path=str(upstream / 'assets/ckpts/twist2_1017_25k.onnx'))]
    config = config or os.environ.get('G1_POLICY_CONFIG') or DEFAULT_POLICY_CONFIG
    if config:
        config = Path(config).resolve()
        extra = json.loads(config.read_text(encoding='utf-8'))
        for row in extra:
            row = dict(row)
            for key in ('path', 'secondary_path', 'planner_path'):
                if key in row:
                    path = Path(row[key])
                    row[key] = str(path if path.is_absolute() else config.parent / path)
            rows.append(row)
    if len(rows) > 128:
        raise ValueError('Too many policies')
    ids, catalog = set(), []
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = opts.inter_op_num_threads = 1
    for i, row in enumerate(rows):
        ident = row['id']
        if (not ident or len(ident) > 80 or not all(c.isascii() and (c.isalnum() or c in '-_') for c in ident)
                or ident in ids):
            raise ValueError(f'Invalid/duplicate policy id: {ident}')
        ids.add(ident)
        for key, limit in [('title', 100), ('note', 140)]:
            if not row.get(key) or len(row[key].encode()) > limit or any(ord(c) < 32 for c in row[key]):
                raise ValueError(f'Invalid policy {key}')
        backend = row.get('backend', 'twist2')
        if backend not in ('twist2', 'groot_decoupled_v1', 'sonic_vr3_v1', 'dexweave_body_v1'):
            raise ValueError(f'Unsupported policy backend: {backend}')
        if (backend in ('groot_decoupled_v1','sonic_vr3_v1')) != bool(row.get('secondary_path')):
            raise ValueError(f'Policy secondary model does not match backend: {ident}')
        if (backend == 'sonic_vr3_v1') != bool(row.get('planner_path')):
            raise ValueError('Planner does not match backend')
        if row.get('input_patch') and not ((backend=='twist2' and row['input_patch'] in ('virtual_leg_lift10_v1','stance_narrow12_v1')) or (backend=='dexweave_body_v1' and row['input_patch']=='stance_narrow12_v1') or (backend=='groot_decoupled_v1' and row['input_patch']=='homie_stand_v1')):
            raise ValueError('Invalid source patch mode')
        if backend == 'sonic_vr3_v1':
            from scripts.sonic_assets import package
            entry={key: row[key] for key in ('id','title','note')}
            entry.update(backend=backend,policy_hz=50,legacy_policy_hz=50)
            entry.update(package(row,assets,opts,checked_model));catalog.append(entry);continue
        source = Path(row['path'])
        digest, rate = checked_model(source, ident, backend, row.get('sha256'), opts)
        filename = ('policy.onnx' if i == 0 else f'policy-{ident}.onnx') if backend in ('twist2','dexweave_body_v1') else f'policy-{ident}-balance.onnx'
        shutil.copy2(source, assets / filename)
        entry = {key: row[key] for key in ('id', 'title', 'note')}
        entry.update(file=filename, sha256=digest, policy_hz=50, legacy_policy_hz=rate, backend=backend)
        if backend == 'groot_decoupled_v1':
            secondary_digest, _ = checked_model(row['secondary_path'], ident+' walk', backend, row.get('secondary_sha256'), opts)
            secondary_file = f'policy-{ident}-walk.onnx'
            shutil.copy2(row['secondary_path'], assets / secondary_file)
            entry.update(secondary_file=secondary_file, secondary_sha256=secondary_digest)
        if row.get('input_patch'):entry['input_patch']=row['input_patch']
        catalog.append(entry)
    for scene in ('lab', 'stand', 'cup', 'push_t', 'tennis', 'rubik', 'wb_dex', 'bimanual', 'basket'):
        base = assets / ('recording_metadata.json' if scene == 'lab' else f'recording_metadata-{scene}.json')
        metadata = json.loads(base.read_text())
        for entry in catalog:
            item = deepcopy(metadata)
            item.pop('input_patch',None)
            if entry.get('input_patch'):item['input_patch']=entry['input_patch']
            item.update(policy_id=entry['id'], policy_title=entry['title'], policy_note=entry['note'],
                        policy_file=entry['file'], policy_sha256=entry['sha256'], policy_hz=entry['policy_hz'],
                        legacy_policy_hz=entry['legacy_policy_hz'], policy_backend=entry['backend'])
            for key in ('policy_secondary_file', 'policy_secondary_sha256', 'policy_planner_file', 'policy_planner_sha256', 'body_motor_profile', 'body_motor_profile_sha256'):
                item.pop(key, None)
            item['sha256'].pop('body_motor_profile.txt', None)
            item['sha256'].pop('policy.onnx', None)
            item['sha256'][entry['file']] = entry['sha256']
            if 'secondary_file' in entry:
                item.update(policy_secondary_file=entry['secondary_file'], policy_secondary_sha256=entry['secondary_sha256'])
                item['sha256'][entry['secondary_file']] = entry['secondary_sha256']
            if entry['backend']=='dexweave_body_v1':
                item['dexweave_live_reference']='absolute GMR pose; causal 50Hz backward joint velocity; no offline support projection'
                item.setdefault('limitations',[]).append('Experimental live source differs from offline support-projected evaluation; exact dynamic replay requires dexweave-reference.csv')
            if entry['backend']=='groot_decoupled_v1':
                profile_root = Path(__file__).resolve().parents[1] / 'policies/homie-v2'
                profile = json.loads((profile_root / 'provenance.json').read_text())['body_motor_profile']
                profile_path = profile_root / profile['file']
                if hashlib.sha256(profile_path.read_bytes()).hexdigest() != profile['sha256']:
                    raise ValueError('Body motor profile SHA256 mismatch')
                shutil.copy2(profile_path, assets / profile['file'])
                item.update(body_motor_profile=profile['id'], body_motor_profile_sha256=profile['sha256'])
                item['sha256'][profile['file']] = profile['sha256']
            if entry['backend']=='sonic_vr3_v1':
                proof=json.loads((Path(__file__).resolve().parents[1]/'policies/sonic-vr3/provenance.json').read_text())
                item.update(policy_planner_file=entry['planner_file'],policy_planner_sha256=entry['planner_sha256'],
                    sonic_planner_seed=11,sonic_planner_schedule='async_5hz',sonic_input='GMR FK wrists/head, autonomous planned legs',
                    body_motor_profile='sonic_b042411f_deploy_motors',body_motor_profile_sha256=proof['sha256']['sonic_motor_profile.txt'])
                item['sha256'][entry['planner_file']]=entry['planner_sha256']
                item['sha256']['sonic_motor_profile.txt']=proof['sha256']['sonic_motor_profile.txt']
                item.setdefault('limitations',[]).append('Asynchronous SONIC plans are not stored; frames support state replay, not exact dynamic restart')
            (assets / f'recording_metadata-{scene}-policy-{entry["id"]}.json').write_text(
                json.dumps(item, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        # Default metadata also explicitly identifies the baseline.
        base.write_text((assets / f'recording_metadata-{scene}-policy-twist2-20k.json').read_text(), encoding='utf-8')
    # Remove candidates left by a previous build; a fresh config is authoritative.
    keep = {row[key] for row in catalog for key in ('file', 'secondary_file', 'planner_file') if key in row}
    for old in assets.glob('policy-*.onnx'):
        if old.name not in keep:
            old.unlink()
    if not any(row['backend'] == 'groot_decoupled_v1' for row in catalog):
        (assets / 'body_motor_profile.txt').unlink(missing_ok=True)
    if not any(row['backend']=='sonic_vr3_v1' for row in catalog):
        for name in ['sonic_motor_profile.txt','sonic_observation_config.yaml','SONIC_SOURCE_LICENSE.txt','SONIC_MODEL_LICENSE.txt','SONIC_NOTICE.txt']:
            (assets/name).unlink(missing_ok=True)
    lines = []
    for row in catalog:
        values = [row[key] for key in ('id', 'file', 'sha256', 'title', 'note')]
        if row['backend'] != 'twist2' or row.get('input_patch'):
            values += [row['backend'], row.get('secondary_file',''), row.get('secondary_sha256','')]
        if row['backend']=='sonic_vr3_v1':
            values += [row['planner_file'],row['planner_sha256']]
        if row.get('input_patch'):values.append(row['input_patch'])
        lines.append(' '.join(json.dumps(value, ensure_ascii=False) for value in values))
    text = '\n'.join(lines) + '\n'
    (assets / 'policy_catalog.txt').write_text(text, encoding='utf-8')
    (assets / 'policy_catalog.json').write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('Quest policies:', ', '.join(row['id'] for row in catalog))
    return catalog
