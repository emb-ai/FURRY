"""Package author policies and optional local ONNX candidates with provenance."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np
import onnxruntime as ort


def export_policies(assets, upstream, config=None):
    assets, upstream = Path(assets), Path(upstream)
    rows = [dict(id='twist2-20k', title='TWIST2: авторская 20k', note='Основная',
                 path=str(upstream / 'assets/ckpts/twist2_1017_20k.onnx')),
            dict(id='twist2-25k', title='TWIST2: авторская 25k', note='Авторский checkpoint',
                 path=str(upstream / 'assets/ckpts/twist2_1017_25k.onnx'))]
    config = config or os.environ.get('G1_POLICY_CONFIG')
    if config:
        config = Path(config).resolve()
        extra = json.loads(config.read_text(encoding='utf-8'))
        for row in extra:
            row = dict(row)
            path = Path(row['path'])
            row['path'] = str(path if path.is_absolute() else config.parent / path)
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
        source = Path(row['path'])
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if row.get('sha256') and row['sha256'] != digest:
            raise ValueError(f'Policy hash mismatch: {ident}')
        session = ort.InferenceSession(str(source), sess_options=opts, providers=['CPUExecutionProvider'])
        inputs, outputs = session.get_inputs(), session.get_outputs()
        if (len(inputs) != 1 or len(outputs) != 1 or inputs[0].type != 'tensor(float)'
                or outputs[0].type != 'tensor(float)' or len(inputs[0].shape) != 2
                or len(outputs[0].shape) != 2 or inputs[0].shape[1] != 1432 or outputs[0].shape[1] != 29
                or not (inputs[0].shape[0] in (1, None) or isinstance(inputs[0].shape[0], str))
                or not (outputs[0].shape[0] in (1, None) or isinstance(outputs[0].shape[0], str))):
            raise ValueError(f'Incompatible Quest policy: {ident}')
        action = session.run(None, {inputs[0].name: np.zeros((1, 1432), np.float32)})[0]
        if action.shape != (1, 29) or not np.isfinite(action).all():
            raise ValueError(f'Invalid policy output: {ident}')
        filename = 'policy.onnx' if i == 0 else f'policy-{ident}.onnx'
        shutil.copy2(source, assets / filename)
        entry = {key: row[key] for key in ('id', 'title', 'note')}
        entry.update(file=filename, sha256=digest)
        catalog.append(entry)
    for scene in ('lab', 'stand', 'cup', 'push_t'):
        base = assets / ('recording_metadata.json' if scene == 'lab' else f'recording_metadata-{scene}.json')
        metadata = json.loads(base.read_text())
        for entry in catalog:
            item = deepcopy(metadata)
            item.update(policy_id=entry['id'], policy_title=entry['title'], policy_note=entry['note'],
                        policy_file=entry['file'], policy_sha256=entry['sha256'])
            item['sha256'].pop('policy.onnx', None)
            item['sha256'][entry['file']] = entry['sha256']
            (assets / f'recording_metadata-{scene}-policy-{entry["id"]}.json').write_text(
                json.dumps(item, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        # Default metadata also explicitly identifies the baseline.
        base.write_text((assets / f'recording_metadata-{scene}-policy-twist2-20k.json').read_text(), encoding='utf-8')
    # Remove candidates left by a previous build; a fresh config is authoritative.
    keep = {row['file'] for row in catalog}
    for old in assets.glob('policy-*.onnx'):
        if old.name not in keep:
            old.unlink()
    text = '\n'.join(' '.join(json.dumps(row[key], ensure_ascii=False) for key in
                                ('id', 'file', 'sha256', 'title', 'note')) for row in catalog) + '\n'
    (assets / 'policy_catalog.txt').write_text(text, encoding='utf-8')
    (assets / 'policy_catalog.json').write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('Quest policies:', ', '.join(row['id'] for row in catalog))
    return catalog
