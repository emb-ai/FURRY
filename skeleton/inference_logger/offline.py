"""Clock fit, immutable episode import, ID linkage and bounded time pairing."""
from bisect import bisect_left
from collections import defaultdict
import csv
import json
from pathlib import Path
import shutil

import numpy as np

from .storage import dumps, read_jsonl, save_json, sha256


def rows(root, name):
    path = Path(root) / (name + '.jsonl')
    return list(read_jsonl(path)) if path.exists() else []


def fit_clocks(probes, max_rtt_ms=50., max_residual_ms=5.):
    grouped = defaultdict(list)
    for row in probes:
        t1, t2, t3, t4 = (int(row[f't{i}_ns']) for i in range(1, 5))
        rtt = t4-t1-(t3-t2)
        if t1 <= t4 and t2 <= t3 and 0 <= rtt <= max_rtt_ms * 1e6:
            grouped[row['clock_segment']].append(((t2+t3)//2, (t1+t4)//2, rtt))
    result = {}
    for segment, points in grouped.items():
        points.sort()
        if len(points) < 3 or points[-1][0] == points[0][0]:
            continue
        q0, h0, _ = points[0]
        x = np.array([q-q0 for q, _, _ in points], dtype=np.float64)
        y = np.array([h-h0 for _, h, _ in points], dtype=np.float64)
        slope, intercept = np.linalg.lstsq(np.column_stack((x/1e9, np.ones(len(x)))), y/1e9, rcond=None)[0]
        intercept_ns = round(intercept*1e9)
        residual = y-(slope*x+intercept_ns)
        valid = abs(slope-1) < .001 and float(np.max(np.abs(residual))) <= max_residual_ms*1e6
        result[segment] = {'quest_origin_ns': str(q0), 'host_origin_ns': str(h0+intercept_ns),
            'scale': float(slope), 'drift_ppm': float((slope-1)*1e6),
            'quest_start_ns': str(points[0][0]), 'quest_end_ns': str(points[-1][0]),
            'probe_quest_ns': [str(q) for q, _, _ in points],
            'max_residual_ns': float(np.max(np.abs(residual))),
            'rtt_median_ns': float(np.median([p[2] for p in points])),
            'samples': len(points), 'valid': bool(valid),
            'method': 'four timestamps; affine midpoint fit; assumes symmetric delay',
            'uncertainty_note': 'residual and RTT are diagnostics, not guaranteed error bounds'}
    return result


def mapped(t, model, max_probe_age_ns=3_000_000_000):
    t = int(t)
    if not model or not model['valid']:
        return None
    if min(abs(t-int(p)) for p in model['probe_quest_ns']) > max_probe_age_ns:
        return None
    return int(model['host_origin_ns']) + round((t-int(model['quest_origin_ns']))*model['scale'])


def import_episode(session, source):
    session, source = Path(session), Path(source)
    sid = json.loads((session/'session.json').read_text(encoding='utf-8'))['session_id']
    camera = rows(source, 'camera')
    if not any(r.get('payload', {}).get('session_id') == sid for r in camera):
        raise ValueError('Episode has no camera payload with this session_id; refusing a guessed association')
    if not (source/'complete.json').exists():
        raise ValueError('Quest episode is not finalized (complete.json missing)')
    dest = session/'quest_episodes'/source.name
    if dest.resolve().is_relative_to(source.resolve()):
        raise ValueError('Import destination must not be inside the source episode')
    if dest.exists():
        raise FileExistsError(dest)
    dest.parent.mkdir(exist_ok=True)
    shutil.copytree(source, dest)
    hashes = {p.relative_to(source).as_posix(): sha256(p) for p in source.rglob('*') if p.is_file()}
    for relative, checksum in hashes.items():
        if sha256(dest/relative) != checksum:
            raise IOError('Episode copy checksum mismatch: '+relative)
    save_json(dest/'import_provenance.json', {'session_id': sid, 'source_name': source.name,
                                            'checksums': hashes})
    return dest


def validate(root):
    root = Path(root)
    errors, warnings = [], []
    if not (root/'complete.json').exists():
        errors.append('Session incomplete or failed')
    try:
        manifest = json.loads((root/'session.json').read_text(encoding='utf-8'))
        checksums = json.loads((root/'checksums.json').read_text(encoding='utf-8'))
        for relative, checksum in checksums.items():
            path = (root/relative).resolve()
            if not path.is_relative_to(root.resolve()) or not path.is_file() or sha256(path) != checksum:
                errors.append('Checksum mismatch: '+relative)
        for imported in (root/'quest_episodes').glob('*/import_provenance.json'):
            provenance = json.loads(imported.read_text(encoding='utf-8'))
            if provenance['session_id'] != manifest['session_id']:
                errors.append('Imported episode session mismatch')
            for relative, checksum in provenance['checksums'].items():
                path = (imported.parent/relative).resolve()
                if not path.is_relative_to(imported.parent.resolve()) or not path.is_file() or sha256(path) != checksum:
                    errors.append('Imported episode checksum mismatch: '+relative)
        streams = {p.stem: list(read_jsonl(p)) for p in root.glob('*.jsonl')}
        for name, stream in streams.items():
            for row in stream:
                if row.get('schema_version') != 1 or row.get('session_id') != manifest['session_id']:
                    errors.append('Wrong schema/session: '+name)
        frames = streams.get('frames', [])
        ids = [r['frame_id'] for r in frames]
        if len(ids) != len(set(ids)):
            errors.append('Duplicate frame IDs')
        known = set(ids)
        for r in streams.get('skeleton', []):
            if r['frame_id'] not in known:
                errors.append('Skeleton references missing frame '+r['frame_id'])
        times = [int(r['acquisition_monotonic_ns']) for r in frames]
        if any(b <= a for a, b in zip(times, times[1:])):
            errors.append('Nonmonotonic capture acquisition times')
        if not frames:
            warnings.append('No camera frames')
        clocks = fit_clocks(streams.get('sync', []))
        if not any(m['valid'] for m in clocks.values()):
            warnings.append('No valid four-timestamp clock mapping; time alignment unavailable')
        if not streams.get('quest'):
            warnings.append('No Quest uplink')
        if not (root/'quest_episodes').exists():
            warnings.append('No local Quest episode imported: body/reference/robot streams unavailable')
        if (root/'raw.bag').exists() and not (root/'reports'/'bag_validation.json').exists():
            warnings.append('Raw bag has not been verified by full playback')
        elif (root/'reports'/'bag_validation.json').exists():
            if not json.loads((root/'reports'/'bag_validation.json').read_text())['valid']:
                errors.append('Raw bag playback validation failed')
        if (root/'complete.json').exists():
            expected = json.loads((root/'complete.json').read_text(encoding='utf-8'))['counts']
            for name, count in expected.items():
                if len(streams.get(name, [])) != count:
                    errors.append('Row count mismatch: '+name)
    except (OSError, ValueError, KeyError, TypeError) as e:
        errors.append(str(e))
        streams, times, clocks = {}, [], {}
    return {'schema_version': 1, 'valid': not errors, 'errors': errors, 'warnings': warnings,
        'counts': {k: len(v) for k, v in streams.items()}, 'clock_models': clocks,
        'duration_s': (times[-1]-times[0])/1e9 if len(times)>1 else 0.,
        'hardware_verified': False}


def export(root, output, max_gap_ms=50.):
    root, output = Path(root), Path(output)
    qc = validate(root)
    if not qc['valid']:
        raise ValueError('Session validation failed: '+str(qc['errors']))
    clocks = qc['clock_models']
    if not any(m['valid'] for m in clocks.values()):
        raise ValueError('No valid clock mapping. Install the Quest sync patch and record overlapping streams.')
    output.mkdir(parents=True, exist_ok=False)
    save_json(output/'clock_segments.json', clocks)
    skeletons = rows(root, 'skeleton')
    camera_by_id = {r['frame_id']: r for r in skeletons}
    quest = rows(root, 'quest')
    samples = []
    for row in quest:
        p = row['payload']
        t = p.get('quest_monotonic_ns')
        if t is not None:
            h = mapped(t, clocks.get(row['clock_segment']))
            if h is not None:
                samples.append((h, row))
    samples.sort(key=lambda x: x[0])
    sample_times = [t for t, _ in samples]
    paired = 0
    with (output/'acquisition_pairs.jsonl').open('x', encoding='utf-8') as f:
        for camera in skeletons:
            t = int(camera['acquisition_monotonic_ns'])
            i = bisect_left(sample_times, t)
            candidates = [j for j in (i-1, i) if 0 <= j < len(samples)]
            chosen = min(candidates, key=lambda j: abs(sample_times[j]-t)) if candidates else None
            valid = chosen is not None and abs(sample_times[chosen]-t) <= max_gap_ms*1e6
            row = samples[chosen][1] if valid else None
            paired += int(valid)
            f.write(dumps({'frame_id': camera['frame_id'], 'camera': camera,
                'quest': row, 'valid': valid, 'delta_ns': str(sample_times[chosen]-t) if valid else None,
                'time_semantics': 'camera host acquisition vs Quest tracking submit; NOT exposure vs OpenXR sample',
                'interpolated': False})+'\n')
    # Exact source links for all episode streams. Repeated robot steps are retained.
    linked = 0
    with (output/'episode_links.jsonl').open('x', encoding='utf-8') as out:
        for episode in sorted((root/'quest_episodes').glob('*')):
            associations = defaultdict(set)
            for r in rows(episode, 'camera'):
                fid = r.get('payload', {}).get('frame_id')
                if fid in camera_by_id:
                    associations[str(r['input_sequence'])].add(fid)
            for name in ('input', 'body', 'frames', 'mimic', 'events'):
                path = episode/(name+'.csv')
                if not path.exists():
                    continue
                with path.open(encoding='utf-8-sig', newline='') as source:
                    for number, row in enumerate(csv.DictReader(source), 2):
                        ids = sorted(associations.get(row.get('sequence'), []))
                        mapped_times = []
                        if row.get('receive_ns'):
                            for segment, model in clocks.items():
                                t = mapped(row['receive_ns'], model)
                                if t is not None:
                                    mapped_times.append((segment, t))
                        linked += bool(ids)
                        out.write(dumps({'episode': episode.name, 'stream': name,
                            'source_row': number, 'camera_frame_ids_observed_for_input': ids,
                            'runtime_link_valid': bool(ids), 'values': row,
                            'host_receive_ns': str(mapped_times[0][1]) if len(mapped_times)==1 else None,
                            'clock_segment': mapped_times[0][0] if len(mapped_times)==1 else None,
                            'time_mapping_valid': len(mapped_times)==1,
                            'note': 'observed camera payload is not proof of fusion acceptance/use'})+'\n')
            # Keep every additional binary/JSON stream discoverable, without inventing a decoder.
    provenance = {p.relative_to(root).as_posix(): sha256(p) for p in root.rglob('*')
                  if p.is_file() and not p.resolve().is_relative_to(output.resolve())
                  and 'reports' not in p.relative_to(root).parts and 'derived' not in p.relative_to(root).parts}
    report = {'paired_frames': paired, 'camera_frames': len(skeletons), 'episode_rows_with_camera_link': linked,
        'max_gap_ms': max_gap_ms, 'source_checksums': provenance,
        'limitations': ['No camera-to-STAGE transform applied',
                       'OpenXR time is preserved, not assumed equal to steady clock',
                       'acquisition_pairs are approximate acquisition alignment, not clean reference',
                       'ctrl is actuator control, not raw actor output',
                       'Missing policy tensors cannot be recovered from existing episodes']}
    save_json(output/'report.json', report)
    return report


def verify_bag(root):
    import pyrealsense2 as rs
    root = Path(root)
    expected = {r['color_frame_number'] for r in rows(root, 'frames')}
    found = set()
    pipeline, cfg = rs.pipeline(), rs.config()
    cfg.enable_device_from_file(str((root/'raw.bag').resolve()), repeat_playback=False)
    profile = pipeline.start(cfg)
    playback = profile.get_device().as_playback()
    playback.set_real_time(False)
    try:
        while True:
            try:
                frame = pipeline.wait_for_frames(5000).get_color_frame()
                if frame:
                    found.add(frame.get_frame_number())
            except RuntimeError:
                if playback.current_status() != rs.playback_status.stopped:
                    raise
                break
    finally:
        pipeline.stop()
    report = {'valid': bool(found) and expected <= found, 'indexed_frames': len(expected),
              'decoded_color_frames': len(found), 'missing_frame_numbers': sorted(expected-found)}
    (root/'reports').mkdir(exist_ok=True)
    save_json(root/'reports'/'bag_validation.json', report)
    return report
