"""Validate local Quest input/state streams without publishing participant data."""
import argparse
import csv
import json
import math
from pathlib import Path

INPUT_FIELDS = (
    'sequence,xr_time_ns,receive_ns,valid,head_flags,left_flags,right_flags,'
    'left_active,right_active,grip_left,grip_right'
).split(',') + [f'{pose}_{axis}' for pose in ('head', 'left', 'right')
               for axis in ('x', 'y', 'z', 'qw', 'qx', 'qy', 'qz')]
FRAME_FIELDS = (
    'receive_ns,sequence,step,sim_time,reset,calibrate,apply_reference,valid,'
    'grip_left,grip_right,mode,ik_error,limited'
).split(',')


def read_stream(path, fields):
    with path.open(newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != fields:
            raise ValueError(f'{path.name}: unexpected schema columns')
        rows = list(reader)
    if any(None in row or any(v is None or v == '' for v in row.values()) for row in rows):
        raise ValueError(f'{path.name}: incomplete row')
    return rows


BODY_FIELDS = ['sequence', 'time_ns', 'supported', 'valid', 'confidence', 'skeleton_version'] + [
    name for j in range(14) for name in ([f'flags_{j}'] +
    [f'{kind}_{j}_{k}' for kind in ('pose', 'rest') for k in range(7)])]
MIMIC_FIELDS = ['sequence', 'step', 'sim_time'] + [f'mimic_{k}' for k in range(35)]


def inspect(path):
    path = Path(path)
    manifest = json.loads((path / 'manifest.json').read_text())
    if manifest['schema_version'] != 1:
        raise ValueError('Unsupported schema')
    dimensions = [manifest[name] for name in ('nq', 'nv', 'nu')]
    if any(type(n) is not int or n < 0 for n in dimensions):
        raise ValueError('Invalid model dimensions')
    fields = FRAME_FIELDS + [f'reference_{i}' for i in range(29)]
    for name, size in zip(('qpos', 'qvel', 'ctrl'), dimensions):
        fields += [f'{name}_{i}' for i in range(size)]
    inputs = read_stream(path / 'input.csv', INPUT_FIELDS)
    frames = read_stream(path / 'frames.csv', fields)
    events = read_stream(path / 'events.csv', ['receive_ns', 'sequence', 'event'])
    is_gmr = manifest.get('retargeting') == 'native_gmr_meta_full_body'
    bodies = read_stream(path / 'body.csv', BODY_FIELDS) if is_gmr else []
    commands = read_stream(path / 'mimic.csv', MIMIC_FIELDS) if is_gmr else []
    if is_gmr:
        if [r['sequence'] for r in bodies] != [r['sequence'] for r in inputs]:
            raise ValueError('Body stream does not match input sequence')
        if len(commands) != len(frames) or any(any(a[k] != b[k] for k in ('sequence', 'step', 'sim_time')) for a,b in zip(commands,frames)):
            raise ValueError('Mimic stream does not match state frames')
        for body, source in zip(bodies, inputs):
            if body['valid'] not in ('0','1') or body['supported'] not in ('0','1'):
                raise ValueError('Invalid body boolean')
            if body['valid']=='1' and (body['supported']!='1' or not 0<float(body['confidence'])<=1 or int(body['time_ns'])<=0 or
                    any(int(body[f'flags_{j}']) & 3 != 3 for j in range(14))):
                raise ValueError('Valid body lacks valid skeleton')
    for row in inputs + frames + bodies + commands:
        if None in row or any(v is None or not math.isfinite(float(v)) for v in row.values()):
            raise ValueError('Incomplete/nonfinite row')
    for rows, names in ((inputs, ('valid', 'left_active', 'right_active')),
                        (frames, ('reset', 'calibrate', 'apply_reference', 'valid', 'limited'))):
        if any(row[name] not in ('0', '1') for row in rows for name in names):
            raise ValueError('Invalid boolean value')
    seqs = [int(r['sequence']) for r in inputs]
    if any(s < 0 for s in seqs) or any(b <= a for a, b in zip(seqs, seqs[1:])):
        raise ValueError('Non-increasing source sequence')
    for rows in (inputs, frames, events):
        times = [int(r['receive_ns']) for r in rows]
        if any(b < a for a, b in zip(times, times[1:])):
            raise ValueError('Non-monotonic receive timestamps')
    known = dict(zip(seqs, inputs))
    if any(int(r['sequence']) not in known for r in frames):
        raise ValueError('State references absent input')
    for row in inputs:
        if row['valid'] == '1' and (any(int(row[f'{p}_flags']) & 3 != 3 for p in ('head', 'left', 'right'))
                                    or any(row[f'{p}_active'] != '1' for p in ('left', 'right'))):
            raise ValueError('Valid input lacks valid poses or active controllers')
    body_by_sequence = {int(r['sequence']):r for r in bodies}
    calibrated = False
    replayable = 0
    for row in frames:
        source = known[int(row['sequence'])]
        if row['reset'] == '1':
            calibrated = False
        source_valid = source['valid']=='1'
        if is_gmr:
            body = body_by_sequence[int(row['sequence'])]
            source_valid = body['valid']=='1' and int(source['head_flags']) & 3 == 3 and abs(int(source['xr_time_ns'])-int(body['time_ns']))<200_000_000
        if row['valid'] == '1' and not source_valid:
            raise ValueError('Valid state references invalid input')
        if (row['calibrate'] == '1' or row['apply_reference'] == '1') and row['valid'] != '1':
            raise ValueError('Calibration/reference used invalid or stale tracking')
        if row['calibrate'] == '1':
            calibrated = True
        replayable += row['apply_reference'] == '1' and calibrated
    complete = path / 'complete.json'
    finalized = complete.exists()
    if finalized:
        counts = json.loads(complete.read_text())
        if counts.get('schema_version') != 1 or counts.get('reason') not in ('user_stop', 'app_shutdown'):
            raise ValueError('Invalid completion marker')
        if counts['input_rows'] != len(inputs) or counts['frame_rows'] != len(frames):
            raise ValueError('Completion counts differ from streams')
    source_times = [int(r['xr_time_ns']) for r in inputs]
    if any(b <= a for a, b in zip(source_times, source_times[1:])):
        raise ValueError('Non-increasing source timestamps')
    gaps_ms = [(b-a)/1e6 for a,b in zip(source_times, source_times[1:])]
    return dict(finalized=finalized, input_rows=len(inputs), state_rows=len(frames),
                invalid_input_rows=sum(r['valid'] != '1' for r in inputs),
                stale_or_invalid_control_rows=sum(r['valid'] != '1' for r in frames),
                calibration_events=sum(r['event'] == 'calibrate' for r in events),
                reset_events=sum(r['event'] == 'reset' for r in events),
                max_source_gap_ms=max(gaps_ms, default=0),
                replayable_state_rows=replayable,
                replay_ready=finalized and replayable > 0,
                warning=None if finalized else 'Interrupted/unfinalized episode; retain for diagnostics only')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('episode', type=Path)
    args = parser.parse_args()
    print(json.dumps(inspect(args.episode), indent=2))
