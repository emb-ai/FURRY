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
    for row in inputs + frames:
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
    calibrated = False
    replayable = 0
    for row in frames:
        source = known[int(row['sequence'])]
        if row['reset'] == '1':
            calibrated = False
        if row['valid'] == '1' and source['valid'] != '1':
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
