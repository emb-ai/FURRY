"""Validate local Quest input/state streams without publishing participant data."""
import argparse
import csv
import json
import math
from pathlib import Path


def inspect(path):
    path = Path(path)
    manifest = json.loads((path / 'manifest.json').read_text())
    if manifest['schema_version'] != 1:
        raise ValueError('Unsupported schema')
    with (path / 'input.csv').open() as f:
        inputs = list(csv.DictReader(f))
    with (path / 'frames.csv').open() as f:
        frames = list(csv.DictReader(f))
    with (path / 'events.csv').open() as f:
        events = list(csv.DictReader(f))
    for row in inputs + frames:
        if None in row or any(v is None or not math.isfinite(float(v)) for v in row.values()):
            raise ValueError('Incomplete/nonfinite row')
    seqs = [int(r['sequence']) for r in inputs]
    if any(b <= a for a, b in zip(seqs, seqs[1:])):
        raise ValueError('Non-increasing source sequence')
    for rows in (inputs, frames, events):
        times = [int(r['receive_ns']) for r in rows]
        if any(b < a for a, b in zip(times, times[1:])):
            raise ValueError('Non-monotonic receive timestamps')
    known = set(seqs)
    if any(int(r['sequence']) not in known for r in frames):
        raise ValueError('State references absent input')
    expected = 42 + manifest['nq'] + manifest['nv'] + manifest['nu']
    if any(len(r) != expected for r in frames):
        raise ValueError('State dimensions differ from manifest')
    complete = path / 'complete.json'
    finalized = complete.exists()
    if finalized:
        counts = json.loads(complete.read_text())
        if counts['input_rows'] != len(inputs) or counts['frame_rows'] != len(frames):
            raise ValueError('Completion counts differ from streams')
    source_times = [int(r['xr_time_ns']) for r in inputs]
    gaps_ms = [(b-a)/1e6 for a,b in zip(source_times, source_times[1:])]
    return dict(finalized=finalized, input_rows=len(inputs), state_rows=len(frames),
                invalid_input_rows=sum(r['valid'] != '1' for r in inputs),
                stale_or_invalid_control_rows=sum(r['valid'] != '1' for r in frames),
                calibration_events=sum(r['event'] == 'calibrate' for r in events),
                reset_events=sum(r['event'] == 'reset' for r in events),
                max_source_gap_ms=max(gaps_ms, default=0),
                replay_ready=any(r['calibrate'] == '1' for r in frames),
                warning=None if finalized else 'Interrupted/unfinalized episode; retain for diagnostics only')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('episode', type=Path)
    args = parser.parse_args()
    print(json.dumps(inspect(args.episode), indent=2))
