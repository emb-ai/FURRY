"""Summarize a local GMR episode and matching runtime telemetry without uploading it.

The clocks used here are receive_ns and wall_s (both steady_clock), not XR time.
Only an incomplete final telemetry row may be discarded. Episode streams must
pass the strict validator. Meta confidence/flags do not measure tracking error.
"""
import argparse
import csv
import io
import json
from pathlib import Path

import numpy as np
from inspect_recording import inspect


def telemetry(path):
    lines = Path(path).read_text().splitlines()
    if len(lines) < 2:
        raise ValueError('Empty telemetry')
    columns = len(next(csv.reader([lines[0]])))
    bad = [n for n, line in enumerate(lines[1:], 1)
           if len(next(csv.reader([line]))) != columns]
    if bad and bad != [len(lines) - 1]:
        raise ValueError('Malformed telemetry before final row')
    if bad:
        lines.pop()
    values = np.atleast_1d(np.genfromtxt(io.StringIO('\n'.join(lines)), delimiter=',', names=True))
    if any(not np.isfinite(values[name]).all() for name in values.dtype.names):
        raise ValueError('Nonfinite telemetry')
    return values, len(bad)


def analyze(episode, runtime):
    result = inspect(episode)
    frames = np.atleast_1d(np.genfromtxt(episode / 'frames.csv', delimiter=',', names=True))
    start, end = frames['receive_ns'][[0, -1]] / 1e9
    stats, discarded = telemetry(runtime)
    # States 1 and 5 both apply the full-body reference; 5 flags a high residual.
    selected = stats[(stats['wall_s'] >= start) & (stats['wall_s'] <= end)
                     & np.isin(stats['state'], [1, 5])]
    metrics = ['cycle_ms', 'physics_ms', 'gmr_ms', 'inference_ms', 'rtf',
               'contacts', 'max_depth_mm', 'body_gap_ms', 'input_age_ms',
               'body_confidence', 'height', 'tilt', 'leg_error_deg']
    result.update(duration_s=float(end - start),
                  telemetry_incomplete_tail_rows=discarded,
                  tracking_telemetry_samples=len(selected),
                  tracking_percentiles={name: dict(zip(('min', 'median', 'p95', 'max'),
                      map(float, np.percentile(selected[name], [0, 50, 95, 100]))))
                      for name in metrics} if len(selected) else {},
                  events=list(csv.DictReader((episode / 'events.csv').open())))
    # Export relative event times rather than device clock values.
    for event in result['events']:
        event['elapsed_s'] = int(event.pop('receive_ns')) / 1e9 - start
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('episode', type=Path)
    parser.add_argument('runtime_stats', type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze(args.episode, args.runtime_stats), indent=2))
