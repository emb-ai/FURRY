"""Inspect a guided human capture without treating frozen robot states as motion.

Usage: .venv/bin/python scripts/inspect_capture.py EPISODE_DIRECTORY
The test split must remain held out. A complete protocol is not a guarantee of
accurate Meta leg estimation or a ready-to-train motion file.
"""
import argparse
import csv
import itertools
import json
import math
from pathlib import Path


def inspect_capture(path):
    path = Path(path)
    plan = json.loads((path/'capture_plan.json').read_text())
    if plan.get('protocol') not in ('quest_virtual_legs_v1','quest_virtual_legs_v2') or plan.get('split') not in ('train', 'test'):
        raise ValueError('Unknown capture protocol or split')
    count = 15 if plan['split'] == 'train' else 5
    if len(plan['stages']) != count or plan['target_seconds'] != count*60:
        raise ValueError('Unexpected plan duration')
    summary_path = path/'capture_summary.json'
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else None
    known, eligible, source_valid, previous = set(), set(), 0, -1
    with (path/'input.csv').open() as fi, (path/'body.csv').open() as fb:
        for inp, body in itertools.zip_longest(csv.DictReader(fi), csv.DictReader(fb)):
            if inp is None or body is None or inp['sequence'] != body['sequence']:
                raise ValueError('Input/body streams differ')
            seq = int(inp['sequence'])
            if seq <= previous:
                raise ValueError('Source sequences are not increasing')
            previous = seq; known.add(seq)
            source_valid += body['valid'] == '1'
            if body['valid'] == '1' and int(inp['head_flags']) & 3 == 3 and abs(int(inp['xr_time_ns'])-int(body['time_ns'])) < 200_000_000:
                eligible.add(seq)
    total, last_time, rows = 0., -1, 0
    stages_seen = set()
    with (path/'capture_timeline.csv').open() as f:
        for row in csv.DictReader(f):
            rows += 1
            seq, stamp, stage, status = (int(row[k]) for k in ('sequence','receive_ns','stage','status'))
            current, stage_time = float(row['accepted_seconds']), float(row['stage_seconds'])
            if seq not in known or stamp < last_time or not 0 <= stage < count or not 0 <= status <= 6:
                raise ValueError('Invalid timeline identity, clock or stage')
            if row['source_valid'] == '1' and seq not in eligible:
                raise ValueError('Timeline claims valid tracking for an invalid source')
            if not math.isfinite(current+stage_time) or not total <= current <= count*60 or not 0 <= stage_time <= 60:
                raise ValueError('Invalid accepted duration')
            if current > total:
                if any(row[k] != '1' for k in ('focused','source_valid','calibrated')) or status not in (4,5,6):
                    raise ValueError('Time credited while capture was ineligible')
                if current-total > .100001:
                    raise ValueError('A long stall was counted')
            total, last_time = current, stamp
            stages_seen.add(stage)
    finalized = (path/'complete.json').exists()
    if finalized:
        marker = json.loads((path/'complete.json').read_text())
        if marker.get('schema_version') != 1 or marker.get('reason') not in ('user_stop','app_shutdown'):
            raise ValueError('Invalid completion marker')
        if marker.get('input_rows') != len(known) or marker.get('frame_rows') != rows:
            raise ValueError('Completion counts differ from capture streams')
    if summary:
        times = summary['stage_seconds']
        if summary['split'] != plan['split'] or summary['target_seconds'] != count*60 or len(times) != count:
            raise ValueError('Summary does not match plan')
        if any(not math.isfinite(v) or not 0 <= v <= 60 for v in times):
            raise ValueError('Invalid per-stage times')
        if abs(sum(times)-summary['accepted_seconds']) > 1e-6 or abs(total-summary['accepted_seconds']) > 1e-6:
            raise ValueError('Summary duration differs from timeline')
        if summary['completed'] and (total != count*60 or len(stages_seen) != count):
            raise ValueError('Incomplete timeline labelled complete')
    return {'protocol': plan['protocol'], 'split': plan['split'], 'finalized': finalized,
            'protocol_completed': bool(finalized and summary and summary['completed']),
            'accepted_seconds': total, 'target_seconds': count*60,
            'source_rows': len(known), 'valid_body_rows': source_valid, 'timeline_rows': rows,
            'capture_kind': 'human_skeleton_only',
            'use': 'held-out evaluation only' if plan['split'] == 'test' else 'candidate human motions for curation and retargeting',
            'robot_frames': 'frozen calibration snapshots; not training motion targets'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('episode', type=Path)
    args = parser.parse_args()
    print(json.dumps(inspect_capture(args.episode), indent=2))
