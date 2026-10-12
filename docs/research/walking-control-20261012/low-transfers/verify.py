"""Verify the published compact evidence without private recordings or MuJoCo."""
import csv
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
report = json.loads((HERE / 'audit.json').read_text())
for name, digest in report['artifacts_sha256'].items():
    assert hashlib.sha256((HERE / name).read_bytes()).hexdigest() == digest, name
with (HERE / 'low-transfer-22.csv').open() as stream:
    rows = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(stream)]
assert len(rows) == 22
assert len({row['source_sequence'] for row in rows}) == 22
assert all(b['source_elapsed_s'] > a['source_elapsed_s'] for a, b in zip(rows, rows[1:]))
assert all(math.isfinite(v) for row in rows for v in row.values())
for row in rows:
    expected_lift = max(0, -min(row['legacy_pre_sole_left_m'], row['legacy_pre_sole_right_m']))
    assert abs(expected_lift - row['legacy_global_lift_m']) < 1e-12
    for side in ['left', 'right']:
        assert abs(row[f'legacy_pre_sole_{side}_m'] + expected_lift -
                   row[f'legacy_target_sole_{side}_m']) < 1e-12
    assert abs(row['legacy_target_sole_left_m']) < 1e-12
    assert row['legacy_phase'] == -1
    assert row['legacy_physical_floor_force_left_n'] == 0
    assert row['root_morph_physical_floor_force_left_n'] == 0
sample = next(row for row in rows if row['source_sequence'] == 26105)
assert abs(sample['root_morph_reference_sole_left_m'] - .02235602674295534) < 1e-12
assert .0009 < sample['legacy_reference_sole_left_m'] < .0016
last = rows[-1]
left = math.hypot(last['raw_ball_left_dx_m'], last['raw_ball_left_dz_m'])
right = math.hypot(last['raw_ball_right_dx_m'], last['raw_ball_right_dz_m'])
assert abs(left - .4867935234118519) < 1e-12
assert abs(right - .01292352537666845) < 1e-12
for group in report['coverage'].values():
    if isinstance(group, dict):
        assert group['stance'] + group['swing'] + group['unknown'] == group['total']
print(json.dumps(dict(status='PASS',samples=len(rows),left_ball_displacement_m=left,
                      opposite_ball_displacement_m=right,
                      window_swing_fraction=112/26400,fullclip_swing_fraction=510/142668), indent=2))
