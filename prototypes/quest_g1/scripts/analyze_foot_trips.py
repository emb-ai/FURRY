"""Premature foot touchdowns ("trips") and toe targets below the floor.

Reads CSVs from `replay_gmr_dynamics` or `ablation_motion` (any mode). The
reference foot is rebuilt from the 35-value command the policy received, so
saved, retarget, direct and synthetic-Meta rows are scored the same way.

A trip is a physical touchdown while the commanded foot is still in swing:
lifted more than 1 cm above the other commanded foot and moving faster than
0.3 m/s. A braking trip also has more than 50 N of horizontal floor force
against the foot's travel. Toe-target columns exist only where GMR solved the
frame; a value below -5 mm asks for the toe frame under a flat grounded foot.
Trips are not falls: a robot can trip often and still complete a motion.
"""
import argparse
import json
from pathlib import Path
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT/'vendor/TWIST2/assets/g1/g1_sim2sim_29dof.xml'
# Lowest points of TWIST2's four foot pads in the ankle-roll frame.
SOLE = np.array([[-.05, .025, -.035], [-.05, -.025, -.035], [.12, .03, -.035], [.12, -.03, -.035]])
SIDES = ('l', 'r')
WARMUP_S = 1.
AIR_N, LANDING_N, AIR_ROWS = 5., 50., 3
SWING_M, SWING_SPEED, BRAKING_N = .01, .3, 50.
TOE_BELOW_M, TOE_LOOKBACK_ROWS = -.005, 10


def matrix(a, prefix, n):
    return np.column_stack([a[prefix + str(k)] for k in range(n)])


def blocks(a):
    """Applying rows, split at segment changes and time gaps."""
    rows = np.flatnonzero(a['applying'] > 0)
    if not len(rows):
        return []
    breaks = np.flatnonzero((np.diff(rows) != 1) | (np.diff(a['segment'][rows]) != 0) |
                            (abs(np.diff(a['sim_s'][rows]) - .01) > 1e-6)) + 1
    return np.split(rows, breaks)


def command_poses(cmd, dt):
    """Integrate commanded local velocity and yaw rate into reference qpos."""
    yaw = np.cumsum(cmd[:, 5])*dt
    heading = np.c_[np.cos(yaw), np.sin(yaw)]
    vx, vy = cmd[:, 0], cmd[:, 1]
    world = np.c_[heading[:, 0]*vx - heading[:, 1]*vy, heading[:, 1]*vx + heading[:, 0]*vy]
    xy = np.cumsum(world, axis=0)*dt
    quat = Rotation.from_euler('xyz', np.c_[cmd[:, 3], cmd[:, 4], yaw]).as_quat()[:, [3, 0, 1, 2]]
    return np.c_[xy, cmd[:, 2], quat, cmd[:, 6:]]


def reference_feet(model, poses, dt):
    """Commanded foot height above the other foot, and horizontal foot speed."""
    data = mujoco.MjData(model)
    feet = [model.body(side + '_ankle_roll_link').id for side in ('left', 'right')]
    low, xy = np.zeros((len(poses), 2)), np.zeros((len(poses), 2, 2))
    for n, q in enumerate(poses):
        data.qpos[:] = q
        mujoco.mj_kinematics(model, data)
        for k, body in enumerate(feet):
            points = data.xpos[body] + SOLE @ data.xmat[body].reshape(3, 3).T
            low[n, k] = points[:, 2].min()
            xy[n, k] = data.xpos[body][:2]
    height = low - low.min(axis=1, keepdims=True)
    speed = np.zeros((len(poses), 2))
    span = 5
    if len(poses) > 2*span:
        speed[span:-span] = np.linalg.norm(xy[2*span:] - xy[:-2*span], axis=2)/(2*span*dt)
    return height, speed


def touchdowns(force):
    """Rows where a foot lands after at least AIR_ROWS unloaded rows."""
    events, air = [], 0
    for n, value in enumerate(force):
        if value < AIR_N:
            air += 1
            continue
        if air >= AIR_ROWS and value >= LANDING_N:
            events.append(n)
        air = 0
    return events


def score_block(height, speed, force, brake, toe, warm_rows):
    """Counts for one continuous block; arrays are rows x (left, right)."""
    out = dict(rows=max(0, len(height) - warm_rows), touchdowns=0, trips=0, braking_trips=0,
               toe_rows=0, toe_below_rows=0, swing_toe_rows=0, swing_toe_below_rows=0,
               trips_with_toe_below=0)
    for k in range(2):
        below = np.isfinite(toe[:, k]) & (toe[:, k] < TOE_BELOW_M)
        swing = height[:, k] > SWING_M
        valid = np.isfinite(toe[warm_rows:, k])
        out['toe_rows'] += int(valid.sum())
        out['toe_below_rows'] += int(below[warm_rows:].sum())
        out['swing_toe_rows'] += int((valid & swing[warm_rows:]).sum())
        out['swing_toe_below_rows'] += int((below & swing)[warm_rows:].sum())
        for n in touchdowns(force[:, k]):
            if n < warm_rows:
                continue
            out['touchdowns'] += 1
            if height[n, k] > SWING_M and speed[n, k] > SWING_SPEED:
                out['trips'] += 1
                out['braking_trips'] += int(brake[n, k] > BRAKING_N)
                out['trips_with_toe_below'] += int(below[max(0, n - TOE_LOOKBACK_ROWS):n + 1].any())
    return out


def analyze(path, model=None):
    a = np.atleast_1d(np.genfromtxt(path, delimiter=',', names=True))
    missing = [name for name in ('floor_force_l', 'floor_brake_l', 'toe_target_l') if name not in a.dtype.names]
    if missing:
        raise SystemExit(f'{path}: CSV lacks per-foot columns {missing}; rebuild the replay harness')
    model = model or mujoco.MjModel.from_xml_path(str(MODEL))
    total = None
    for rows in blocks(a):
        b = a[rows]
        height, speed = reference_feet(model, command_poses(matrix(b, 'cmd_', 35), .01), .01)
        pair = lambda name: np.c_[b[name + '_l'], b[name + '_r']]
        counts = score_block(height, speed, pair('floor_force'), pair('floor_brake'), pair('toe_target'),
                             int(round(WARMUP_S/.01)))
        total = counts if total is None else {k: total[k] + v for k, v in counts.items()}
    if total is None or not total['rows']:
        return dict(file=str(path), seconds=0.)
    seconds = total['rows']*.01
    share = lambda part, whole: total[part]/total[whole] if total[whole] else None
    return dict(file=str(path), seconds=round(seconds, 2),
                touchdowns_per_s=round(total['touchdowns']/seconds, 3),
                trips_per_s=round(total['trips']/seconds, 3),
                braking_trips_per_s=round(total['braking_trips']/seconds, 3),
                trips=total['trips'], braking_trips=total['braking_trips'],
                toe_target_rows=total['toe_rows'],
                toe_below_share=share('toe_below_rows', 'toe_rows'),
                swing_toe_below_share=share('swing_toe_below_rows', 'swing_toe_rows'),
                trips_with_toe_below_share=(total['trips_with_toe_below']/total['trips']
                                            if total['trips'] and total['toe_rows'] else None))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('csv', nargs='+', type=Path)
    parser.add_argument('--output', type=Path, help='Write aggregate JSON here')
    args = parser.parse_args()
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    results = [analyze(path, model) for path in args.csv]
    for result in results:
        print(json.dumps(result))
    if args.output:
        args.output.write_text(json.dumps(results, indent=2) + '\n')


if __name__ == '__main__':
    main()
