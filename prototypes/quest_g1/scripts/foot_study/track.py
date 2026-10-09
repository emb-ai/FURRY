"""Tracking and trip metrics for replay_meta_skeleton CSVs.

usage: track.py variant.csv [source.csv]
source.csv (default: variant) supplies the unmodified reference command used to
classify touchdowns; rows are aligned by sim_s (identical windows and timing).
A swing is a source-reference foot > 1 cm above the other foot and moving
> 0.3 m/s for at least 80 ms. Physical clearance uses the four TWIST2 pad
points in every variant, so it is comparable across foot geometries.
"""
import sys, json
from pathlib import Path
import numpy as np
import mujoco
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.analyze_foot_trips import (MODEL, SOLE, blocks, matrix, command_poses, reference_feet,
                                        touchdowns, WARMUP_S, SWING_M, SWING_SPEED, BRAKING_N)

model = mujoco.MjModel.from_xml_path(str(MODEL))
data = mujoco.MjData(model)
FEET = [model.body(s + '_ankle_roll_link').id for s in ('left', 'right')]
# GMR joint order per leg: hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll; right leg +6.
LEG = {'hip_pitch': 0, 'hip_roll': 1, 'knee': 3, 'ankle_pitch': 4}


def physical_feet(q):
    low, xy = np.zeros((len(q), 2)), np.zeros((len(q), 2, 2))
    for n, row in enumerate(q):
        data.qpos[:] = row
        mujoco.mj_kinematics(model, data)
        for k, b in enumerate(FEET):
            pts = data.xpos[b] + SOLE @ data.xmat[b].reshape(3, 3).T
            low[n, k] = pts[:, 2].min()
            xy[n, k] = data.xpos[b][:2]
    speed = np.zeros((len(q), 2))
    speed[1:] = np.linalg.norm(np.diff(xy, axis=0), axis=2) / .01
    return low, speed


def lag_ms(ref, phys, max_lag=40):
    ref, phys = ref - ref.mean(), phys - phys.mean()
    scores = [np.dot(ref[:len(ref) - L], phys[L:]) / (len(ref) - L) for L in range(max_lag + 1)]
    return int(np.argmax(scores)) * 10


def empty():
    return dict(rows=0, touchdowns=0, trips=0, braking_trips=0, trips_vs_command=0, scuffs=0,
                braking_scuffs=0, swings=[], err={k: [] for k in LEG}, swing_err={k: [] for k in LEG}, lags=[])


def accumulate(T, b, src):
    warm = int(round(WARMUP_S / .01))
    cmd = matrix(b, 'cmd_', 35)
    ch, cs = reference_feet(model, command_poses(cmd, .01), .01)
    sh, ss = reference_feet(model, command_poses(src, .01), .01)
    q = matrix(b, 'q', 36)
    ph, ps = physical_feet(q)
    force = np.c_[b['floor_force_l'], b['floor_force_r']]
    brake = np.c_[b['floor_brake_l'], b['floor_brake_r']]
    T['rows'] += len(b) - warm
    for k in range(2):
        for n in touchdowns(force[:, k]):
            if n < warm:
                continue
            T['touchdowns'] += 1
            if sh[n, k] > SWING_M and ss[n, k] > SWING_SPEED:
                T['trips'] += 1
                T['braking_trips'] += int(brake[n, k] > BRAKING_N)
            if ch[n, k] > SWING_M and cs[n, k] > SWING_SPEED:
                T['trips_vs_command'] += 1
            if ps[max(0, n - 3):n, k].mean() > SWING_SPEED:
                T['scuffs'] += 1
                T['braking_scuffs'] += int(brake[n, k] > BRAKING_N)
        on = (sh[:, k] > SWING_M) & (ss[:, k] > SWING_SPEED)
        on[:warm] = False
        edges = np.flatnonzero(np.diff(np.r_[0, on.astype(int), 0]))
        for i0, i1 in zip(edges[::2], edges[1::2]):
            if i1 - i0 < 8:
                continue
            T['swings'].append((sh[i0:i1, k].max(), ph[i0:i1, k].max(), float((force[i0:i1, k] > 5).mean()),
                                ch[i0:i1, k].max()))
            for name, j in LEG.items():
                j += 6 * k
                T['swing_err'][name].append(q[i0:i1, 7 + j] - cmd[i0:i1, 6 + j])
    for name, j in LEG.items():
        for jj in (j, j + 6):
            T['err'][name].append(q[warm:, 7 + jj] - cmd[warm:, 6 + jj])
    if len(b) > 300:
        T['lags'].append(lag_ms(cmd[warm:, 9], q[warm:, 10]))


def summary(T):
    sec = T['rows'] * .01
    sw = np.array(T['swings']) if T['swings'] else np.zeros((0, 4))
    rms = lambda xs: round(float(np.sqrt(np.mean(np.concatenate(xs) ** 2))), 3) if xs else None
    mean = lambda xs: round(float(np.mean(np.concatenate(xs))), 3) if xs else None
    med = lambda col: round(float(np.median(sw[:, col]) * 100), 2) if len(sw) else None
    r = lambda x: round(x, 3)
    return dict(seconds=round(sec, 1), touchdowns_per_s=r(T['touchdowns'] / sec),
                trips=T['trips'], trips_per_s=r(T['trips'] / sec), braking_trips_per_s=r(T['braking_trips'] / sec),
                trips_vs_command_per_s=r(T['trips_vs_command'] / sec),
                scuffs_per_s=r(T['scuffs'] / sec), braking_scuffs_per_s=r(T['braking_scuffs'] / sec),
                swings=len(sw), ref_peak_median_cm=med(0), cmd_peak_median_cm=med(3), phys_peak_median_cm=med(1),
                swings_phys_below_1cm=r(float((sw[:, 1] < .01).mean())) if len(sw) else None,
                swing_floor_contact_share=r(float(sw[:, 2].mean())) if len(sw) else None,
                rmse_rad={k: rms(v) for k, v in T['err'].items()},
                swing_rmse_rad={k: rms(v) for k, v in T['swing_err'].items()},
                swing_bias_rad={k: mean(v) for k, v in T['swing_err'].items()},
                knee_lag_ms_median=float(np.median(T['lags'])) if T['lags'] else None)


COLUMNS = (['sim_s', 'stage', 'segment', 'applying'] + ['cmd_%d' % k for k in range(35)] + ['q%d' % k for k in range(36)] +
           ['floor_force_l', 'floor_force_r', 'floor_brake_l', 'floor_brake_r'])


def load(path):
    # Only the needed columns: the full CSVs are ~270 MB and genfromtxt needs gigabytes.
    with open(path) as f:
        header = f.readline().strip().split(',')
    cols = [header.index(c) for c in COLUMNS]
    raw = np.loadtxt(path, delimiter=',', skiprows=1, usecols=cols, ndmin=2)
    out = np.empty(len(raw), dtype=[(c, 'f8') for c in COLUMNS])
    for k, c in enumerate(COLUMNS):
        out[c] = raw[:, k]
    return out


def run(path, source_path):
    a = load(path)
    s = a if source_path == path else load(source_path)
    src_index = {round(t, 2): i for i, t in enumerate(s['sim_s'])}
    total, stages = empty(), {}
    for rows in blocks(a):
        b = a[rows]
        idx = np.array([src_index.get(round(t, 2), -1) for t in b['sim_s']])
        b, idx = b[idx >= 0], idx[idx >= 0]
        if len(b) <= int(WARMUP_S / .01) + 50:
            continue
        src = matrix(s[idx], 'cmd_', 35)
        stage = int(b['stage'][0])
        accumulate(total, b, src)
        accumulate(stages.setdefault(stage, empty()), b, src)
    out = summary(total)
    out['file'] = path.split('/')[-1]
    out['stages'] = {k: {f: v for f, v in summary(T).items() if f in
                         ('seconds', 'trips_per_s', 'scuffs_per_s', 'ref_peak_median_cm', 'phys_peak_median_cm', 'swings')}
                     for k, T in sorted(stages.items())}
    return out


if __name__ == '__main__':
    print(json.dumps(run(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else sys.argv[1])))
