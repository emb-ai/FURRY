"""Root tracking of a replay_meta_skeleton CSV against the GMR reference root, per window.

usage: root.py variant.csv
The robot and reference frames differ by a constant offset, so drift is the
change of the XY offset since the end of the 1 s warm-up. Per window: mean
drift, robot path length / reference path length (sampled every 0.5 s) and the
absolute heading change difference at the window end. Prints medians over windows.
"""
import sys, json
import numpy as np

COLS = ['sim_s', 'segment', 'q0', 'q1', 'ref0', 'ref1', 'q3', 'q6', 'ref3', 'ref6']
WARM = 100  # rows at 100 Hz


def yaw(w, z):  # heading of a yaw-dominant quaternion (MuJoCo order w, x, y, z)
    return 2 * np.arctan2(z, w)


def windows(path):
    with open(path) as f:
        header = f.readline().strip().split(',')
    a = np.loadtxt(path, delimiter=',', skiprows=1, usecols=[header.index(c) for c in COLS], ndmin=2)
    cut = np.flatnonzero((np.diff(a[:, 0]) < 0) | (np.diff(a[:, 1]) != 0)) + 1
    for b in np.split(a, cut):
        b = b[WARM:]
        b = b[np.isfinite(b[:, 4])]
        if len(b) >= 500:
            yield b


def run(path):
    drift, ratio, heading = [], [], []
    length = lambda xy: np.linalg.norm(np.diff(xy[::50], axis=0), axis=1).sum()
    for b in windows(path):
        d = b[:, 2:4] - b[:, 4:6]
        drift.append(np.linalg.norm(d - d[0], axis=1).mean())
        ref = length(b[:, 4:6])
        if ref > 1:
            ratio.append(length(b[:, 2:4]) / ref)
        h = yaw(b[:, 6], b[:, 7]) - yaw(b[:, 8], b[:, 9])
        heading.append(abs(np.degrees(np.angle(np.exp(1j * (h[-1] - h[0]))))))
    med = lambda x, n: round(float(np.median(x)), n)
    return dict(file=path.split('/')[-1], windows=len(drift), drift_m=med(drift, 2),
                path_ratio=med(ratio, 2), heading_err_deg=med(heading, 1))


if __name__ == '__main__':
    print(json.dumps(run(sys.argv[1])))
