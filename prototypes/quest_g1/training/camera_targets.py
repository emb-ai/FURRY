"""Offline-cleaned camera legs for reward targets (fused body text in, out).

Inputs come from camera_commands: `{prefix}.fused.txt/.csv` (live fusion
replayed) and `{prefix}.meta.fused.txt/.csv` (same ticks, packets "-").
The cleaned body goes back through camera_commands as a body override.

Measured on ep5-ep9 (2026-10-13):
- Only the shin offset is a stable per-session bias (a few degrees, up to
  ~17); removing it cut held-out quiet-stance error 9.1 -> 4.6 deg.
  Thigh "bias" is wander, and correcting it made held-out stance worse.
- Camera error is mostly depth wander at 1-4 Hz. Low-ankle travel is ~97%
  along the depth axis, so low ankles are smoothed with a velocity penalty.
  Robot stance-foot skating p90 drops 47-57 -> 24-27 cm/s (Meta 20-28),
  and hip/knee motion amplitude is kept.
One session (camera unmoved) has one shin bias. Per-episode estimates from
short or busy episodes swing by tens of degrees (2026-10-14, ep12: right shin
-12..+39 deg; a raised leg under still Meta legs counts as "quiet"), so
--session pools the quiet-stance frames of all listed episode prefixes and
takes the per-frame median; --shin passes that printed value directly.
Usage: camera_targets.py prefix out.txt [--session PREFIX ... | --shin LP LR RP RR (deg)]
"""
import argparse
import numpy as np
from types import SimpleNamespace
from scipy.linalg import solve_banded
from scipy.ndimage import binary_closing, binary_opening, percentile_filter
from scipy.signal import butter, sosfiltfilt
from scipy.spatial.transform import Rotation as R

HIP, KNEE, ANKLE = (2, 3), (4, 5), (6, 7)
UP = np.array([0., 1, 0])


def load(prefix):
    a = np.loadtxt(prefix + '.fused.txt')
    with open(prefix + '.fused.csv') as f:
        names = f.readline().strip().split(',')
    v = np.loadtxt(prefix + '.fused.csv', delimiter=',', skiprows=1, ndmin=2)
    return a, SimpleNamespace(**{n: v[:, i] for i, n in enumerate(names)})


def pos(a, j):
    return a[:, 11 + 14 * j: 14 + 14 * j]


def times(a):
    # Body timestamps repeat when the runtime re-delivers a frame.
    t = a[:, 2] * 1e-9
    return t - t[0] + np.arange(len(t)) * 1e-9


def frame(a):
    left = pos(a, 2) - pos(a, 3); left[:, 1] = 0
    left /= np.linalg.norm(left, axis=1, keepdims=True)
    return left, np.cross(left, UP)


def angles(a, left, fwd):
    """[T, 2 legs, 2 bones, 2 (pitch, roll)] radians; pitch>0 points forward."""
    out = np.empty((len(a), 2, 2, 2))
    for s in range(2):
        for b, (p, q) in enumerate(((HIP[s], KNEE[s]), (KNEE[s], ANKLE[s]))):
            d = pos(a, q) - pos(a, p); down = -d[:, 1]
            out[:, s, b, 0] = np.arctan2((d * fwd).sum(1), down)
            out[:, s, b, 1] = np.arctan2((d * left).sum(1), down)
    return out


def quiet(meta, t, speed=.05, win=.5):
    """Meta pelvis and both ankles slower than `speed` m/s, averaged over win s."""
    v = np.zeros(len(meta))
    for j in (0, 6, 7):
        v = np.maximum(v, np.minimum(np.linalg.norm(np.gradient(pos(meta, j), t, axis=0), axis=1), 10))
    k = max(1, int(win / np.median(np.diff(t))))
    return np.convolve(v, np.ones(k) / k, 'same') < speed


def swing(a, b):
    a = a / np.linalg.norm(a, axis=1, keepdims=True); b = b / np.linalg.norm(b, axis=1, keepdims=True)
    axis = np.cross(a, b); s = np.linalg.norm(axis, axis=1)
    return R.from_rotvec(axis / np.maximum(s, 1e-12)[:, None] * np.arctan2(s, (a * b).sum(1))[:, None])


def set_leg(out, s, hip, knee, ankle):
    """Move knee/ankle positions and swing the hip/knee orientations to match."""
    for jq, before, after in ((HIP[s], pos(out, KNEE[s]) - hip, knee - hip),
                              (KNEE[s], pos(out, ANKLE[s]) - pos(out, KNEE[s]), ankle - knee)):
        qa = 11 + 14 * jq + 3
        r = swing(before, after) * R.from_quat(out[:, qa:qa + 4][:, [1, 2, 3, 0]])
        out[:, qa:qa + 4] = r.as_quat()[:, [3, 0, 1, 2]]
    out[:, 11 + 14 * KNEE[s]:14 + 14 * KNEE[s]] = knee
    out[:, 11 + 14 * ANKLE[s]:14 + 14 * ANKLE[s]] = ankle


def quiet_shin(cam, meta, st, t):
    """Camera minus Meta shin [pitch, roll] on quiet camera-leg stance, [N, 2 legs, 2]."""
    left, fwd = frame(meta)
    on = (st.camera_legs == 2) & (st.camera_weight > .9)
    q = quiet(meta, t) & on
    return (angles(cam, left, fwd) - angles(meta, left, fwd))[q][:, :, 1]


def session_bias(prefixes):
    """Per-frame median of the quiet-stance shin offsets pooled over episodes."""
    rows = []
    for p in prefixes:
        cam, st = load(p); meta, _ = load(p + '.meta')
        rows.append(quiet_shin(cam, meta, st, times(cam)))
    return np.median(np.concatenate(rows), 0), sum(map(len, rows))


def shin_bias(cam, meta, st, t, hz=4., shin=None):
    """Session-constant shin pitch/roll bias from quiet stance (or the given
    `shin` [2 legs, 2]), Meta bone lengths from the Meta hip, 4 Hz zero-phase
    bone directions."""
    left, fwd = frame(meta)
    b = np.zeros((2, 2, 2))
    if shin is not None:
        b[:, 1] = shin
    else:
        d = quiet_shin(cam, meta, st, t)
        if len(d):
            b[:, 1] = d.mean(0)
    out = cam.copy(); w = np.clip(st.camera_weight, 0, 1)
    sos = butter(2, hz, fs=1 / np.median(np.diff(t)), output='sos')
    rot = lambda axis, angle: R.from_rotvec(axis * angle[:, None])
    for s in range(2):
        hip = pos(meta, HIP[s])
        L1 = np.linalg.norm(pos(meta, KNEE[s]) - hip, axis=1)
        L2 = np.linalg.norm(pos(meta, ANKLE[s]) - pos(meta, KNEE[s]), axis=1)
        dirs = []
        for k, (p, c) in enumerate(((HIP[s], KNEE[s]), (KNEE[s], ANKLE[s]))):
            v = pos(cam, c) - pos(cam, p); v /= np.linalg.norm(v, axis=1, keepdims=True)
            v = rot(fwd, -w * b[s, k, 1]).apply(rot(left, w * b[s, k, 0]).apply(v))
            v = sosfiltfilt(sos, v, axis=0); dirs.append(v / np.linalg.norm(v, axis=1, keepdims=True))
        knee = hip + L1[:, None] * dirs[0]
        set_leg(out, s, hip, knee, knee + L2[:, None] * dirs[1])
    return out, b


def depth_band(ankle, t):
    dt = np.median(np.diff(t)); fs = 1 / dt
    a = sosfiltfilt(butter(2, 3, fs=fs, output='sos'), ankle, axis=0)
    floor = percentile_filter(a[:, 1], 10, size=max(3, int(10. / dt)), mode='nearest')
    return a, a[:, 1] - floor


def low_runs(ankle, t, h=.05, close=.1, min_s=.15):
    """Ankle within h of its rolling floor (3 Hz smoothed), no speed test."""
    dt = np.median(np.diff(t)); _, height = depth_band(ankle, t)
    s = binary_closing(height < h, np.ones(max(1, int(close / dt))))
    return binary_opening(s, np.ones(max(1, int(min_s / dt))))


def static_stance(ankle, t, h=.04, v_across=.15, v_up=.10, close=.1, min_s=.2):
    """Low ankle that is slow across the camera depth axis and vertically.
    The depth axis (main 1-4 Hz horizontal axis) carries the wander."""
    dt = np.median(np.diff(t)); fs = 1 / dt
    band = sosfiltfilt(butter(2, [1, 4], btype='band', fs=fs, output='sos'), ankle[:, [0, 2]], axis=0)
    across = np.linalg.eigh(np.cov(band.T))[1][:, 0]
    a, height = depth_band(ankle, t)
    vel = np.gradient(a, axis=0) / dt
    s = (height < h) & (np.abs(vel[:, [0, 2]] @ across) < v_across) & (np.abs(vel[:, 1]) < v_up)
    s = binary_closing(s, np.ones(max(1, int(close / dt))))
    return binary_opening(s, np.ones(max(1, int(min_s / dt))))


def ls_smooth(x0, t, tau):
    """argmin_x sum |x-x0|^2 + sum (tau_k/dt_k)^2 |x_k+1 - x_k|^2: a velocity
    penalty whose time constant varies per step (0 = follow the data)."""
    dt = np.maximum(np.diff(t), 1e-3)
    lam = (np.minimum(tau[:-1], tau[1:]) / dt) ** 2
    ab = np.zeros((3, len(x0)))
    ab[1] = 1.; ab[1, :-1] += lam; ab[1, 1:] += lam
    ab[0, 1:] = -lam; ab[2, :-1] = -lam
    return solve_banded((1, 1), ab, x0)


def foot_lock(ankle0, t, on, tau_low=.5, tau_static=1.5):
    """Strong smoothing on static stance, medium on any low ankle (keeps slow
    real slides), none in swing; no edge blends. Returns ankle, low mask."""
    low = low_runs(ankle0, t) & on
    tau = np.where(static_stance(ankle0, t) & low, tau_static, np.where(low, tau_low, 0.))
    return ls_smooth(ankle0, t, tau), low


def two_bone(hip, ankle, L1, L2, pole):
    """Knee for hip->ankle with bone lengths and the bend toward `pole`."""
    d = ankle - hip; dist = np.linalg.norm(d, axis=1)
    dist_c = np.clip(dist, np.abs(L1 - L2) + 1e-3, L1 + L2 - 1e-4)
    u = d / dist[:, None]
    ankle = hip + u * dist_c[:, None]
    a = (L1 ** 2 - L2 ** 2 + dist_c ** 2) / (2 * dist_c)
    hgt = np.sqrt(np.maximum(L1 ** 2 - a ** 2, 0))
    p = pole - (pole * u).sum(1, keepdims=True) * u
    p /= np.maximum(np.linalg.norm(p, axis=1, keepdims=True), 1e-9)
    return hip + u * a[:, None] + p * hgt[:, None], ankle


def lock_feet(rows, meta, st, t):
    out = rows.copy(); masks = []
    on = st.camera_weight > .5  # Meta legs are already planted by their generator.
    fwd = np.cross(pos(meta, 2) - pos(meta, 3), UP); fwd /= np.linalg.norm(fwd, axis=1, keepdims=True)
    for s in range(2):
        hip = pos(meta, HIP[s])
        L1 = np.linalg.norm(pos(meta, KNEE[s]) - hip, axis=1)
        L2 = np.linalg.norm(pos(meta, ANKLE[s]) - pos(meta, KNEE[s]), axis=1)
        knee0, ankle0 = pos(rows, KNEE[s]), pos(rows, ANKLE[s])
        ankle_t, low = foot_lock(ankle0, t, on)
        # Bend toward the camera knee; nearly straight leg bends forward.
        pole = knee0 - (hip + (ankle0 - hip) * (L1 / (L1 + L2))[:, None])
        weak = np.linalg.norm(pole, axis=1) < .01
        pole[weak] = fwd[weak]
        knee, ankle = two_bone(hip, ankle_t, L1, L2, pole)
        set_leg(out, s, hip, knee, ankle)
        masks.append(low)
    return out, np.stack(masks, 1)


def clean(prefix, shin=None):
    cam, st = load(prefix); meta, _ = load(prefix + '.meta')
    t = times(cam)
    rows, bias = shin_bias(cam, meta, st, t, shin=shin)
    rows, low = lock_feet(rows, meta, st, t)
    return rows, dict(shin_bias_deg=np.degrees(bias[:, 1]).round(1).tolist(), low_share=low.mean(0).round(2).tolist())


if __name__ == '__main__':
    p = argparse.ArgumentParser(usage=__doc__)
    p.add_argument('prefix'); p.add_argument('out')
    g = p.add_mutually_exclusive_group()
    g.add_argument('--session', nargs='+', metavar='PREFIX', help='pool the shin bias over these episodes')
    g.add_argument('--shin', nargs=4, type=float, metavar=('LP', 'LR', 'RP', 'RR'), help='shin bias in degrees')
    a = p.parse_args()
    shin = None
    if a.session:
        shin, frames = session_bias(a.session)
        print('session shin bias deg', np.degrees(shin).round(1).tolist(), 'from', frames, 'quiet frames')
    elif a.shin:
        shin = np.radians(np.reshape(a.shin, (2, 2)))
    rows, info = clean(a.prefix, shin)
    np.savetxt(a.out, rows, fmt='%.17g')
    print(a.prefix, info)
