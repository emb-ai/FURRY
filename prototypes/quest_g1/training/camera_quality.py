"""Per-episode check before camera_dataset.py: usable/camera share, live vs
replay leg mismatch, leg activity of the cleaned target.
Usage (from training/): camera_quality.py EPISODE_ID ... [--target clean3]
Episodes are looked up in ../outputs/camera_skeleton/ep/*/episode-ID with
prefixes ../outputs/camera_skeleton/targets/ID (CAMERA_RECORDING.md §5)."""
import argparse
from pathlib import Path
import numpy as np
from camera_dataset import ticks

ROOT = Path(__file__).resolve().parent.parent / 'outputs/camera_skeleton'

p = argparse.ArgumentParser(usage=__doc__)
p.add_argument('ids', nargs='+'); p.add_argument('--target', default='clean3')
a = p.parse_args()
total = usable = 0.
for e in a.ids:
    try:
        ep = next(ROOT.glob(f'ep/*/episode-{e}'))
        d = ticks(ep, str(ROOT / 'targets' / e), a.target)
    except Exception as x:
        print(e, 'FAIL', x); continue
    dur, ok = d['t'][-1] - d['t'][0], d['ok']
    legs = d['q'][ok][:, 7:19]  # qpos36: 7 root terms, legs first among the joints
    knee, hip = legs[:, [3, 9]], legs[:, [0, 6]]
    # Activity proxy: either knee bent > 34 deg or either thigh > 29 deg forward.
    active = ((knee > .6) | (hip < -.5)).any(1).mean() if len(legs) else 0.
    pct = lambda x, q: np.degrees(np.quantile(x, q)) if len(x) else 0.
    print(f"{e} dur={dur:6.1f}s usable={ok.mean():.2f} camera={(d['camera'] > .5).mean():.2f} "
          f"mism50={np.median(d['mismatch']):.2f} mism95={np.quantile(d['mismatch'], .95):.2f} "
          f"knee_p95={pct(knee, .95):.0f} hip_p5={pct(hip, .05):.0f} active={active:.2f}", flush=True)
    total += dur; usable += ok.mean() * dur
print(f'total {total:.0f}s usable {usable:.0f}s')
