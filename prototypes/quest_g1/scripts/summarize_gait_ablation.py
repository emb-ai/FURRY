"""Paired offline ablations, retaining full rollout outcomes separately.

Tracking uses the same segment/time intersection for each variant and its
matched baseline. Raw participant trajectories remain local.
"""
import argparse
import json
import re
from pathlib import Path
import mujoco
import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.spatial.transform import Rotation


def matrix(a, prefix, n):
    return np.column_stack([a[prefix + str(k)] for k in range(n)])


def rms(a):
    return float(np.sqrt(np.mean(np.square(a)))) if np.size(a) else None


def fk_wrists(m, poses):
    """Wrist positions and orientations in each pose's pelvis frame."""
    d = mujoco.MjData(m)
    points, orientations = [], []
    bodies = [m.body(side + '_wrist_yaw_link').id for side in ['left', 'right']]
    for q in poses:
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        root = Rotation.from_quat(q[[4, 5, 6, 3]])
        points.append(root.inv().apply(d.xpos[bodies] - q[:3]))
        wrist = Rotation.from_matrix(d.xmat[bodies].reshape(2, 3, 3))
        orientations.append((root.inv() * wrist).as_matrix())
    return np.asarray(points), np.asarray(orientations)


def orientation_rms_deg(actual, target):
    a = Rotation.from_matrix(actual.reshape(-1, 3, 3))
    b = Rotation.from_matrix(target.reshape(-1, 3, 3))
    return rms(np.rad2deg((b.inv() * a).magnitude()))


def active_mask(a):
    mask = a['applying'] > 0
    for segment in np.unique(a['segment']):
        active = (a['segment'] == segment) & mask
        if active.any():
            mask[active & (a['wall_s'] < a['wall_s'][active][0] + 1)] = False
    return mask


def align_pair(a, b):
    """Exact input-time matching per segment; either trace may end first."""
    left, right = [], []
    for segment in np.intersect1d(a['segment'], b['segment']):
        ai = np.flatnonzero(a['segment'] == segment)
        bi = np.flatnonzero(b['segment'] == segment)
        at, bt = a['wall_s'][ai], b['wall_s'][bi]
        if np.any(np.diff(at) <= 0) or np.any(np.diff(bt) <= 0):
            raise ValueError('Nonmonotonic wall time inside a segment')
        hi = np.clip(np.searchsorted(bt, at), 0, len(bt) - 1)
        lo = np.maximum(hi - 1, 0)
        nearest = np.where(abs(bt[lo] - at) < abs(bt[hi] - at), lo, hi)
        good = abs(bt[nearest] - at) <= 1e-8
        left.extend(ai[good])
        right.extend(bi[nearest[good]])
    return np.asarray(left, dtype=int), np.asarray(right, dtype=int)


def full_outcomes(a, log):
    q = matrix(a, 'q', 36)
    result = {'max_tilt_deg': float(a['tilt'].max()),
              'min_height_m': float(a['height'].min()),
              'max_contacts': int(a['contacts'].max()),
              'falls': log.count('fell=1'), 'segments': [], 'path_m': 0.0}
    recorded = {int(s): (int(f), float(z)) for s, f, z in
                re.findall(r'segment=(\d+) fell=(\d+) minz=([\d.eE+-]+)', log)}
    result['fall_times'] = [{'wall_s': float(w), 'sim_s': float(s)} for w, s in
                           re.findall(r'fall wall=([\d.eE+-]+) sim=([\d.eE+-]+)', log)]
    for segment in np.unique(a['segment']):
        sel = a['segment'] == segment
        xy = q[sel, :2]
        path = float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())
        delta = xy[-1] - xy[0]
        detail = {'segment': int(segment), 'first_wall_s': float(a['wall_s'][sel][0]),
                  'last_wall_s': float(a['wall_s'][sel][-1]),
                  'duration_sim_s': float(a['sim_s'][sel][-1] - a['sim_s'][sel][0]),
                  'net_displacement_xy_m': delta.tolist(),
                  'net_displacement_m': float(np.linalg.norm(delta)), 'path_m': path}
        if int(segment) in recorded:
            fell, minz = recorded[int(segment)]
            detail.update(fell=bool(fell), min_height_m=minz)
            result['min_height_m'] = min(result['min_height_m'], minz)
        elif len(np.unique(a['segment'])) == 1:
            detail['fell'] = bool(result['falls'])
            z = re.search(r'minz=([\d.eE+-]+)', log)
            if z:
                detail['min_height_m'] = float(z[1])
                result['min_height_m'] = min(result['min_height_m'], float(z[1]))
        result['segments'].append(detail)
        result['path_m'] += path
    return result


def continuous_blocks(a):
    """Do not join pause gaps into the filter's time series."""
    breaks = np.flatnonzero((np.diff(a['segment']) != 0) |
                           (abs(np.diff(a['sim_s']) - .01) > 1e-6)) + 1
    return np.split(np.arange(len(a)), breaks)


def paired_metrics(a, b, m):
    """b supplies the shared target for this matched surviving interval."""
    if not len(a):
        return None
    q, ref, v = matrix(a, 'q', 36), matrix(a, 'ref', 36), matrix(a, 'v', 35)
    cmd, common = matrix(a, 'cmd_', 35), matrix(b, 'cmd_', 35)
    truth = matrix(b, 'truth', 36) if 'truth0' in b.dtype.names else None
    target = truth if truth is not None else matrix(b, 'ref', 36)
    result = {'samples': len(a), 'p95_tilt_deg': float(np.percentile(a['tilt'], 95))}
    loaded = a['foot_force'] > 20
    result['loaded_foot_slip_p95_m_s'] = (float(np.percentile(a['foot_slip'][loaded], 95))
                                        if loaded.any() else None)
    for label, sl in [('legs', slice(0, 12)), ('arms', slice(15, 29))]:
        result[label + '_tracking_rmse_rad'] = rms((q[:, 7:] - cmd[:, 6:])[:, sl])
        result[label + '_common_ref_rmse_rad'] = rms((q[:, 7:] - common[:, 6:])[:, sl])
        jitter = []
        for block in continuous_blocks(a):
            if len(block) > 30:
                filtered = sosfiltfilt(butter(3, 6, fs=100, btype='highpass', output='sos'),
                                      v[block, 6:][:, sl], axis=0)
                jitter.append(filtered)
        result[label + '_velocity_above6Hz_rms_rad_s'] = (rms(np.concatenate(jitter))
                                                        if jitter else None)
        if truth is not None:
            result[label + '_retarget_truth_rmse_rad'] = rms((ref[:, 7:] - truth[:, 7:])[:, sl])
            result[label + '_physics_truth_rmse_rad'] = rms((q[:, 7:] - truth[:, 7:])[:, sl])
    xy_error, ref_xy_error = [], []
    for segment in np.unique(a['segment']):
        sel = a['segment'] == segment
        # Initial translation alignment only; never erase heading error.
        desired = target[sel, :2] - target[sel, :2][0]
        xy_error.append((q[sel, :2] - q[sel, :2][0]) - desired)
        ref_xy_error.append((ref[sel, :2] - ref[sel, :2][0]) - desired)
    result['root_shared_xy_trajectory_rmse_m'] = rms(np.linalg.norm(np.concatenate(xy_error), axis=1))
    result['root_retarget_shared_xy_trajectory_rmse_m'] = rms(np.linalg.norm(np.concatenate(ref_xy_error), axis=1))
    idx = np.arange(0, len(a), 5)
    actual_pos, actual_rot = fk_wrists(m, q[idx])
    desired_pos, desired_rot = fk_wrists(m, target[idx])
    retarget_pos, retarget_rot = fk_wrists(m, ref[idx])
    result['wrist_common_target_rms_m'] = rms(np.linalg.norm(actual_pos - desired_pos, axis=2))
    result['wrist_retarget_target_rms_m'] = rms(np.linalg.norm(retarget_pos - desired_pos, axis=2))
    result['wrist_common_target_orientation_rms_deg'] = orientation_rms_deg(actual_rot, desired_rot)
    result['wrist_retarget_target_orientation_rms_deg'] = orientation_rms_deg(retarget_rot, desired_rot)
    return result


def summarize(path, base, m):
    a = np.atleast_1d(np.genfromtxt(path, delimiter=',', names=True))
    b = np.atleast_1d(np.genfromtxt(base, delimiter=',', names=True))
    result = full_outcomes(a, path.with_suffix('.log').read_text())
    ai, bi = align_pair(a, b)
    good = active_mask(a)[ai] & active_mask(b)[bi]
    ai, bi = ai[good], bi[good]
    pair_a, pair_b = a[ai], b[bi]
    comparison = paired_metrics(pair_a, pair_b, m)
    result.update(comparison or {'samples': 0, 'failure': 'No matched active samples after 1 s warmup'})
    result['matched_baseline'] = paired_metrics(pair_b, pair_b, m)
    result['paired_windows'] = [
        {'segment': int(segment), 'samples': int((pair_a['segment'] == segment).sum()),
         'first_wall_s': float(pair_a['wall_s'][pair_a['segment'] == segment][0]),
         'last_wall_s': float(pair_a['wall_s'][pair_a['segment'] == segment][-1])}
        for segment in np.unique(pair_a['segment'])]
    result['limitations'] = [
        'Offline replay; no hardware validation or human foot ground truth.',
        'Full outcomes and paired metrics use different horizons; inspect falls and paired_windows together.',
        'Synthetic truth is a robot FK round-trip, not independent human tracking validation.'
        if 'truth0' in a.dtype.names else
        'Recorded shared target is baseline GMR; common-reference error does not establish human pose accuracy.',
        'Above-6 Hz velocity measures roughness and can include intended motion harmonics.',
        'Root-relative wrists omit global base drift, reported separately as XY trajectory error.'
    ]
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('directory', type=Path)
    args = p.parse_args()
    m = mujoco.MjModel.from_xml_path('android/assets/gmr_model.xml')
    result = {}
    for folder in sorted(args.directory.iterdir()):
        if not folder.is_dir():
            continue
        entry = {}
        for label in ['replay', 'walk', 'arms']:
            path = folder / (label + '.csv')
            base = args.directory / 'baseline' / (label + '.csv')
            if path.exists() and path.with_suffix('.log').exists() and base.exists():
                entry[label] = summarize(path, base, m)
        if entry:
            result[folder.name] = entry
    (args.directory / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    for variant, entry in result.items():
        for task, r in entry.items():
            if not r['samples']:
                print(variant, task, r['failure'], 'falls=', r['falls'])
                continue
            matched = r['matched_baseline']
            print(f"{variant:16} {task:6} falls={r['falls']} paired_n={r['samples']} "
                  f"tilt95={r['p95_tilt_deg']:.2f}/{matched['p95_tilt_deg']:.2f} "
                  f"legerr={r['legs_common_ref_rmse_rad']:.3f}/{matched['legs_common_ref_rmse_rad']:.3f} "
                  f"wrist={r['wrist_common_target_rms_m']:.3f}/{matched['wrist_common_target_rms_m']:.3f} "
                  f"rootXY={r['root_shared_xy_trajectory_rmse_m']:.3f}/{matched['root_shared_xy_trajectory_rmse_m']:.3f}")
