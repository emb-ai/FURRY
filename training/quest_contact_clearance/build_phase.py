"""Conservative TRAIN-only kinematic teacher from authored motions, never Quest legs.

FK uses actual collision spheres. Complete swing intervals require low-speed
stance before/after, continuous contralateral support and plausible duration.
This is a confidence-filtered reference teacher, not measured human contact.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import mujoco
from contact_clearance import sha


def infer(clearance, foot_pos, fps):
    n = len(clearance)
    speed = np.linalg.norm(np.gradient(foot_pos, 1/fps, axis=0), axis=-1)
    # Authored retargets have a constant sole-to-floor offset with the new spheres.
    # Infer a single clip floor ONLY for phase classification, never from the actor.
    slow = speed < .15
    if slow.sum() < max(6, round(.2*fps)):
        return np.zeros((n,2,3),np.float32), []
    floor = float(np.quantile(clearance[slow], .20))
    if abs(floor) > .065:
        return np.zeros((n,2,3),np.float32), []
    clearance = clearance - floor
    stance = (np.abs(clearance) <= .012) & (speed <= .20)
    result = np.zeros((n, 2, 3), np.float32)  # target, valid, expected stance
    result[..., 1] = stance
    result[..., 2] = stance
    intervals = []
    for side in range(2):
        elevated = clearance[:, side] > .015
        edges = np.diff(np.r_[False, elevated, False].astype(int))
        for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            pad = max(1, round(.20*fps))
            before, after = max(0, start-pad), min(n, end+pad)
            duration = (end-start)/fps
            if start < pad or end+pad >= n or not .12 <= duration <= 1.0:
                continue
            if not stance[before:start, side].any() or not stance[end:after, side].any():
                continue
            if stance[start:end, 1-side].mean() < .85:
                continue
            peak = clearance[start:end, side].max()
            if not .02 <= peak <= .16 or speed[start:end, side].max() > 3.:
                continue
            lo = before + np.flatnonzero(stance[before:start, side])[-1]
            hi = end + np.flatnonzero(stance[end:after, side])[0]
            phase = np.linspace(0, 1, hi-lo+1)
            # Fixed floor clearance profile, bounded at 3.5 cm, zero at touchdown.
            result[lo:hi+1, side, 0] = .035*np.sin(np.pi*phase)**2
            result[lo:hi+1, side, 1] = 1
            result[lo:hi+1, side, 2] = 0
            result[[lo, hi], side, 2] = 1
            intervals.append([side, int(lo), int(hi)])
    return result, intervals


def build(dataset, assets, output):
    from author_dataset import load_motion
    manifest = json.loads((dataset/'manifest.json').read_text())
    model = mujoco.MjModel.from_xml_path(str(assets/'scene.xml'))
    data = mujoco.MjData(model)
    from g1_sim.controller import JOINTS
    qa = [model.joint(n).qposadr[0] for n in JOINTS]
    feet = [model.body(s+'_ankle_roll_link').id for s in ('left', 'right')]
    from sole_geometry import sole_clearance
    floor = model.geom('floor').id
    arrays, files, coverage, geometry = {}, {}, {}, {}
    for ci, row in enumerate(r for r in manifest['clips'] if r['split'] == 'train'):
        path = dataset/row['file']
        if sha(path) != row['sha256']:
            raise ValueError('Motion hash mismatch: '+row['file'])
        motion = load_motion(path)
        n, fps = len(motion['root_pos']), float(motion['fps'])
        result = np.zeros((n, 2, 3), np.float32)
        intervals = []
        trusted = row.get('source') == 'twist2_author' and row.get('author_split') == 'train'
        categories = {'all': np.ones(n, bool)}
        if trusted:
            height = np.zeros((n, 2)); pos = np.zeros((n, 2, 3))
            for k in range(n):
                data.qpos[:3] = motion['root_pos'][k]
                data.qpos[3:7] = np.asarray(motion['root_rot'][k])[[3, 0, 1, 2]]
                data.qpos[qa] = motion['dof_pos'][k]
                mujoco.mj_forward(model, data)
                pos[k] = data.xpos[feet]
                height[k] = [sole_clearance(model, data, foot, floor) for foot in feet]
            result, intervals = infer(height, pos, fps)
            geometry['motion_%04d' % ci] = np.c_[height, pos.reshape(n,6)].astype(np.float32)
            from scipy.spatial.transform import Rotation
            rot = Rotation.from_quat(motion['root_rot'])
            vel = rot.inv().apply(np.gradient(motion['root_pos'], 1/fps, axis=0))
            yaw = np.gradient(np.unwrap(rot.as_euler('xyz')[:, 2]), 1/fps)
            moving = np.linalg.norm(vel[:, :2], axis=-1) > .05
            categories.update(walking=moving, backward=vel[:, 0] < -.05,
                              yaw_only=(~moving) & (np.abs(yaw) > .15),
                              idle=(~moving) & (np.abs(yaw) <= .15),
                              transition=np.abs(np.gradient(moving.astype(float))) > 0)
        family = row.get('family', row.get('source', 'unknown'))
        for category, mask in categories.items():
            key = family+'/'+category
            c = coverage.setdefault(key, dict(foot_frames=0, valid=0, swing=0))
            c['foot_frames'] += int(mask.sum())*2
            c['valid'] += int(result[mask, :, 1].sum())
            c['swing'] += int(((result[mask, :, 0] > .005) & (result[mask, :, 1] > 0)).sum())
        key = 'motion_%04d' % ci
        arrays[key] = result
        files[row['file']] = dict(key=key, fps=fps, source_sha256=row['sha256'],
                                 trusted_author=trusted, swing_intervals=intervals)
        if ci % 100 == 0:
            print(json.dumps(dict(phase='building_teacher', motions=ci)), flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    np.savez_compressed(output.with_name(output.stem+'-kinematics.npz'), **geometry)
    for c in coverage.values():
        c['valid_fraction'] = c['valid']/max(1, c['foot_frames'])
        c['swing_fraction'] = c['swing']/max(1, c['foot_frames'])
    report = dict(schema='train-kinematic-phase-v1', files=files, coverage=coverage,
                  dataset_manifest_sha256=sha(dataset/'manifest.json'), scene_sha256=sha(assets/'scene.xml'),
                  npz_sha256=sha(output), teacher='authored non-PICO TRAIN kinematics with conservative support confidence',
                  quest_phase_valid=False, validation_used=False, approved=False,
                  limitation='Automatic kinematic consistency is not physical contact ground truth. Requires audit before C.')
    output.with_suffix('.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(dict(phase='teacher_built', coverage=coverage)), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('dataset', 'assets', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    build(a.dataset, a.assets, a.output)
