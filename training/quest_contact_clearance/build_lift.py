"""TRAIN-only reference sole clearance for the lifted-foot penalty, Quest clips only.

Isaac exposes reference body positions but not rotations, so sole clearance of
the reference (floor-colliding spheres, as in sole_geometry) is precomputed
here by MuJoCo FK. Each clip's floor is the 20% quantile of its lower foot;
clips whose floor is off by more than 3 cm stay invalid. Authored clips keep
zeros and valid=False: their swing is handled by the phase teacher.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import mujoco
from contact_clearance import sha

MAX_FLOOR = .03


def normalize(clearance):
    """clearance [N, 2] -> (floor-relative clearance, valid)."""
    floor = float(np.quantile(clearance.min(-1), .20))
    if abs(floor) > MAX_FLOOR:
        return np.zeros_like(clearance, np.float32), False
    return (clearance - floor).astype(np.float32), True


def build(dataset, assets, output):
    from author_dataset import load_motion
    from sole_geometry import sole_clearance
    from g1_sim.controller import JOINTS
    manifest = json.loads((dataset/'manifest.json').read_text())
    model = mujoco.MjModel.from_xml_path(str(assets/'scene.xml'))
    data = mujoco.MjData(model)
    qa = [model.joint(n).qposadr[0] for n in JOINTS]
    feet = [model.body(s+'_ankle_roll_link').id for s in ('left', 'right')]
    floor = model.geom('floor').id
    arrays, files = {}, {}
    for ci, row in enumerate(r for r in manifest['clips'] if r['split'] == 'train'):
        path = dataset/row['file']
        if sha(path) != row['sha256']:
            raise ValueError('Motion hash mismatch: '+row['file'])
        key = 'motion_%04d' % ci
        if row.get('source') != 'quest':
            arrays[key] = np.zeros((1, 2), np.float32)
            files[row['file']] = dict(key=key, valid=False, quest=False)
            continue
        motion = load_motion(path)
        n, fps = len(motion['root_pos']), float(motion['fps'])
        height = np.zeros((n, 2))
        for k in range(n):
            data.qpos[:3] = motion['root_pos'][k]
            data.qpos[3:7] = np.asarray(motion['root_rot'][k])[[3, 0, 1, 2]]
            data.qpos[qa] = motion['dof_pos'][k]
            mujoco.mj_kinematics(model, data)
            height[k] = [sole_clearance(model, data, foot, floor) for foot in feet]
        arrays[key], valid = normalize(height)
        files[row['file']] = dict(key=key, fps=fps, valid=valid, quest=True,
                                  input_source=row.get('input_source', 'quest-meta'),
                                  lifted_foot_seconds=float((arrays[key] > .03).sum()/fps))
        if ci % 100 == 0:
            print(json.dumps(dict(phase='building_lift', motions=ci)), flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    quest = [f for f in files.values() if f['quest']]
    report = dict(schema='train-reference-sole-clearance-v1', files=files,
                  dataset_manifest_sha256=sha(dataset/'manifest.json'), scene_sha256=sha(assets/'scene.xml'),
                  npz_sha256=sha(output), floor='per clip: 20%% quantile of the lower foot, |floor| <= %.2f m' % MAX_FLOOR,
                  quest_clips=len(quest), invalid_quest_clips=sum(not f['valid'] for f in quest),
                  lifted_foot_seconds=sum(f['lifted_foot_seconds'] for f in quest if f['valid']), validation_used=False)
    output.with_suffix('.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ('quest_clips', 'invalid_quest_clips', 'lifted_foot_seconds')}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('dataset', 'assets', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    build(a.dataset, a.assets, a.output)
