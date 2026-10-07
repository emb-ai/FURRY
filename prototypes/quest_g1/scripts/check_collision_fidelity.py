"""Compare fitted robot contacts with the source mesh at identical legal poses.

Run from the Quest prototype root. No recordings or headset are needed. The
compact hulls are the production collision model.
"""
import argparse
import json
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import mujoco
import numpy as np

for project_root in Path(__file__).resolve().parents:
    if (project_root / 'g1_sim').is_dir():
        sys.path.insert(0, str(project_root))
        break

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from g1_sim.scene import make_model
from g1_sim.collisions import primitive_collisions
from g1_sim.controller import DEFAULT, JOINTS


def build_models():
    _, xml = make_model('empty')
    root = ET.fromstring(xml)
    for i, geom in enumerate(root.iter('geom')):
        if not geom.get('name'):
            geom.set('name', f'source_geom_{i}')
    xml = ET.tostring(root, encoding='unicode')
    mesh = mujoco.MjModel.from_xml_string(xml)
    primitives, _ = primitive_collisions(xml)
    return dict(mesh=mesh, primitive=mujoco.MjModel.from_xml_string(primitives))


def contacts(model, data, q):
    data.qpos[:] = model.qpos0
    data.qpos[:7] = q[:7]
    for i, name in enumerate(JOINTS):
        data.qpos[model.joint(name).qposadr[0]] = q[7+i]
    mujoco.mj_forward(model, data)
    pairs = {}
    for contact in data.contact:
        a, b = model.geom_bodyid[contact.geom1], model.geom_bodyid[contact.geom2]
        if a == 0 or b == 0 or contact.dist >= -.0002:
            continue
        pair = tuple(sorted((model.body(a).name, model.body(b).name)))
        pairs[pair] = max(pairs.get(pair, 0), -float(contact.dist))
    return pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='outputs/collision-fidelity-static.json')
    args = parser.parse_args()
    models = build_models()
    datas = {key: mujoco.MjData(model) for key, model in models.items()}
    home = np.r_[0., 0., .793, 1., 0., 0., 0., DEFAULT]
    home[7+16], home[7+23] = .2, -.2
    home_contacts = {key: [dict(pair=list(pair), depth_m=depth) for pair, depth in contacts(model, datas[key], home).items()]
                     for key, model in models.items()}
    rng = np.random.default_rng(512)
    std = np.r_[np.tile([.06, .05, .05, .08, .05, .05], 2), [.1, .1, .1],
                np.tile([.15, .07, .1, .1, .2, .35, .35], 2)]
    scores = {key: defaultdict(lambda: dict(true_contact=0, false_contact=0, missed_contact=0, max_depth_m=0.))
              for key in models if key != 'mesh'}
    for _ in range(1024):
        q = home.copy()
        q[7:] += rng.normal(0, std)
        for j, name in enumerate(JOINTS):
            q[7+j] = np.clip(q[7+j], *models['mesh'].jnt_range[models['mesh'].joint(name).id])
        result = {key: contacts(model, datas[key], q) for key, model in models.items()}
        for pair in set().union(*(set(value) for value in result.values())):
            if not (any('hip_roll' in body for body in pair) or all('wrist_' in body for body in pair)):
                continue
            for key in scores:
                reference, present = pair in result['mesh'], pair in result[key]
                score = scores[key][pair]
                if present:
                    score['true_contact' if reference else 'false_contact'] += 1
                    score['max_depth_m'] = max(score['max_depth_m'], result[key][pair])
                elif reference:
                    score['missed_contact'] += 1
    output = dict(actual_home=home_contacts, poses=1024, random_seed=512, depth_threshold_m=.0002,
                  scores={key: [dict(pair=list(pair), **score) for pair, score in sorted(value.items())] for key, value in scores.items()},
                  limitations=['Static geometric comparison; no causal gait or hardware claims.',
                               'The source mesh can also intersect at HOME; compare depth as well as contact presence.',
                               'Fingers fixed at zero; independent small joint perturbations are not a human motion distribution.'])
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2)+'\n')
    print(path)


if __name__ == '__main__':
    main()
