"""Extract pinned MJX body proxies and fit error-bounded convex hand hulls.

Offline authoring dependency: scipy and trimesh. Runtime only reads collision.json.
Run with --menagerie /path/to/mujoco_menagerie at the pinned revision.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial import ConvexHull, HalfspaceIntersection
import trimesh

ROOT = Path(__file__).resolve().parents[1]
PIN = 'f054586a8e90465d49ee5be15335c4a0c7f57caf'


def fit_hand(vertices, padding=.0001, tolerance=.00075):
    source = trimesh.Trimesh(vertices=vertices, process=False).convex_hull
    directions = np.r_[np.eye(3), -np.eye(3)]
    # Add supporting planes only where the current envelope exceeds the error
    # bound. Unlike a fixed direction grid this resolves long, angled phalanges.
    for _ in range(180):
        support = (vertices @ directions.T).max(axis=0) + padding
        intersections = HalfspaceIntersection(np.c_[directions, -support], vertices.mean(axis=0)).intersections
        points = np.unique(np.round(intersections, 9), axis=0)
        nearest, distances, _ = trimesh.proximity.closest_point(source, points)
        worst = int(np.argmax(distances))
        if distances[worst] <= tolerance:
            break
        normal = (points[worst]-nearest[worst]) / distances[worst]
        directions = np.r_[directions, normal[None]]
    else:
        raise ValueError(f'Hand fit did not converge: error={distances.max()}, planes={len(directions)}, closest repeated normal={np.linalg.norm(directions[:-1]-directions[-1], axis=1).min()}')
    hull = ConvexHull(points)
    faces = hull.simplices.copy()
    for face in faces:
        a, b, c = points[face]
        if np.dot(np.cross(b-a, c-a), a-points.mean(axis=0)) < 0:
            face[1], face[2] = face[2], face[1]
    if len(faces) > 384:
        raise ValueError(f'Hand fit exceeds the 384-triangle limit: {len(faces)}')
    return points, faces, float(distances.max())


def build(menagerie):
    head = subprocess.check_output(['git', '-C', str(menagerie), 'rev-parse', 'HEAD'], text=True).strip()
    if head != PIN:
        raise ValueError(f'Expected Menagerie {PIN}, got {head}')
    root = ET.parse(menagerie/'unitree_g1/g1_mjx.xml').getroot()
    defaults = {
        'collision': dict(type='capsule'),
        'foot_box': dict(type='box', pos='0.04 0 -0.029', size='0.09 0.03 0.008'),
        'foot_capsule': dict(type='capsule', size='0.01'),
    }
    body = []
    for link in root.findall('.//body'):
        for geom in link.findall('geom'):
            if geom.get('class') not in defaults or geom.get('name') in ['left_hand_collision', 'right_hand_collision']:
                continue
            attrs = defaults[geom.get('class')] | {k: v for k, v in geom.attrib.items() if k in ['name', 'type', 'size', 'pos', 'quat', 'fromto']}
            body.append(dict(body=link.get('name'), geom=attrs))
    body_names = {item['geom']['name'] for item in body}
    pairs = [dict(pair.attrib) for pair in ET.parse(menagerie/'unitree_g1/scene_mjx.xml').getroot().findall('contact/pair')
             if pair.get('geom1') in body_names and pair.get('geom2') in body_names]
    source = ROOT/'vendor/TWIST2/assets/g1'
    robot = ET.parse(source/'g1_sim2sim_29dof_with_hands.xml').getroot()
    meshes = {m.get('name', Path(m.get('file')).stem): m.get('file') for m in robot.findall('asset/mesh')}
    hands = []
    for link in robot.findall('.//body'):
        for geom in link.findall('geom'):
            name = geom.get('mesh', '')
            if '_hand_' not in name or geom.get('contype') != '0':
                continue
            path = source/'meshes'/meshes[name]
            mesh = trimesh.load_mesh(path, process=True)
            tolerance = .0015 if 'palm' in name else .00075
            vertices, faces, error = fit_hand(mesh.vertices, tolerance=tolerance)
            print(name, len(faces), 'triangles; envelope error', error*1000, 'mm', flush=True)
            hands.append(dict(body=link.get('name'), name=name+'_collision',
                              pos=geom.get('pos', '0 0 0'), quat=geom.get('quat', '1 0 0 0'),
                              vertices=vertices.tolist(), faces=faces.tolist(),
                              max_envelope_error_m=error,
                              tolerance_m=tolerance,
                              source_file=meshes[name], source_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    assert len(body) == 25 and len(hands) == 16
    out = ROOT/'assets/g1/collision.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(menagerie_revision=PIN, hand_padding_m=.0001,
                                  finger_error_m=.00075, palm_error_m=.0015, max_hand_faces=384,
                                  hand_method='adaptive supporting planes; separate conservative convex envelope for each palm/finger link',
                                  body=body, body_pairs=pairs, hands=hands), indent=2)+'\n')
    print('Body primitives:', len(body), 'hand hulls:', len(hands), 'hand triangles:', sum(len(h['faces']) for h in hands))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--menagerie', type=Path, required=True)
    build(parser.parse_args().menagerie)
