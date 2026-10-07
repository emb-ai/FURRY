"""Render the same articulated robot before/after and inspect hand hull fits.

Writes diagnostic images and JSON under ignored outputs/geometry. Requires the
optional requirements-geometry.txt authoring dependencies and an OpenGL context.
"""
import argparse
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from g1_sim.controller import Controller, UPSTREAM
from g1_sim.geometry import ASSETS
from g1_sim.scene import make_model


def legacy_model(source, xml, directory):
    import fast_simplification
    root = ET.fromstring(xml)
    for asset in root.findall('asset/mesh'):
        name = asset.get('name', Path(asset.get('file')).stem)
        i = source.mesh(name).id
        va, vn, fa, fn = source.mesh_vertadr[i], source.mesh_vertnum[i], source.mesh_faceadr[i], source.mesh_facenum[i]
        v = source.mesh_vert[va:va+vn].astype(float)
        f = source.mesh_face[fa:fa+fn]
        target = min(int(fn), max(200, min(1800, int(fn*.12))))
        if target < fn:
            v, f = fast_simplification.simplify(v, f, target_count=target)
        rotation = np.zeros(9)
        mujoco.mju_quat2Mat(rotation, source.mesh_quat[i])
        v = v @ rotation.reshape(3, 3).T + source.mesh_pos[i]
        points = v[f]
        normals = np.cross(points[:, 1]-points[:, 0], points[:, 2]-points[:, 0])
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
        lines = ['v %.9g %.9g %.9g' % tuple(p) for p in v]
        lines.extend('vn %.9g %.9g %.9g' % tuple(n) for n in normals)
        lines.extend('f '+' '.join(f'{p+1}//{j+1}' for p in face) for j, face in enumerate(f))
        path = directory / f'{name}.obj'
        path.write_text('\n'.join(lines)+'\n')
        asset.set('file', str(path.resolve()))
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))


def pose(model, grip=0):
    data = mujoco.MjData(model)
    Controller(model).initialize(data)
    for i in range(model.njnt):
        name = model.joint(i).name
        if '_hand_' not in name:
            continue
        sign = 1 if name.startswith('left') else -1
        angle = 0 if 'thumb_0' in name else sign*(1 if '_1_' in name else 1.74) if 'thumb' in name else -sign*(1.57 if '_0_' in name else 1.74)
        data.qpos[model.jnt_qposadr[i]] = np.clip(grip*angle, *model.jnt_range[i])
    mujoco.mj_forward(model, data)
    return data


def render(model, data, camera, collision=False, segmentation=False, hand_only=False):
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 0
    opt.geomgroup[3 if collision else 1] = 1
    opt.sitegroup[:] = 0
    with mujoco.Renderer(model, height=720, width=512) as renderer:
        renderer.update_scene(data, camera, scene_option=opt)
        if hand_only:
            for g in renderer.scene.geoms[:renderer.scene.ngeom]:
                mesh_id = model.geom_dataid[g.objid] if g.objtype == mujoco.mjtObj.mjOBJ_GEOM else -1
                name = model.mesh(int(mesh_id)).name if mesh_id >= 0 else ''
                if not name.startswith('left_hand_'):
                    g.rgba[3] = 0
        if collision:
            for g in renderer.scene.geoms[:renderer.scene.ngeom]:
                if g.rgba[3] != 0:
                    g.rgba[:] = [.2, .65, .4, 1]
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        if segmentation:
            renderer.enable_segmentation_rendering()
            return renderer.render()[:, :, 0] >= 0
        return Image.fromarray(renderer.render())


def montage(images, labels, path):
    out = Image.new('RGB', (512*len(images), 790), '#edf1f4')
    try:
        font = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', 22)
    except OSError:
        font = ImageFont.load_default()
    draw = ImageDraw.Draw(out)
    for i, (im, label) in enumerate(zip(images, labels)):
        out.paste(im, (512*i, 60))
        draw.text((512*i+20, 18), label, fill='#17202a', font=font)
    out.save(path)


def inspect(directory):
    directory.mkdir(parents=True, exist_ok=True)
    source, xml = make_model('stand', geometry='source')
    hybrid, _ = make_model('stand')
    # Hide source colliders even though its original XML uses group 0.
    legacy_dir = directory/'legacy'
    legacy_dir.mkdir(exist_ok=True)
    old = legacy_model(source, xml, legacy_dir)
    models = [source, old, hybrid, hybrid]
    labels = ['Source: same articulated hands', 'Previous Quest: 66,509 triangles', 'Remodelled: 88,047 triangles', 'Hybrid collisions: same hands']
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0, 0, .68]
    camera.distance, camera.azimuth, camera.elevation = 2.1, 35, -12
    data = [pose(m) for m in models]
    montage([render(m, d, camera, collision=i == 3) for i, (m, d) in enumerate(zip(models, data))], labels, directory/'comparison.png')
    for grip, name in [(0, 'open'), (.85, 'closed')]:
        data = [pose(m, grip) for m in models]
        positions = []
        for i in range(source.ngeom):
            mid = source.geom_dataid[i]
            if source.geom_type[i] != mujoco.mjtGeom.mjGEOM_MESH or not source.mesh(int(mid)).name.startswith('left_hand_'):
                continue
            va, vn = source.mesh_vertadr[mid], source.mesh_vertnum[mid]
            positions.append(source.mesh_vert[va:va+vn] @ data[0].geom_xmat[i].reshape(3, 3).T + data[0].geom_xpos[i])
        points = np.concatenate(positions)
        camera.lookat[:] = (points.min(axis=0)+points.max(axis=0))/2
        camera.distance, camera.azimuth, camera.elevation = .38, 60, -25
        montage([render(m, d, camera, collision=i == 3, hand_only=True) for i, (m, d) in enumerate(zip(models, data))], labels, directory/f'hand-{name}.png')
    camera.lookat[:] = [0, 0, .68]
    camera.distance, camera.elevation = 2.1, -12
    silhouette = {}
    for azimuth in [0, 90, 180, 270]:
        camera.azimuth = azimuth
        a = render(source, pose(source), camera, segmentation=True)
        b = render(hybrid, pose(hybrid), camera, segmentation=True)
        silhouette[azimuth] = float(np.count_nonzero(a & b)/np.count_nonzero(a | b))
    hands = []
    for hand in json.loads((ASSETS/'collision.json').read_text())['hands']:
        original = trimesh.load_mesh(UPSTREAM/'assets/g1/meshes'/hand['source_file'], process=True).convex_hull
        vertices = np.array(hand['vertices'])
        distances = trimesh.proximity.closest_point(original, vertices)[1]*1000
        hands.append(dict(name=hand['name'], faces=len(hand['faces']), max_vertex_offset_mm=float(distances.max())))
    report = dict(silhouette_iou=silhouette, hand_hull_fit=hands,
                  note='Silhouette: four full-body perspective views, same pose. Hand fit: proxy vertices to source convex hull, not concave CAD surface.')
    (directory/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/geometry')
    inspect(parser.parse_args().output)
