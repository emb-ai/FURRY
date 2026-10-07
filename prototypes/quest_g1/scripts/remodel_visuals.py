"""Rebuild the versioned visual assets with Blender 5.0.1, not at APK build time.

Run: blender --background --factory-startup --python scripts/remodel_visuals.py
Optional arguments after -- are mesh names to rebuild.
"""
from pathlib import Path
import hashlib
import json
import math
import subprocess
import sys
import xml.etree.ElementTree as ET

import bpy
import bmesh

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'vendor/TWIST2/assets/g1'
DEST = ROOT / 'assets/g1/visual'
PIN = 'b06178f19a22f2138cbd31f60c6d494bc263f67d'


def recipe(name, faces):
    # Reconstruct complex CAD assemblies as continuous shells before retopology.
    if name == 'torso_link':
        return .0015, 9000
    if name == 'pelvis_contour_link':
        return .0012, 5000
    if name == 'pelvis':
        return .0012, 2500
    if 'knee_link' in name:
        return .001, 3500
    if 'hand_palm' in name:
        return .0005, 3000
    if 'hand_' in name:
        return (0, faces) if 'thumb_0' in name else (.00035, 900)
    if name == 'head_link':
        return 0, 3500
    if 'ankle_roll' in name:
        return 0, 2200
    if 'wrist_pitch' in name:
        return 0, faces
    if name == 'logo_link':
        return 0, 1200
    return 0, min(faces, max(600, min(2400, int(faces * .35))))


def remodel(path, name):
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    bpy.ops.wm.stl_import(filepath=str(path))
    obj = bpy.context.object
    source_faces = len(obj.data.polygons)
    voxel, target = recipe(name, source_faces)
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bmesh.ops.remove_doubles(bm, verts=list(bm.verts), dist=1e-7)
    bmesh.ops.dissolve_degenerate(bm, edges=list(bm.edges), dist=1e-8)
    bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
    bm.to_mesh(obj.data)
    bm.free()
    if voxel:
        obj.data.remesh_voxel_size = voxel
        obj.data.remesh_voxel_adaptivity = 0
        bpy.ops.object.voxel_remesh()
        smooth = obj.modifiers.new('Remove voxel stair steps', 'SMOOTH')
        smooth.factor = .5
        smooth.iterations = 2
        bpy.ops.object.modifier_apply(modifier=smooth.name)
    obj.data.calc_loop_triangles()
    count = len(obj.data.loop_triangles)
    if count > target:
        decimate = obj.modifiers.new('Shell retopology budget', 'DECIMATE')
        decimate.ratio = target / count
        decimate.use_collapse_triangulate = True
        bpy.ops.object.modifier_apply(modifier=decimate.name)
    # Split shading at real creases; keep curved shells smooth.
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bmesh.ops.triangulate(bm, faces=list(bm.faces))
    for face in bm.faces:
        face.smooth = True
    for edge in bm.edges:
        edge.smooth = edge.is_manifold and edge.calc_face_angle() < math.radians(40)
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()
    obj.data.calc_loop_triangles()
    out = DEST / (name + '.obj')
    # Explicit corner normals survive both MuJoCo and the GLES export stream.
    lines = ['# G1 exterior visual; derived from pinned Unitree/TWIST2 CAD.']
    lines.extend('v %.9g %.9g %.9g' % tuple(v.co) for v in obj.data.vertices)
    normal_ids, normal_map = [], {}
    for normal in obj.data.corner_normals:
        value = 'vn %.7g %.7g %.7g' % tuple(normal.vector)
        if value not in normal_map:
            normal_map[value] = len(normal_map) + 1
        normal_ids.append(normal_map[value])
    lines.extend(normal_map)
    for tri in obj.data.loop_triangles:
        lines.append('f ' + ' '.join(f'{vi+1}//{normal_ids[li]}' for vi, li in zip(tri.vertices, tri.loops)))
    out.write_text('\n'.join(lines) + '\n')
    return dict(source_file=path.name, source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                source_triangles=source_faces, triangles=len(obj.data.loop_triangles),
                voxel_size_m=voxel, target_triangles=target,
                method='voxel union, shell retopology' if voxel else 'weld, feature-preserving reduction',
                sha256=hashlib.sha256(out.read_bytes()).hexdigest())


def main():
    head = subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip()
    if head != PIN:
        raise ValueError(f'Expected TWIST2 {PIN}, got {head}')
    DEST.mkdir(parents=True, exist_ok=True)
    manifest_path = DEST / 'manifest.json'
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    records = previous.get('meshes', {})
    selected = sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else []
    for asset in ET.parse(SOURCE/'g1_sim2sim_29dof_with_hands.xml').getroot().findall('asset/mesh'):
        name = asset.get('name', Path(asset.get('file')).stem)
        if selected and name not in selected:
            continue
        records[name] = remodel(SOURCE/'meshes'/asset.get('file'), name)
        print(name, records[name], flush=True)
    manifest = dict(source_twist2=PIN, blender=bpy.app.version_string, meshes=records,
                    triangles=sum(r['triangles'] for r in records.values()))
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True)+'\n')
    print('Visual triangles:', manifest['triangles'], flush=True)


if __name__ == '__main__':
    main()
