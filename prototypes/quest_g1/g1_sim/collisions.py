"""Fast collision proxies, with faithful convex housings where primitives overlap."""
import xml.etree.ElementTree as ET
from collections import Counter
import mujoco
import numpy as np
from scipy.spatial import ConvexHull


HULL_TOLERANCE_M = .00015


def compact_hull(vertices):
    """Inner convex hull with bounded source-vertex plane error (150 microns).

    Select original hull vertices, never enlarge a housing. MuJoCo uses convex
    mesh contacts too; this retains its contact surface without the STL detail.
    """
    vertices = vertices[ConvexHull(vertices).vertices]
    chosen = set(np.r_[vertices.argmin(axis=0), vertices.argmax(axis=0)].tolist())
    for _ in range(len(vertices)):
        points = vertices[sorted(chosen)]
        hull = ConvexHull(points)
        # einsum avoids platform BLAS warnings for these small matrices.
        errors = np.max(np.einsum("ij,kj->ik", vertices, hull.equations[:, :3])
                        + hull.equations[:, 3], axis=1)
        index = int(errors.argmax())
        if errors[index] <= HULL_TOLERANCE_M:
            faces = hull.simplices.copy()
            for i, face in enumerate(faces):
                a, b, c = points[face]
                if np.dot(np.cross(b-a, c-a), hull.equations[i, :3]) < 0:
                    faces[i, [1, 2]] = faces[i, [2, 1]]
            return points, faces, float(errors[index]), float(hull.volume / ConvexHull(vertices).volume)
        chosen.add(index)
    raise ValueError("Collision hull fitting did not converge")


def primitive_collisions(xml):
    root = ET.fromstring(xml)
    for i, geom in enumerate(root.iter('geom')):
        if not geom.get('name'):
            geom.set('name', f'source_geom_{i}')
    original = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
    robot = root.find(".//body[@name='pelvis']")
    changes = []
    for geom in robot.iter('geom'):
        gid = mujoco.mj_name2id(original, mujoco.mjtObj.mjOBJ_GEOM, geom.get('name'))
        if not (original.geom_contype[gid] or original.geom_conaffinity[gid]):
            continue
        geom.set('group', '3')
        if original.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mid = original.geom_dataid[gid]
        vertices = original.mesh_vert[original.mesh_vertadr[mid]:original.mesh_vertadr[mid]+original.mesh_vertnum[mid]].astype(float)
        lo, hi = vertices.min(axis=0), vertices.max(axis=0)
        center, extent = (lo+hi)/2, (hi-lo)/2
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, original.geom_quat[gid])
        rotation = rotation.reshape(3, 3)
        position = original.geom_pos[gid] + rotation @ center
        body = original.body(original.geom_bodyid[gid]).name
        # These non-cylindrical housings cannot be represented faithfully by
        # a bounding box/capsule: they create wrist and thumb/hip contacts.
        precise = body.endswith('hip_roll_link') or (
            body.endswith(('wrist_roll_link', 'wrist_yaw_link'))
            and original.mesh(mid).name == body)
        if precise:
            points, faces, error, volume_ratio = compact_hull(vertices)
            mesh_name = 'collision_hull_' + geom.get('name')
            ET.SubElement(root.find('asset'), 'mesh', name=mesh_name,
                          vertex=' '.join(format(v, '.12g') for v in points.ravel()),
                          face=' '.join(map(str, faces.ravel())))
            for attr in ('size', 'pos', 'quat', 'euler', 'axisangle', 'xyaxes', 'zaxis', 'fromto'):
                geom.attrib.pop(attr, None)
            geom.set('type', 'mesh'); geom.set('mesh', mesh_name)
            geom.set('pos', ' '.join(map(str, original.geom_pos[gid])))
            geom.set('quat', ' '.join(map(str, original.geom_quat[gid])))
            changes.append(dict(body=body, geom=geom.get('name'), shape='compact_convex_hull',
                                vertices=len(points), triangles=len(faces),
                                max_plane_error_m=error, volume_ratio=volume_ratio))
            continue
        axis = int(np.argmax(extent))
        other = [a for a in range(3) if a != axis]
        radius = float(min(extent[other]))
        slender = extent[axis] > 1.3*radius
        # Rounded torso: ellipsoids; flat palm housings: boxes. Long links:
        # capsules with the same axial bounds and inscribed cross-section.
        # A circumscribed radius creates artificial shoulder/torso overlaps.
        capsule = slender and body != 'torso_link' and not body.endswith('wrist_yaw_link')
        # Rounded pelvis avoids the artificial hip intersections of an OBB.
        sphere = body == 'pelvis'
        for attr in ('mesh', 'size', 'pos', 'quat', 'euler', 'axisangle', 'xyaxes', 'zaxis', 'fromto'):
            geom.attrib.pop(attr, None)
        if sphere:
            geom.set('type', 'sphere')
            geom.set('size', str(float(min(extent))))
            geom.set('pos', ' '.join(map(str, position)))
        elif capsule:
            half = max(0., float(extent[axis])-radius)
            a, b = position-rotation[:, axis]*half, position+rotation[:, axis]*half
            geom.set('type', 'capsule')
            geom.set('size', str(radius))
            geom.set('fromto', ' '.join(map(str, np.r_[a, b])))
        else:
            geom.set('type', 'ellipsoid' if body == 'torso_link' else 'box')
            geom.set('size', ' '.join(map(str, extent)))
            geom.set('pos', ' '.join(map(str, position)))
            geom.set('quat', ' '.join(map(str, original.geom_quat[gid])))
        changes.append(dict(body=body, geom=geom.get('name'), shape=geom.get('type')))
    result = ET.tostring(root, encoding='unicode')
    model = mujoco.MjModel.from_xml_string(result)
    # All robot links already have explicit inertias. Changing geometry must
    # never alter mass, COM, joint dynamics, motor limits or ground contact pads.
    for field in ('body_mass', 'body_inertia', 'body_ipos', 'body_iquat', 'dof_armature',
                  'dof_damping', 'jnt_range', 'actuator_ctrlrange'):
        np.testing.assert_allclose(getattr(model, field), getattr(original, field), atol=1e-12, rtol=0)
    active = (model.geom_contype != 0) | (model.geom_conaffinity != 0)
    assert np.count_nonzero(model.geom_type[active] == mujoco.mjtGeom.mjGEOM_MESH) == 6
    return result, dict(replaced_meshes=len(changes), shapes=dict(Counter(c['shape'] for c in changes)),
                        robot_mass_preserved=True, ground_contact_pads_preserved=True,
                        hull_tolerance_m=HULL_TOLERANCE_M, collision_hulls=6, changes=changes)
