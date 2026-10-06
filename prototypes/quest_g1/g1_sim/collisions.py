"""Replace only robot collision meshes with fitted primitives, preserving dynamics."""
import xml.etree.ElementTree as ET
from collections import Counter
import mujoco
import numpy as np


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
    assert not np.any(model.geom_type[active] == mujoco.mjtGeom.mjGEOM_MESH)
    return result, dict(replaced_meshes=len(changes), shapes=dict(Counter(c['shape'] for c in changes)),
                        robot_mass_preserved=True, ground_contact_pads_preserved=True, changes=changes)
