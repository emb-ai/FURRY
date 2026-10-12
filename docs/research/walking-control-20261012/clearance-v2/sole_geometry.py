"""Signed clearance of floor-contacting foot shapes, including shape support.

Visual meshes and foot box shapes that cannot collide with the floor are
excluded by the model collision masks. No simulation state is changed.
"""
import numpy as np
import mujoco


def sole_clearance(model, data, foot_body, floor):
    normal = data.geom_xmat[floor].reshape(3,3)[:,2]
    origin = data.geom_xpos[floor]
    values=[]
    for geom in np.flatnonzero(model.geom_bodyid == foot_body):
        if not ((model.geom_contype[geom] & model.geom_conaffinity[floor]) or
                (model.geom_contype[floor] & model.geom_conaffinity[geom])):
            continue
        kind = model.geom_type[geom]
        if kind == mujoco.mjtGeom.mjGEOM_SPHERE:
            support = model.geom_size[geom,0]
        elif kind == mujoco.mjtGeom.mjGEOM_BOX:
            local_normal = data.geom_xmat[geom].reshape(3,3).T @ normal
            support = np.abs(local_normal) @ model.geom_size[geom]
        else:
            raise ValueError('Unsupported floor-contacting sole geometry: '+str(kind))
        values.append(float(normal @ (data.geom_xpos[geom]-origin)-support))
    if not values: raise ValueError('Foot has no floor-contacting geometry')
    return min(values)
