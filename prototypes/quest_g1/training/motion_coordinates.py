"""Coordinate convention at the Quest NPZ / upstream TWIST2 PKL boundary."""
import numpy as np
from scipy.spatial.transform import Rotation


def world_offsets_to_root(offsets, root_xyzw):
    """Convert world-axis (body - pelvis) offsets into pelvis-oriented axes.

    Quest NPZ retains the historical world-axis representation; upstream PKL
    rotates these offsets by root_rot, so it requires the inverse rotation here.
    This is a rotation only: root positions, robot scale and joint angles stay fixed.
    """
    rotation = Rotation.from_quat(root_xyzw).as_matrix()
    return np.einsum('nji,nkj->nki', rotation, offsets)
