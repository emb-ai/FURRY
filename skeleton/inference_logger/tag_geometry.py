"""AprilTag pose geometry bundled for portable inference logging.

Copied from sirius_twist dual_tag_fusion.py / stereo_calibration.py.
"""
import cv2
import numpy as np


def square(size):
    h = size / 2
    return np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], np.float32)


def tag_pose(corners, size, k, d):
    obj = square(size)
    ok, rotations, translations, errors = cv2.solvePnPGeneric(
        obj, np.asarray(corners, float), np.asarray(k, float), np.asarray(d, float),
        flags=cv2.SOLVEPNP_IPPE_SQUARE)
    candidates = []
    if not ok:
        return None
    for rv, tv, error in zip(rotations, translations, errors):
        r, t = cv2.Rodrigues(rv)[0], tv.ravel()
        if np.all((obj @ r.T + t)[:, 2] > .1) and float(error.ravel()[0]) <= 2:
            candidates.append((float(error.ravel()[0]), r, t))
    candidates.sort(key=lambda x: x[0])
    if not candidates:
        return None
    # Ambiguous planar solutions can turn anatomical offsets inside out.
    if len(candidates) > 1 and candidates[1][0] < candidates[0][0] + .15:
        if np.linalg.norm(candidates[0][1] - candidates[1][1]) > .3:
            return None
    return candidates[0][1:]
