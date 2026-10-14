"""Raw 2D keypoints for the headset's ray-based leg solve.

The headset reprojects knees and ankles onto these pixels instead of using
D435 depth. Points are sent before One-Euro smoothing (the headset smooths
its own leg targets) and without lens distortion, so a pinhole model with
intr = [fx, fy, cx, cy] maps them to camera rays exactly.
"""
import cv2
import numpy as np

MIN_CONF = 0.05


def wire_keypoints(xy, conf, intrinsics, dist=None, min_conf=MIN_CONF):
    """{coco index: (u, v, conf)} for detected keypoints, undistorted pixels.

    xy: (17, 2) detector pixels; conf: (17,) scores; intrinsics: [fx, fy, cx, cy];
    dist: OpenCV-order distortion coefficients or None. Scores below min_conf
    are dropped, not zeroed: an absent key means "not detected".
    """
    xy = np.asarray(xy, np.float64).reshape(-1, 2)
    conf = np.asarray(conf, np.float64).reshape(-1)
    keep = [i for i in range(min(len(xy), len(conf)))
            if np.isfinite(conf[i]) and conf[i] >= min_conf and np.isfinite(xy[i]).all()]
    if not keep:
        return {}
    points = xy[keep]
    if dist is not None and np.any(np.abs(np.asarray(dist, np.float64)) > 1e-9):
        fx, fy, cx, cy = intrinsics
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float64)
        points = cv2.undistortPoints(points.reshape(-1, 1, 2), K, np.asarray(dist, np.float64), P=K).reshape(-1, 2)
    return {i: (float(u), float(v), float(conf[i])) for i, (u, v) in zip(keep, points)}
