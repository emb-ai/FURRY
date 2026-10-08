"""Камера → STAGE по одному AprilTag 36h11 известного размера.

Маркер лежит в кадре камеры. solvePnP по цветной картинке даёт его позу
в осях камеры, без глубины. Та же поза в STAGE — это поза контроллера в
момент касания: точка uplink стоит в центре маркера, контроллер смотрит
в наклейку, его «верх» совпадает с верхом маркера.

Оси маркера (OpenCV): начало в центре, X вправо, Y вверх, Z из плоскости
к камере. Контроллер OpenXR смотрит вдоль своего −Z, поэтому при таком
касании оси маркера совпадают с осями контроллера: R_stage = R_controller.

Связка масштаб 1, на весь запуск:
    p_stage = R @ p_camera + t
После фиксации суставы уходят как frame=pelvis-relative: смещения от таза
уже в осях STAGE, таз — абсолютный. Повторный поворот на шлеме не нужен.
"""
import json
from pathlib import Path

import cv2
import numpy as np

DICTIONARY_ID = cv2.aruco.DICT_APRILTAG_36h11
MIN_SAMPLES = 5
WINDOW_S = 0.5
MAX_REPROJ_PX = 2.0
MIN_EDGE_PX = 40.0
MAX_TRANSLATION_M = 0.015
MAX_ROTATION_DEG = 2.0
HANDS = {"right": "ctrl_r", "left": "ctrl_l"}


def object_points(size_m):
    """Углы чёрного квадрата: левый-верх, правый-верх, правый-низ, левый-низ."""
    half = float(size_m) / 2.0
    return np.array([[-half, half, 0], [half, half, 0],
                     [half, -half, 0], [-half, -half, 0]], np.float64)


def dictionary():
    if hasattr(cv2.aruco, "getPredefinedDictionary"):
        return cv2.aruco.getPredefinedDictionary(DICTIONARY_ID)
    return cv2.aruco.Dictionary_get(DICTIONARY_ID)


def generate_marker(marker_id, pixels, border_bits=1):
    """Картинка для печати. size_m — сторона чёрного квадрата, не всей картинки."""
    image = np.zeros((int(pixels), int(pixels)), np.uint8)
    book = dictionary()
    if hasattr(cv2.aruco, "generateImageMarker"):
        cv2.aruco.generateImageMarker(book, int(marker_id), int(pixels), image, int(border_bits))
    else:
        cv2.aruco.drawMarker(book, int(marker_id), int(pixels), image, int(border_bits))
    return image


def _detector():
    book = dictionary()
    if hasattr(cv2.aruco, "DetectorParameters"):
        params = cv2.aruco.DetectorParameters()
    else:
        params = cv2.aruco.DetectorParameters_create()
    if hasattr(params, "cornerRefinementMethod") and hasattr(cv2.aruco, "CORNER_REFINE_SUBPIX"):
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    if hasattr(cv2.aruco, "ArucoDetector"):
        return cv2.aruco.ArucoDetector(book, params)
    return book, params


def detect_corners(image, marker_id, detector):
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if isinstance(detector, tuple):
        corners, ids, _ = cv2.aruco.detectMarkers(gray, detector[0], parameters=detector[1])
    else:
        corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return None
    for corner, found in zip(corners, ids.reshape(-1)):
        if int(found) == int(marker_id):
            return np.asarray(corner, np.float64).reshape(4, 2)
    return None


def pose_from_corners(corners, camera_matrix, dist, size_m):
    """R, t, средняя репроекция (px). Маркер перед камерой, меньшая ошибка."""
    obj = object_points(size_m)
    corners = np.asarray(corners, np.float64).reshape(4, 2)
    dist = np.zeros(5) if dist is None else np.asarray(dist, np.float64).reshape(-1)
    flag = getattr(cv2, "SOLVEPNP_IPPE_SQUARE", cv2.SOLVEPNP_ITERATIVE)
    try:
        count, rvecs, tvecs, _ = cv2.solvePnPGeneric(
            obj, corners, camera_matrix, dist, flags=flag)
    except cv2.error:
        count, rvecs, tvecs, _ = cv2.solvePnPGeneric(
            obj, corners, camera_matrix, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    best = None
    for rvec, tvec in zip(list(rvecs)[:count], list(tvecs)[:count]):
        translation = np.asarray(tvec, np.float64).reshape(3)
        if translation[2] <= 0:
            continue
        projected, _ = cv2.projectPoints(obj, rvec, tvec, camera_matrix, dist)
        error = float(np.mean(np.linalg.norm(projected.reshape(4, 2) - corners, axis=1)))
        if best is None or error < best[0]:
            rotation, _ = cv2.Rodrigues(rvec)
            best = (error, rotation, translation)
    if best is None:
        return None
    return best[1], best[2], best[0]


def quat_wxyz_to_R(quaternion):
    quaternion = np.asarray(quaternion, np.float64)
    norm = np.linalg.norm(quaternion)
    if not np.isfinite(norm) or norm < 1e-8:
        raise ValueError("quaternion")
    w, x, y, z = quaternion / norm
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def controller_marker_pose(position, quaternion):
    """Поза маркера в STAGE. См. договорённость об осях в шапке модуля."""
    return quat_wxyz_to_R(quaternion), np.asarray(position, np.float64).reshape(3)


def camera_to_stage(R_cam, t_cam, R_stage, t_stage):
    """p_stage = R @ p_cam + t. Обе позы — один и тот же маркер."""
    rotation = np.asarray(R_stage, np.float64) @ np.asarray(R_cam, np.float64).T
    translation = (np.asarray(t_stage, np.float64).reshape(3)
                   - rotation @ np.asarray(t_cam, np.float64).reshape(3))
    return rotation, translation


def mean_rotation(rotations):
    mean = np.mean(np.stack(rotations), axis=0)
    u, _, vt = np.linalg.svd(mean)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    return rotation


def rotation_angle_deg(a, b):
    cosine = (np.trace(np.asarray(a).T @ np.asarray(b)) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def to_stage_relative(points, rotation, translation):
    """Точки камеры → (смещения от таза в STAGE, абсолютный таз).

    None, если нет обоих бёдер: середину таза тогда не из чего считать,
    а абсолютные координаты в pelvis-relative кадре были бы ошибкой.
    """
    staged = {i: rotation @ np.asarray(p, np.float64) + translation for i, p in points.items()}
    if 11 not in staged or 12 not in staged:
        return None
    pelvis = 0.5 * (staged[11] + staged[12])
    return {i: p - pelvis for i, p in staged.items()}, pelvis


class MarkerSession:
    def __init__(self, marker_m, marker_id=0, hand="right"):
        if marker_m <= 0:
            raise ValueError("marker size")
        if hand not in HANDS:
            raise ValueError("hand")
        self.marker_m = float(marker_m)
        self.marker_id = int(marker_id)
        self.hand = hand
        self.detector = _detector()
        self.samples = []
        self.corners = None
        self.locked = False
        self.R = None
        self.t = None
        self.reproj_px = None

    def observe(self, image, camera_matrix, dist, now=None):
        if now is None:
            import time
            now = time.monotonic()
        corners = detect_corners(image, self.marker_id, self.detector)
        self.corners = corners
        if corners is None:
            return None
        pose = pose_from_corners(corners, camera_matrix, dist, self.marker_m)
        if pose is None:
            return None
        rotation, translation, reproj = pose
        edges = np.linalg.norm(np.roll(corners, -1, axis=0) - corners, axis=1)
        self.samples.append((now, rotation, translation, reproj, float(edges.min())))
        self.samples = [s for s in self.samples if now - s[0] <= WINDOW_S]
        return pose

    def _stable(self):
        if len(self.samples) < MIN_SAMPLES:
            return None, f"мало кадров с маркером ({len(self.samples)}/{MIN_SAMPLES})"
        _, rotations, translations, errors, edges = zip(*self.samples)
        if min(edges) < MIN_EDGE_PX:
            return None, f"маркер мелкий ({min(edges):.0f} px, нужно ≥ {MIN_EDGE_PX:.0f})"
        reproj = float(np.median(errors))
        if reproj > MAX_REPROJ_PX:
            return None, f"репроекция маркера {reproj:.2f} px"
        rotation = mean_rotation(rotations)
        translation = np.median(np.stack(translations), axis=0)
        shift = max(np.linalg.norm(t - translation) for t in translations)
        twist = max(rotation_angle_deg(rotation, r) for r in rotations)
        if shift > MAX_TRANSLATION_M or twist > MAX_ROTATION_DEG:
            return None, f"маркер дрожит: {shift * 1000:.1f} мм, {twist:.2f}°"
        return (rotation, translation, reproj), None

    def try_lock(self, uplink):
        stable, reason = self._stable()
        if stable is None:
            return False, reason
        if not uplink:
            return False, "нет свежей позы контроллера"
        ctrl = uplink.get(HANDS[self.hand]) or {}
        if not ctrl.get("tracked"):
            return False, "контроллер не в трекинге"
        try:
            R_stage, t_stage = controller_marker_pose(ctrl["p"], ctrl["q"])
        except (KeyError, TypeError, ValueError):
            return False, "поза контроллера не читается"
        R_cam, t_cam, reproj = stable
        self.R, self.t = camera_to_stage(R_cam, t_cam, R_stage, t_stage)
        self.reproj_px = reproj
        self.locked = True
        return True, (f"камера зафиксирована в STAGE, репроекция {reproj:.2f} px")

    def save(self, path):
        if not self.locked:
            raise RuntimeError("связка ещё не зафиксирована")
        payload = {
            "R": self.R.tolist(), "t": self.t.tolist(),
            "marker_id": self.marker_id, "marker_m": self.marker_m,
            "hand": self.hand, "reproj_px": self.reproj_px,
        }
        path = Path(path)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        tmp.replace(path)

    def load(self, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if int(data["marker_id"]) != self.marker_id or abs(float(data["marker_m"]) - self.marker_m) > 1e-6:
            raise ValueError("в файле другой маркер или другой размер")
        rotation = np.asarray(data["R"], np.float64)
        translation = np.asarray(data["t"], np.float64)
        if rotation.shape != (3, 3) or translation.shape != (3,):
            raise ValueError("битая связка")
        if abs(np.linalg.det(rotation) - 1.0) > 1e-3:
            raise ValueError("битая связка")
        self.R, self.t = rotation, translation
        self.reproj_px = float(data.get("reproj_px") or 0)
        self.locked = True


def write_marker(path, marker_m, marker_id=0, pixels=1200):
    image = generate_marker(marker_id, pixels)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"не записан {path}")
    print(f"{path}: чёрный квадрат AprilTag 36h11 id={marker_id} должен быть {marker_m:.3f} м. "
          f"Белая кайма в этот размер не входит.", flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="PNG маркера для печати")
    parser.add_argument("--write", required=True)
    parser.add_argument("--marker-m", type=float, default=0.16)
    parser.add_argument("--marker-id", type=int, default=0)
    parser.add_argument("--pixels", type=int, default=1200)
    args = parser.parse_args()
    write_marker(args.write, args.marker_m, args.marker_id, args.pixels)
