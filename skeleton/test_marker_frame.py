import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from marker_frame import (MarkerSession, camera_to_stage, controller_marker_pose,
                          generate_marker, object_points, pose_from_corners,
                          to_stage_relative)


def rotation_y(deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def facing_camera(tilt_deg=0):
    """Маркер смотрит в камеру: его +Z к началу координат, +Y вверх кадра."""
    rx = np.diag([1.0, -1.0, -1.0])
    return rotation_y(tilt_deg) @ rx


def render_marker(R, t, K, size_m=0.16, pixels=400):
    """Чёрный квадрат маркера на белом поле, в ту же сторону, что и картинка."""
    marker = generate_marker(0, pixels)
    src = np.array([[0, 0], [pixels, 0], [pixels, pixels], [0, pixels]], np.float32)
    dst, _ = cv2.projectPoints(object_points(size_m), cv2.Rodrigues(R)[0], t.reshape(3, 1), K, None)
    dst = dst.reshape(4, 2).astype(np.float32)
    canvas = np.full((720, 1280), 255, np.uint8)
    warp = cv2.warpPerspective(marker, cv2.getPerspectiveTransform(src, dst), (1280, 720),
                               borderValue=255)
    mask = np.zeros(canvas.shape, np.uint8)
    cv2.fillConvexPoly(mask, np.round(dst).astype(np.int32), 255)
    canvas[mask > 0] = warp[mask > 0]
    return canvas, dst


class MarkerFrameTest(unittest.TestCase):
    def test_rigid_mapping_and_marker_origin(self):
        R_cam, t_cam = rotation_y(20), np.array([0.1, -0.2, 1.5])
        R_stage_cam, t_stage_cam = rotation_y(-35), np.array([1.0, 2.0, 3.0])
        R_stage, t_stage = R_stage_cam @ R_cam, R_stage_cam @ t_cam + t_stage_cam
        R, t = camera_to_stage(R_cam, t_cam, R_stage, t_stage)
        np.testing.assert_allclose(R, R_stage_cam, atol=1e-9)
        np.testing.assert_allclose(t, t_stage_cam, atol=1e-9)
        np.testing.assert_allclose(R @ t_cam + t, t_stage, atol=1e-9)

    def test_controller_touch_uses_its_axes(self):
        R, t = controller_marker_pose([1, 2, 3], [1, 0, 0, 0])
        np.testing.assert_allclose(R, np.eye(3), atol=1e-9)
        np.testing.assert_allclose(t, [1, 2, 3])
        # Контроллер смотрит вдоль −Z. Маркерный +Z смотрит на камеру, это +Z контроллера.
        quarter = np.array([np.cos(np.pi / 4), 0, np.sin(np.pi / 4), 0])
        R, _ = controller_marker_pose([0, 0, 0], quarter)
        np.testing.assert_allclose(R[:, 2], [1, 0, 0], atol=1e-9)

    def test_pelvis_relative_drops_translation(self):
        points = {11: np.array([0.0, 0.0, 2.0]), 12: np.array([0.2, 0.0, 2.0]),
                  13: np.array([0.0, 0.4, 2.1])}
        R, t = rotation_y(90), np.array([5.0, -1.0, 0.4])
        rel, pelvis = to_stage_relative(points, R, t)
        pelvis_cam = 0.5 * (points[11] + points[12])
        np.testing.assert_allclose(rel[13], R @ (points[13] - pelvis_cam), atol=1e-9)
        np.testing.assert_allclose(pelvis, R @ pelvis_cam + t, atol=1e-9)
        self.assertIsNone(to_stage_relative({11: points[11]}, R, t))

    def test_detected_pose_matches_projection(self):
        K = np.array([[900, 0, 640], [0, 900, 360], [0, 0, 1]], np.float64)
        R, t = facing_camera(15), np.array([0.05, -0.02, 1.4])
        image, _ = render_marker(R, t, K)
        session = MarkerSession(0.16)
        pose = session.observe(image, K, None, now=0)
        self.assertIsNotNone(pose)
        found_R, found_t, reproj = pose
        self.assertLess(reproj, 1.5)
        np.testing.assert_allclose(found_t, t, atol=0.01)
        self.assertLess(np.degrees(np.arccos(np.clip((np.trace(found_R.T @ R) - 1) / 2, -1, 1))), 1.5)

    def test_lock_requires_stable_marker_and_controller(self):
        K = np.array([[900, 0, 640], [0, 900, 360], [0, 0, 1]], np.float64)
        R, t = facing_camera(), np.array([0.0, 0.0, 1.5])
        image, _ = render_marker(R, t, K)
        session = MarkerSession(0.16, hand="right")
        uplink = {"ctrl_r": {"p": [1.0, 2.0, 3.0], "q": [1, 0, 0, 0], "tracked": True}}
        session.observe(image, K, None, now=0)
        ok, _ = session.try_lock(uplink)
        self.assertFalse(ok)
        for i in range(4):
            session.observe(image, K, None, now=0.05 * (i + 1))
        ok, message = session.try_lock(None)
        self.assertFalse(ok)
        self.assertIn("контроллер", message)
        ok, message = session.try_lock(uplink)
        self.assertTrue(ok, message)
        self.assertTrue(session.locked)
        np.testing.assert_allclose(session.R @ t + session.t, [1, 2, 3], atol=0.01)
        shaken = list(session.samples[-1])
        shaken[2] = shaken[2] + np.array([0.05, 0, 0])
        session.samples[-1] = tuple(shaken)
        session.locked = False
        ok, message = session.try_lock(uplink)
        self.assertFalse(ok)
        self.assertIn("дрожит", message)

    def test_session_roundtrip_rejects_other_size(self):
        session = MarkerSession(0.16)
        session.R, session.t = np.eye(3), np.array([1.0, 0.0, 0.0])
        session.reproj_px = 0.4
        session.locked = True
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "session.json"
            session.save(path)
            loaded = MarkerSession(0.16)
            loaded.load(path)
            np.testing.assert_allclose(loaded.t, [1, 0, 0])
            with self.assertRaises(ValueError):
                MarkerSession(0.10).load(path)


if __name__ == "__main__":
    unittest.main()
