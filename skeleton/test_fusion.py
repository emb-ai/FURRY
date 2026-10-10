import unittest

import numpy as np
import pyrealsense2 as rs

from fusion import fuse_joints, window_median


def intrinsics():
    intr = rs.intrinsics()
    intr.width, intr.height = 1280, 720
    intr.fx = intr.fy = 931.0
    intr.ppx, intr.ppy = 640.0, 360.0
    intr.model = rs.distortion.none
    intr.coeffs = [0.0] * 5
    return intr


def scene():
    """Таз и голень на 2.70 м, справа от голени пол на 3.00 м (в полосе ±0.5)."""
    depth = np.full((720, 1280), 6.0, np.float32)  # дальний фон вне полосы
    depth[300:420, 560:720] = 2.70                   # таз
    depth[420:720, 600:700] = 3.00                   # пол за голенью
    depth[420:720, 600:640] = 2.70                   # голень: 40 px шириной
    return depth


class DistalDepthTest(unittest.TestCase):
    def test_floor_behind_ankle_does_not_pull_depth(self):
        depth = scene()
        # кейпоинт у края голени: в окне 9x9 голени 4 столбца из 9
        pts = {11: (620.0, 360.0), 12: (680.0, 360.0), 16: (640.0, 600.0)}
        out = fuse_joints(depth, pts, intrinsics())
        self.assertAlmostEqual(float(out[16][2]), 2.70, places=2)
        # медиана того же окна попадает на пол — исходная ошибка
        self.assertAlmostEqual(window_median(depth, 640, 600, 2.7), 3.00, places=2)

    def test_proximal_joints_keep_median(self):
        depth = scene()
        depth[355:358, 615:625] = 2.40  # тонкий объект перед бедром
        out = fuse_joints(depth, {11: (620.0, 360.0), 12: (680.0, 360.0)}, intrinsics())
        self.assertAlmostEqual(float(out[11][2]), 2.70, places=2)


if __name__ == "__main__":
    unittest.main()
