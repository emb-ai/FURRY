import unittest

import numpy as np
import pyrealsense2 as rs

from fusion import BoneLengths, OneEuro, constrain_legs, fuse_joints, window_median


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


class LegConstraintTest(unittest.TestCase):
    def setUp(self):
        self.intr = intrinsics()
        self.hip = np.array([0.0, 0.0, 2.70], np.float32)
        # колено поднято: на 0.25 м ближе к камере, длина бедра 0.40
        self.knee = self.hip + np.array([0.0, np.sqrt(0.40 ** 2 - 0.25 ** 2), -0.25], np.float32)
        self.lengths = BoneLengths(min_samples=1)
        self.lengths.update({11: self.hip, 13: self.knee})

    def pixel(self, p):
        return tuple(rs.rs2_project_point_to_pixel(self.intr, p.tolist()))

    def test_bone_length_corrects_noisy_knee_depth(self):
        noisy = self.knee.copy()
        noisy[2] += 0.06  # шум глубины колена; 2D точное
        raw = {11: self.hip, 13: noisy}
        out = constrain_legs(raw, {13: self.pixel(self.knee)}, self.intr, self.lengths)
        self.assertLess(abs(float(out[13][2] - self.knee[2])), 0.01)

    def test_unknown_length_leaves_raw(self):
        raw = {11: self.hip, 13: self.knee + np.float32(0.05)}
        out = constrain_legs(raw, {13: self.pixel(self.knee)}, self.intr, BoneLengths())
        np.testing.assert_array_equal(out[13], raw[13])


class OneEuroLagTest(unittest.TestCase):
    def test_slow_walk_lag_under_60_ms(self):
        """Шаг 0.3 м/с на 6 Гц: дефолтный фильтр отстаёт меньше 60 мс (было 133)."""
        f, dt, v = OneEuro(), 1 / 6, 0.3
        for k in range(60):
            out = f(np.array([v * k * dt, 0.0, 0.0]), k * dt)
        lag_s = (v * 59 * dt - out[0]) / v
        self.assertLess(lag_s, 0.060)
        self.assertGreater(lag_s, 0.0)


if __name__ == "__main__":
    unittest.main()
