import json
import unittest

import cv2
import numpy as np

from keypoints2d import wire_keypoints
from skeleton_server import packet

INTR = [910.0, 905.0, 640.0, 360.0]


class Keypoints2DTest(unittest.TestCase):
    def test_drops_low_confidence_and_keeps_pixels(self):
        xy = np.tile([[100.0, 200.0]], (17, 1))
        conf = np.full(17, .9)
        conf[13] = .01
        conf[14] = np.nan
        out = wire_keypoints(xy, conf, INTR)
        self.assertNotIn(13, out)
        self.assertNotIn(14, out)
        self.assertEqual(out[15], (100.0, 200.0, .9))

    def test_undistorts_with_opencv_model(self):
        K = np.array([[INTR[0], 0, INTR[2]], [0, INTR[1], INTR[3]], [0, 0, 1]])
        dist = np.array([.1, -.05, 0, 0, 0])
        ray = np.array([[[.3, -.2, 1.0]]])
        distorted, _ = cv2.projectPoints(ray, np.zeros(3), np.zeros(3), K, dist)
        out = wire_keypoints(distorted.reshape(1, 2), [.9], INTR, dist)
        self.assertAlmostEqual(out[0][0], INTR[0] * .3 + INTR[2], places=2)
        self.assertAlmostEqual(out[0][1], INTR[1] * -.2 + INTR[3], places=2)

    def test_packet_fields(self):
        kp = {11: (1, 2, .9), 13: (3, 4, .8), 9: (5, 6, .7)}
        R = np.eye(3)
        p = packet({}, {}, {}, 1, 1, 30, "test", kp2d=kp, intrinsics=INTR,
                   stage_from_camera=(R, [1, 2, 3]))
        self.assertEqual(p["intr"], INTR)
        self.assertEqual(p["kp2d"], {"lhip": [1.0, 2.0, .9], "lknee": [3.0, 4.0, .8]})
        self.assertEqual(p["stage_from_camera"]["R"], [1, 0, 0, 0, 1, 0, 0, 0, 1])
        self.assertEqual(p["stage_from_camera"]["t"], [1.0, 2.0, 3.0])
        json.dumps(p, allow_nan=False)
        plain = packet({}, {}, {}, 1, 1, 30, "test")
        self.assertNotIn("kp2d", plain)
        self.assertNotIn("stage_from_camera", plain)


if __name__ == "__main__":
    unittest.main()
