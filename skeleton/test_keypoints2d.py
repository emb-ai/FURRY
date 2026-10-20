import unittest

import cv2
import numpy as np

from keypoints2d import wire_keypoints

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

if __name__ == "__main__":
    unittest.main()
