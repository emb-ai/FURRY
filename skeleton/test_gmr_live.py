"""GMR retargeting of a camera-only body pose: joint angles follow the human's.

Needs MuJoCo, Mink and daqp (prototypes/quest_g1/android/requirements-gmr-test.txt);
skipped without them.
"""
import unittest

import numpy as np

try:
    import mink  # noqa: F401
    import mujoco  # noqa: F401
except ImportError:
    mink = None

from body2d import Body, BodySolver, camera_from_pitch
from test_body2d import directions, observe


@unittest.skipIf(mink is None, "MuJoCo/Mink not installed")
class RetargetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from gmr_live import Retargeter
        cls.body = Body(1.75)
        cls.retargeter = Retargeter(cls.body)
        cls.camera = camera_from_pitch(12, 1.2, 910, 905, 640, 360)

    def angles(self, lift=(0, 0), arm=0, heading=0.):
        solver = BodySolver(self.camera, self.body)
        kp, _ = observe(solver, heading, directions(heading, lift, arm))
        pose = solver.solve(kp, 0.)
        self.retargeter.reset()
        for _ in range(3):
            q = self.retargeter.solve(pose)
        m = self.retargeter.model
        return {j: np.degrees(q[m.jnt_qposadr[m.joint(j).id]]) for j in
                ("left_hip_pitch_joint", "left_knee_joint", "right_hip_pitch_joint", "right_knee_joint",
                 "left_shoulder_pitch_joint", "right_shoulder_pitch_joint", "left_elbow_joint")}, q

    def test_standing_is_the_zero_pose(self):
        a, q = self.angles()
        for j in ("left_hip_pitch_joint", "left_knee_joint", "right_hip_pitch_joint", "right_knee_joint",
                  "left_shoulder_pitch_joint"):
            self.assertLess(abs(a[j]), 3, j)
        self.assertAlmostEqual(a["left_elbow_joint"], 90, delta=5)  # G1: 90 deg is a straight arm
        self.assertAlmostEqual(q[2], .793, delta=.03)  # pelvis at standing height

    def test_knee_lift_maps_to_hip_and_knee(self):
        for heading in (0., np.radians(40)):
            a, _ = self.angles((60, 70), heading=heading)
            self.assertAlmostEqual(a["left_hip_pitch_joint"], -60, delta=4)
            self.assertAlmostEqual(a["left_knee_joint"], 70, delta=4)
            self.assertLess(abs(a["right_knee_joint"]), 3)

    def test_high_knee_and_forward_arms(self):
        a, _ = self.angles((90, 90))
        self.assertAlmostEqual(a["left_hip_pitch_joint"], -90, delta=4)
        self.assertAlmostEqual(a["left_knee_joint"], 90, delta=4)
        a, _ = self.angles(arm=90)
        self.assertAlmostEqual(a["left_shoulder_pitch_joint"], -90, delta=5)
        self.assertAlmostEqual(a["right_shoulder_pitch_joint"], -90, delta=5)

    def test_render_camera_matches_operator_camera(self):
        view = self.retargeter.camera_view(self.camera)
        self.assertAlmostEqual(view["elevation"], -12, delta=1e-6)
        self.assertAlmostEqual(view["fovy"], np.degrees(2 * np.arctan(360 / 905)), delta=1e-6)


if __name__ == "__main__":
    unittest.main()
