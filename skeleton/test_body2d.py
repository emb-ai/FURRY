"""Synthetic checks of the headset-free 2D body solve (body2d.py)."""
import unittest

import numpy as np

from body2d import COCO, INDEX, NAMES, Body, BodySolver, camera_from_gravity, camera_from_pitch

LEGS = [INDEX[n] for n in ("lknee", "rknee", "lankle", "rankle")]


def bone(hip_deg, knee_deg):
    h, k = np.radians(hip_deg), np.radians(hip_deg - knee_deg)
    return np.array([0, -np.cos(h), np.sin(h)]), np.array([0, -np.cos(k), np.sin(k)])


def directions(psi, lift=(0, 0), arm=0):
    """Left knee lift (hip, knee flexion) and both arms raised forward, facing heading psi."""
    c, s = np.cos(psi), np.sin(psi)
    turn = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    a = np.radians(arm)
    raised = np.array([0, -np.cos(a), np.sin(a)])
    d = [*bone(*lift), *bone(0, 0), raised, raised, raised, raised]
    return np.array([turn @ v for v in d])


def observe(solver, psi, d, rng=None, noise=0.):
    """Pixels of a standing operator 3 m in front of the camera."""
    x = np.r_[0., 0., -3., psi, 0., 0., 0.]
    p = solver.points(x, d)
    x[1] = solver.body.ankle_height - min(p[5, 1], p[6, 1])
    p = solver.points(x, d)
    uv, _ = solver.camera.project(p)
    if rng is not None:
        uv = uv + rng.normal(0, noise, uv.shape)
    return {c: (uv[INDEX[n]][0], uv[INDEX[n]][1], .9) for n, c in COCO.items()}, p


class BodySolveTest(unittest.TestCase):
    def setUp(self):
        self.camera = camera_from_pitch(12, 1.2, 910, 905, 640, 360)

    def track(self, psi, poses, rng=None, noise=0.):
        solver = BodySolver(self.camera, Body(1.75))
        errors = []
        for n, (lift, arm) in enumerate(poses):
            kp, truth = observe(solver, psi, directions(psi, lift, arm), rng, noise)
            pose = solver.solve(kp, n / 30)
            self.assertIsNotNone(pose)
            got = np.array([pose.points[name] for name in NAMES])
            errors.append(np.linalg.norm(got - truth, axis=1))
        return np.array(errors)

    def test_standing_is_exact(self):
        for psi in (0., np.radians(40)):
            e = self.track(psi, [((0, 0), 0)] * 5)
            self.assertLess(e.max(), .02)

    def test_tracked_knee_lift_recovers_on_both_views(self):
        # Up to 90 deg and back down. Near standing the two ray-sphere roots
        # are close and both anatomical, so small lifts may lag; the branch
        # must not stay locked once the limits rule the wrong one out.
        lifts = [((a, a), 0) for a in np.linspace(0, 90, 20)] + [((a, a), 0) for a in np.linspace(90, 0, 20)]
        for psi in (0., np.radians(40)):
            e = self.track(psi, [((0, 0), 0)] * 3 + lifts)[:, LEGS].max(1)
            self.assertLess(np.median(e), .03, f"heading {np.degrees(psi):.0f}")
            self.assertLess(e[-15:].max(), .06, f"heading {np.degrees(psi):.0f}: stuck after lowering the leg")

    def test_frontal_forward_arm_raise(self):
        e = self.track(0., [((0, 0), a) for a in np.linspace(0, 120, 20)])
        self.assertLess(np.median(e.max(1)), .05)

    def test_noisy_keypoints_stay_bounded(self):
        # Regression bound of this fixture, not an accuracy claim.
        rng = np.random.default_rng(3)
        poses = [((a, a), 0) for a in np.linspace(0, 80, 15)] + [((a, a), 0) for a in np.linspace(80, 0, 15)]
        e = self.track(0., poses, rng, 3.)[:, LEGS]
        self.assertLess(np.median(e), .12)

    def test_missing_hips_gives_no_pose(self):
        solver = BodySolver(self.camera)
        kp, _ = observe(solver, 0., directions(0.))
        del kp[11]
        self.assertIsNone(solver.solve(kp, 0.))

    def test_gravity_camera_is_level(self):
        cam = camera_from_gravity([0, -9.8, 0], 1.3, 900, 900, 640, 360)
        np.testing.assert_allclose(cam.R @ [0, -1, 0], [0, 1, 0], atol=1e-9)  # camera up is world up
        np.testing.assert_allclose(cam.R @ [0, 0, 1], [0, 0, -1], atol=1e-9)  # looks along world -Z
        self.assertAlmostEqual(np.linalg.det(cam.R), 1.)
        tilted = camera_from_pitch(20, 1.3, 900, 900, 640, 360)
        self.assertAlmostEqual(tilted.forward[1], -np.sin(np.radians(20)))


if __name__ == "__main__":
    unittest.main()
