import unittest
import numpy as np
from camera_validation import lift_events, lift_summary

DT = .01


def lift(seconds, peak, robot_scale, drop_after=None, n=400, start=50):
    """One left-foot lift; the robot follows at robot_scale and lands after drop_after s."""
    ref = np.zeros((n, 2)); robot = np.zeros((n, 2)); contact = np.ones((n, 2), bool)
    k = round(seconds / DT)
    ref[start:start + k, 0] = peak
    follow = k if drop_after is None else round(drop_after / DT)
    robot[start:start + follow, 0] = peak * robot_scale
    contact[start:start + follow, 0] = False
    return ref, robot, contact


class LiftMetric(unittest.TestCase):
    def test_follow_then_drop(self):
        ref, robot, contact = lift(2., .10, .8, drop_after=.5)
        (e,) = lift_events(ref, robot, contact, DT)
        self.assertAlmostEqual(e['seconds'], 2.)
        self.assertAlmostEqual(e['peak_ratio'], .8)
        self.assertAlmostEqual(e['retention_1s'], 0.)
        self.assertAlmostEqual(e['contact_fraction'], .75)
        s = lift_summary([e], ref, robot, contact, DT)
        self.assertEqual((s['events'], s['held_events'], s['retention_1s_events']), (1, 1, 1))
        self.assertAlmostEqual(s['lifted_contact_fraction'], .75)
        self.assertAlmostEqual(s['lifted_shortfall_cm_mean'], .25 * 2 + .75 * 10)
        moving = np.zeros(len(ref), bool); moving[:100] = True  # lift frames 50..99 moving
        s = lift_summary([e], ref, robot, contact, DT, moving)
        self.assertAlmostEqual(s['lifted_seconds_moving'], .5)
        self.assertAlmostEqual(s['lifted_contact_fraction_moving'], 0.)
        self.assertAlmostEqual(s['lifted_contact_fraction_still'], 1.)

    def test_short_and_low_lifts(self):
        ref, robot, contact = lift(.3, .10, 1.)
        (e,) = lift_events(ref, robot, contact, DT)
        self.assertNotIn('retention_1s', e)
        self.assertEqual(lift_summary([e], ref, robot, contact, DT)['held_events'], 0)
        ref, robot, contact = lift(1., .04, 1.)  # above LIFT_ON, below LIFT_PEAK
        self.assertEqual(lift_events(ref, robot, contact, DT), [])


if __name__ == '__main__':
    unittest.main()
