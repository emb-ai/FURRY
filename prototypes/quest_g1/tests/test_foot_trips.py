"""Trip scoring must count only landings during commanded swing."""
import tempfile
import unittest
from pathlib import Path
import mujoco
import numpy as np
from scripts.analyze_foot_trips import MODEL, analyze, command_poses, reference_feet, score_block, touchdowns
from g1_sim.controller import standing_command


class FootTripTests(unittest.TestCase):
    def test_touchdown_needs_real_flight_and_load(self):
        force = np.r_[200, 0, 0, 0, 120, 0, 0, 120, 0, 0, 0, 30, 0, 0, 0, 80.]
        # Two-row flights and light contacts are not landings.
        self.assertEqual(touchdowns(force), [4, 15])

    def test_swing_landing_is_a_trip_and_stance_landing_is_not(self):
        rows = 40
        height, speed = np.zeros((rows, 2)), np.zeros((rows, 2))
        force, brake = np.full((rows, 2), 300.), np.zeros((rows, 2))
        toe = np.full((rows, 2), np.nan)
        # Left lands at row 30 while commanded 3 cm up and moving 0.8 m/s.
        force[25:30, 0] = 0
        height[20:35, 0], speed[20:35, 0], brake[30, 0] = .03, .8, 120
        toe[28, 0] = -.01
        # Right lands at row 30 too, but its commanded foot is already planted.
        force[25:30, 1] = 0
        out = score_block(height, speed, force, brake, toe, warm_rows=10)
        self.assertEqual((out['touchdowns'], out['trips'], out['braking_trips']), (2, 1, 1))
        self.assertEqual(out['trips_with_toe_below'], 1)
        self.assertEqual((out['toe_rows'], out['toe_below_rows'], out['swing_toe_below_rows']), (1, 1, 1))
        # Landings inside warmup are ignored.
        self.assertEqual(score_block(height, speed, force, brake, toe, warm_rows=35)['touchdowns'], 0)

    def test_command_reference_feet(self):
        model = mujoco.MjModel.from_xml_path(str(MODEL))
        command = np.tile(standing_command(), (40, 1))
        height, speed = reference_feet(model, command_poses(command, .01), .01)
        self.assertLess(abs(height).max(), 1e-9)
        self.assertLess(speed.max(), 1e-9)
        command[:, 0] = .5
        poses = command_poses(command, .01)
        np.testing.assert_allclose(poses[-1, 0], .5*.4, atol=1e-9)
        _, speed = reference_feet(model, poses, .01)
        np.testing.assert_allclose(speed[5:-5], .5, atol=1e-6)

    def test_csv_roundtrip_and_old_format(self):
        rows = 300
        command = np.tile(standing_command(), (rows, 1))
        names = ['sim_s', 'segment', 'applying'] + [f'cmd_{k}' for k in range(35)]
        names += [f'{f}_{s}' for f in ('floor_force', 'floor_brake', 'toe_target') for s in 'lr']
        force = np.full((rows, 2), 300.)
        force[200:205, 0] = 0  # A standing re-step is not a trip.
        table = np.c_[np.arange(rows)*.01, np.ones(rows), np.ones(rows), command,
                      force, np.zeros((rows, 2)), np.full((rows, 2), np.nan)]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'replay.csv'
            np.savetxt(path, table, delimiter=',', header=','.join(names), comments='')
            result = analyze(path)
            self.assertEqual((result['seconds'], result['trips']), (2.0, 0))
            self.assertAlmostEqual(result['touchdowns_per_s'], .5)
            self.assertIsNone(result['toe_below_share'])
            old = Path(folder)/'old.csv'
            np.savetxt(old, table[:, :38], delimiter=',', header=','.join(names[:38]), comments='')
            with self.assertRaises(SystemExit):
                analyze(old)


if __name__ == '__main__':
    unittest.main()
