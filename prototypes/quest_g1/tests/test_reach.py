"""Scripted cup reach: command, recording contract, and a short tracked rollout."""
from pathlib import Path
import tempfile
import unittest

import numpy as np

from g1_sim.controller import JOINTS, standing_command
from g1_sim.reach import (
    BLEND, CUP_X, CUP_Y, CUP_Z, SCENE, ScriptedReach, arm_target_delta, generate, place_cup, sample_cup,
)
from g1_sim.scene import make_model
from g1_sim.state import SCHEMA


class ReachCommandTests(unittest.TestCase):
    def test_cup_samples_stay_on_the_table_region(self):
        rng = np.random.default_rng(0)
        for _ in range(100):
            cup = sample_cup(rng)
            self.assertGreaterEqual(cup[0], CUP_X[0])
            self.assertLessEqual(cup[0], CUP_X[1])
            self.assertGreaterEqual(cup[1], CUP_Y[0])
            self.assertLessEqual(cup[1], CUP_Y[1])
            self.assertEqual(cup[2], CUP_Z)

    def test_shoulder_roll_tracks_cup_y_and_the_blend_starts_at_standing(self):
        model, _ = make_model(SCENE)
        low = ScriptedReach(model, np.array([0.40, CUP_Y[0], CUP_Z]))
        high = ScriptedReach(model, np.array([0.40, CUP_Y[1], CUP_Z]))
        self.assertGreater(high.goal[29], low.goal[29])
        self.assertAlmostEqual(high.goal[28], standing_command()[28] - 0.95)
        np.testing.assert_allclose(high.goal[:6], standing_command()[:6])
        self.assertEqual(len(high.goal), 35)
        data = type("Data", (), {"time": 0.0})()
        command, grip = low(model, data, None)
        np.testing.assert_allclose(command, standing_command())
        self.assertEqual(grip, 0.0)
        data.time = BLEND
        command, grip = low(model, data, None)
        np.testing.assert_allclose(command, low.goal)
        limits = np.array([model.jnt_range[model.joint(name).id] for name in JOINTS[22:29]])
        self.assertTrue(np.all(low.goal[28:35] >= limits[:, 0] - 1e-9))
        self.assertTrue(np.all(low.goal[28:35] <= limits[:, 1] + 1e-9))
        self.assertGreater(arm_target_delta(CUP_Y[1])[1], arm_target_delta(CUP_Y[0])[1])


class ReachRolloutTests(unittest.TestCase):
    def test_scripted_reach_approaches_the_cup_and_exports_state_v1(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = generate(Path(directory), demos=1, seconds=3.0, seed=2)
            row = manifest["demos"][0]
            export = np.load(Path(directory) / row["state"], allow_pickle=False)
            self.assertEqual(str(export["schema"]), SCHEMA)
            self.assertEqual(export["state"].shape, (60, 48))
            self.assertEqual(export["command"].shape, (60, 35))
        self.assertEqual(row["scene"], SCENE)
        self.assertLess(row["final_tip_distance_m"], row["initial_tip_distance_m"] - 0.08)
        self.assertGreater(row["minimum_pelvis_height_m"], 0.70)
        self.assertLess(row["maximum_pelvis_tilt_deg"], 15)
        self.assertEqual(row["mujoco_warnings"], 0)
        self.assertEqual(row["state_dim"], 48)
        self.assertEqual(row["state_rows"], 60)


class PlaceCupTests(unittest.TestCase):
    def test_place_cup_is_read_back_from_the_free_joint(self):
        model, _ = make_model(SCENE)
        import mujoco
        data = mujoco.MjData(model)
        mujoco.mj_resetData(model, data)
        place_cup(model, data, [0.41, -0.11, CUP_Z])
        address = int(model.joint("cup_free").qposadr[0])
        np.testing.assert_allclose(data.qpos[address:address+3], [0.41, -0.11, CUP_Z])


if __name__ == "__main__":
    unittest.main()
