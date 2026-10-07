"""Cup and push-T success rules, seeded starts, and a short tracked rollout."""
from pathlib import Path
import tempfile
import unittest

import mujoco
import numpy as np

from g1_sim.controller import Controller
from g1_sim.reach import sample_cup
from g1_sim.scene import make_model
from g1_sim.tasks import (
    CUP_EVAL_X, CUP_EVAL_Y, CUP_EVAL_YAW, CUP_GRIP, CUP_HOLD_M, CUP_SUCCESS_Z, FALL_HEIGHT_M,
    ROOT_XY, ROOT_YAW, SCHEMA, TARGET_AHEAD, T_X, T_Y, T_YAW, coverage, cup_picked, fell, reset, run,
    sample_start,
)


class StartSamplingTests(unittest.TestCase):
    def test_demo_cup_matches_the_reach_box_and_eval_is_wider(self):
        for seed in range(30):
            demo = sample_start("cup", seed, "demo")
            np.testing.assert_allclose(demo["cup_xyz"], sample_cup(np.random.default_rng(seed)))
            self.assertEqual(demo["cup_yaw"], 0.0)
            self.assertEqual((demo["root_x"], demo["root_y"], demo["root_yaw"]), (0.0, 0.0, 0.0))
            again = sample_start("cup", seed, "eval")
            self.assertEqual(again, sample_start("cup", seed, "eval"))
            self.assertGreaterEqual(again["cup_xyz"][0], CUP_EVAL_X[0])
            self.assertLess(again["cup_xyz"][0], CUP_EVAL_X[1])
            self.assertGreaterEqual(again["cup_xyz"][1], CUP_EVAL_Y[0])
            self.assertLess(again["cup_xyz"][1], CUP_EVAL_Y[1])
            self.assertGreaterEqual(again["cup_yaw"], CUP_EVAL_YAW[0])
            self.assertLess(again["cup_yaw"], CUP_EVAL_YAW[1])
            self.assertGreaterEqual(again["root_x"], ROOT_XY[0])
            self.assertLess(again["root_x"], ROOT_XY[1])
            self.assertGreaterEqual(again["root_yaw"], ROOT_YAW[0])
            self.assertLess(again["root_yaw"], ROOT_YAW[1])
        self.assertNotEqual(sample_start("cup", 0, "eval")["cup_xyz"], sample_start("cup", 1, "eval")["cup_xyz"])

    def test_push_t_demo_is_the_scene_pose_and_eval_keeps_a_gap(self):
        demo = sample_start("push_t", 4, "demo")
        self.assertEqual(demo["t_xyz"], [0.65, 0.0, 0.101])
        self.assertEqual(demo["target_xyz"], [1.55, 0.0, 0.002])
        self.assertEqual(demo["t_yaw"], 0.0)
        for seed in range(30):
            start = sample_start("push_t", seed, "eval")
            self.assertEqual(start, sample_start("push_t", seed, "eval"))
            self.assertGreaterEqual(start["t_xyz"][0], T_X[0])
            self.assertLess(start["t_xyz"][0], T_X[1])
            self.assertGreaterEqual(start["t_xyz"][1], T_Y[0])
            self.assertLess(start["t_xyz"][1], T_Y[1])
            self.assertGreaterEqual(start["t_yaw"], T_YAW[0])
            self.assertLess(start["t_yaw"], T_YAW[1])
            ahead = start["target_xyz"][0] - start["t_xyz"][0]
            self.assertGreaterEqual(ahead, TARGET_AHEAD[0])
            self.assertLess(ahead, TARGET_AHEAD[1])
        self.assertNotEqual(sample_start("push_t", 0, "eval")["t_xyz"], sample_start("push_t", 1, "eval")["t_xyz"])


class PredicateTests(unittest.TestCase):
    def test_resting_cup_is_not_a_pickup_and_the_threshold_is_geometric(self):
        model, _ = make_model("cup")
        data = mujoco.MjData(model)
        controller = Controller(model)
        reset(model, data, controller, sample_start("cup", 0, "demo"))
        self.assertFalse(cup_picked(model, data, 0.0))
        self.assertFalse(cup_picked(model, data, 1.0))
        self.assertLess(float(data.body("cup").xpos[2]), CUP_SUCCESS_Z)
        address = int(model.joint("cup_free").qposadr[0])
        tip = np.array(data.body("right_hand_middle_finger_tip").xpos)
        data.qpos[address:address+3] = tip
        data.qpos[address+3:address+7] = [1, 0, 0, 0]
        mujoco.mj_forward(model, data)
        self.assertFalse(cup_picked(model, data, 1.0))
        # Lift the kinematic tree so the fingertip itself clears the pickup height.
        data.qpos[2] += CUP_SUCCESS_Z - float(data.body("right_hand_middle_finger_tip").xpos[2]) + 0.01
        mujoco.mj_forward(model, data)
        data.qpos[address:address+3] = data.body("right_hand_middle_finger_tip").xpos
        mujoco.mj_forward(model, data)
        distance = float(np.linalg.norm(data.body("cup").xpos - data.body("right_hand_middle_finger_tip").xpos))
        self.assertLessEqual(distance, CUP_HOLD_M)
        self.assertGreaterEqual(float(data.body("cup").xpos[2]), CUP_SUCCESS_Z)
        self.assertFalse(cup_picked(model, data, CUP_GRIP - 0.01))
        self.assertTrue(cup_picked(model, data, CUP_GRIP))

    def test_aligned_t_covers_the_target_and_a_distant_t_does_not(self):
        model, _ = make_model("push_t")
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        self.assertLess(coverage(data), 0.05)
        start = sample_start("push_t", 0, "demo")
        start["t_xyz"] = list(start["target_xyz"])
        start["t_xyz"][2] = 0.101
        start["t_yaw"] = start["target_yaw"]
        controller = Controller(model)
        reset(model, data, controller, start)
        self.assertAlmostEqual(coverage(data), 1.0)
        for seed in range(10):
            reset(model, data, controller, sample_start("push_t", seed, "eval"))
            self.assertLess(coverage(data), 0.25)

    def test_low_pelvis_is_a_fall(self):
        model, _ = make_model("cup")
        data = mujoco.MjData(model)
        controller = Controller(model)
        reset(model, data, controller, sample_start("cup", 0, "demo"))
        self.assertFalse(fell(model, data))
        data.qpos[2] = FALL_HEIGHT_M - 0.01
        mujoco.mj_forward(model, data)
        self.assertTrue(fell(model, data))


class ResetTests(unittest.TestCase):
    def test_the_same_seed_replays_after_steps_and_a_later_seed_moves_the_mark(self):
        model, _ = make_model("push_t")
        data = mujoco.MjData(model)
        controller = Controller(model)
        start = sample_start("push_t", 2, "eval")
        reset(model, data, controller, start)
        qpos = data.qpos.copy()
        target = data.body("t_target").xpos.copy()
        for _ in range(5):
            controller.step(data)
        reset(model, data, controller, start)
        np.testing.assert_allclose(data.qpos, qpos)
        np.testing.assert_allclose(data.qvel, 0)
        self.assertEqual(data.time, 0)
        self.assertEqual(controller.step_index, 0)
        np.testing.assert_allclose(data.body("t_target").xpos, target)
        other = sample_start("push_t", 3, "eval")
        reset(model, data, controller, other)
        self.assertGreater(abs(float(data.body("t_target").xpos[0]) - float(target[0])), 1e-3)


class RolloutTests(unittest.TestCase):
    def test_standing_push_t_times_out_without_covering_the_mark(self):
        with tempfile.TemporaryDirectory() as directory:
            metrics = run("push_t", 1, seconds=0.4, distribution="eval", source="standing",
                          record=Path(directory) / "trial.npz")
            stored = np.load(Path(directory) / "trial.npz", allow_pickle=False)
            self.assertEqual(str(stored["scene"]), "push_t")
            self.assertGreater(stored["time"].shape[0], 0)
        self.assertEqual(metrics["schema"], SCHEMA)
        self.assertEqual(metrics["outcome"], "timeout")
        self.assertFalse(metrics["success"])
        self.assertLess(metrics["maximum_coverage"], 0.25)
        self.assertGreater(metrics["minimum_pelvis_height_m"], 0.70)
        self.assertEqual(metrics["mujoco_warnings"], 0)
        self.assertIsNone(metrics["success_time_s"])


if __name__ == "__main__":
    unittest.main()
