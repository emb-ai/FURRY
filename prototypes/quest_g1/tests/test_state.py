"""State layout and desktop-NPZ converter."""
from pathlib import Path
import tempfile
import unittest

import mujoco
import numpy as np

from g1_sim.controller import JOINTS
from g1_sim.scene import make_model
from g1_sim.state import (
    ACTION_HZ, COMMAND_NAMES, RECORD_HZ, ROBOT_NAMES, SCHEMA, body_field_names,
    convert_recording, encode_state, layout_names,
)


def write_episode(path, scene="lab", hands=True, rows=11):
    model, xml = make_model(scene, hands=hands)
    data = mujoco.MjData(model)
    apply_pose(model, data, yaw=0.0)
    qpos, qvel, command, grip = [], [], [], []
    for i in range(rows):
        data.qpos[model.joint("left_elbow_joint").qposadr[0]] = 0.01 * i
        qpos.append(data.qpos.copy())
        qvel.append(data.qvel.copy())
        command.append(np.full(len(COMMAND_NAMES), i, dtype=np.float64))
        grip.append(0.01 * i)
    np.savez_compressed(
        path, time=0.01 * np.arange(1, rows + 1), qpos=qpos, qvel=qvel,
        command=command, grip=grip, scene=scene, scene_xml=xml, metadata="{}")
    return model


def apply_pose(model, data, yaw=0.0, cup=(0.5, 0.0, 0.9)):
    mujoco.mj_resetData(model, data)
    data.qpos[:] = 0
    data.qvel[:] = 0
    data.qpos[:7] = [0, 0, 0.8, np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    for i, name in enumerate(JOINTS):
        data.qpos[model.joint(name).qposadr[0]] = 0.01 * (i + 1)
    if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cup_free") >= 0:
        address = int(model.joint("cup_free").qposadr[0])
        data.qpos[address:address+3] = cup
        data.qpos[address+3:address+7] = [1, 0, 0, 0]
    if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "t_box_free") >= 0:
        address = int(model.joint("t_box_free").qposadr[0])
        data.qpos[address:address+7] = [0.9, 0.4, 0.1, 1, 0, 0, 0]
    data.qvel[:6] = [0.4, 0, 0, 0, 0.2, 0]
    mujoco.mj_forward(model, data)


class StateLayoutTests(unittest.TestCase):
    def test_dimensions_and_field_order(self):
        expected = {"stand": 39, "cup": 48, "push_t": 57, "lab": 66}
        for scene, width in expected.items():
            model, _ = make_model(scene)
            names = layout_names(model)
            self.assertEqual(names[:len(ROBOT_NAMES)], ROBOT_NAMES)
            self.assertEqual(names[-1], "grip")
            self.assertEqual(len(names), width)
            self.assertEqual(len(COMMAND_NAMES), 35)
        lab = layout_names(make_model("lab")[0])
        cup = layout_names(make_model("cup")[0])
        push = layout_names(make_model("push_t")[0])
        self.assertLess(lab.index("cup_pos_x"), lab.index("t_box_pos_x"))
        self.assertLess(lab.index("t_box_pos_x"), lab.index("t_target_pos_x"))
        self.assertNotIn("t_box_pos_x", cup)
        self.assertNotIn("cup_pos_x", push)
        self.assertEqual(body_field_names("cup")[-6:], tuple(f"cup_rot6d_{i}" for i in range(6)))

    def test_pose_is_read_by_name_not_qpos_address(self):
        encoded = []
        addresses = []
        for hands in (True, False):
            model, _ = make_model("lab", hands=hands)
            data = mujoco.MjData(model)
            apply_pose(model, data)
            encoded.append(encode_state(model, data, grip=0.25))
            addresses.append(int(model.joint("cup_free").qposadr[0]))
        self.assertNotEqual(addresses[0], addresses[1])
        np.testing.assert_allclose(encoded[0], encoded[1])
        names = layout_names(make_model("lab")[0])
        self.assertAlmostEqual(encoded[0][names.index("left_elbow_joint")], 0.01 * (JOINTS.index("left_elbow_joint") + 1))
        self.assertAlmostEqual(encoded[0][names.index("grip")], 0.25)

    def test_object_pose_and_root_velocity_use_the_pelvis_frame(self):
        model, _ = make_model("lab")
        data = mujoco.MjData(model)
        names = layout_names(model)
        apply_pose(model, data, yaw=0.0)
        level = encode_state(model, data, 0)
        self.assertAlmostEqual(level[names.index("root_z")], 0.8)
        self.assertAlmostEqual(level[names.index("root_roll")], 0)
        self.assertAlmostEqual(level[names.index("cup_pos_x")], 0.5)
        self.assertAlmostEqual(level[names.index("cup_pos_z")], 0.1)
        np.testing.assert_allclose(level[names.index("cup_rot6d_0"):names.index("cup_rot6d_0")+6], [1, 0, 0, 1, 0, 0])
        self.assertAlmostEqual(level[names.index("root_lin_vel_x")], 0.4)
        self.assertAlmostEqual(level[names.index("root_ang_vel_y")], 0.2)

        apply_pose(model, data, yaw=np.pi / 2)
        turned = encode_state(model, data, 0)
        self.assertAlmostEqual(turned[names.index("cup_pos_x")], 0, places=6)
        self.assertAlmostEqual(turned[names.index("cup_pos_y")], -0.5, places=6)
        self.assertAlmostEqual(turned[names.index("cup_pos_z")], 0.1, places=6)
        np.testing.assert_allclose(
            turned[names.index("cup_rot6d_0"):names.index("cup_rot6d_0")+6],
            [0, 1, -1, 0, 0, 0], atol=1e-6)
        self.assertAlmostEqual(turned[names.index("root_lin_vel_x")], 0, places=6)
        self.assertAlmostEqual(turned[names.index("root_lin_vel_y")], -0.4, places=6)
        self.assertAlmostEqual(turned[names.index("root_ang_vel_y")], 0.2, places=6)

        data.qpos[3:7] = [np.cos(0.15), 0, np.sin(0.15), 0]
        pitched = encode_state(model, data, 0)
        self.assertAlmostEqual(pitched[names.index("root_pitch")], 0.3, places=6)
        self.assertAlmostEqual(pitched[names.index("root_roll")], 0, places=6)


class ConverterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.source = self.path / "episode.npz"
        self.model = write_episode(self.source)
        self.raw = self.source.read_bytes()

    def test_subsample_keeps_stored_commands(self):
        destination = self.path / "state.npz"
        meta = convert_recording(self.source, destination)
        export = np.load(destination, allow_pickle=False)
        self.assertEqual(meta["schema"], SCHEMA)
        self.assertEqual(meta["action_hz"], ACTION_HZ)
        self.assertEqual(meta["rows"], 3)
        self.assertEqual(meta["state_dim"], 66)
        self.assertEqual(str(export["schema"]), SCHEMA)
        self.assertEqual(tuple(export["state_names"]), layout_names(self.model))
        self.assertEqual(tuple(export["command_names"]), COMMAND_NAMES)
        np.testing.assert_array_equal(export["source_index"], [0, 5, 10])
        np.testing.assert_array_equal(export["command"][:, 0], [0, 5, 10])
        np.testing.assert_array_equal(export["time"], [0.01, 0.06, 0.11])
        names = layout_names(self.model)
        elbow = names.index("left_elbow_joint")
        np.testing.assert_allclose(export["state"][:, elbow], [0, 0.05, 0.10])
        np.testing.assert_allclose(export["grip"], [0, 0.05, 0.10])
        np.testing.assert_allclose(export["state"][:, names.index("grip")], export["grip"])
        self.assertEqual(self.source.read_bytes(), self.raw)

    def test_rejects_overwrite_interpolation_and_bad_time(self):
        with self.assertRaisesRegex(ValueError, "raw episode"):
            convert_recording(self.source, self.source)
        with self.assertRaisesRegex(ValueError, "divisor"):
            convert_recording(self.source, self.path / "out.npz", action_hz=32)
        with np.load(self.source) as episode:
            shifted = {key: np.array(episode[key]) for key in episode.files}
        shifted["time"][1] += 0.002
        bad = self.path / "bad.npz"
        np.savez_compressed(bad, **shifted)
        with self.assertRaisesRegex(ValueError, "0.010 s"):
            convert_recording(bad, self.path / "out.npz")
        self.assertEqual(RECORD_HZ // ACTION_HZ, 5)


if __name__ == "__main__":
    unittest.main()
