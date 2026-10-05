"""Integration checks for policy stability, scene contacts and reset semantics."""
import unittest

import mujoco
import numpy as np

from g1_sim.__main__ import arm_command
from g1_sim.controller import Controller
from g1_sim.scene import make_model


class SimulationTests(unittest.TestCase):
    def test_full_robot_stays_balanced_with_hands_and_arm_commands(self):
        model, _ = make_model("lab")
        data = mujoco.MjData(model)
        controller = Controller(model)
        controller.initialize(data)
        initial = data.qpos.copy()
        min_z, max_tilt = 1., 0.
        for i in range(60000):
            controller.step(data, arm_command(data.time), grip=float((i//10000) % 2))
            min_z = min(min_z, data.qpos[2])
            max_tilt = max(max_tilt, np.arccos(np.clip(data.xmat[model.body("pelvis").id,8], -1, 1)))
        self.assertTrue(np.isfinite(data.qpos).all())
        self.assertGreater(min_z, .70)
        self.assertLess(max_tilt, np.radians(15))
        self.assertLess(np.linalg.norm(data.qpos[:2]), .30)
        self.assertEqual(sum(w.number for w in data.warning), 0)
        self.assertAlmostEqual(data.body("cup").xpos[2], .75, delta=.02)
        self.assertAlmostEqual(data.body("t_box").xpos[2], .10, delta=.02)
        controller.initialize(data)
        np.testing.assert_array_equal(data.qpos, initial)
        self.assertEqual(data.time, 0)
        self.assertEqual(controller.step_index, 0)
        self.assertTrue(all(np.count_nonzero(h) == 0 for h in controller.history))

    def test_t_box_responds_to_force_and_cup_falls_onto_table(self):
        model, _ = make_model("lab")
        data = mujoco.MjData(model)
        controller = Controller(model)
        controller.initialize(data)
        cup_q = model.joint("cup_free").qposadr[0]
        data.qpos[cup_q+2] += .2
        box_id = model.body("t_box").id
        start_x = data.body("t_box").xpos[0]
        for i in range(3000):
            data.xfrc_applied[box_id, 0] = 25 if i < 500 else 0
            controller.step(data)
        self.assertGreater(data.body("t_box").xpos[0]-start_x, .1)
        self.assertAlmostEqual(data.body("cup").xpos[2], .75, delta=.02)
        cup_id = model.body("cup").id
        contact_bodies = {(model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]) for c in data.contact}
        table_id = model.body("table").id
        self.assertTrue((cup_id, table_id) in contact_bodies or (table_id, cup_id) in contact_bodies)


if __name__ == "__main__":
    unittest.main()
