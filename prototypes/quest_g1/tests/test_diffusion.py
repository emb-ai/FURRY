"""Low-dimensional diffusion policy: schedule and a tiny overfit."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError:
    torch = None


@unittest.skipUnless(torch is not None, "torch is an optional training dependency")
class DiffusionPolicyTests(unittest.TestCase):
    def test_schedule_decreases_and_a_constant_map_can_be_fit(self):
        from g1_sim.diffusion import POLICY_SCHEMA, Policy, cosine_alpha_bar, fit_checkpoint
        alpha = cosine_alpha_bar(50).numpy()
        self.assertEqual(len(alpha), 50)
        self.assertTrue(np.all(np.diff(alpha) <= 1e-6))
        self.assertLess(alpha[-1], 0.1)
        self.assertGreater(alpha[0], 0.9)
        observations, actions = [], []
        for value in (0.2, 0.5, 0.8):
            observations.append(np.full((8, 2, 4), value))
            actions.append(np.full((8, 4, 4), value))
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "policy.pt"
            config = fit_checkpoint(
                np.concatenate(observations), np.concatenate(actions), checkpoint,
                names=["s0", "s1", "s2", "s3"], scene="cup", steps=400, seed=0,
                action_horizon=1, width=64, diffusion_steps=16, inference_steps=8)
            self.assertEqual(config["schema"], POLICY_SCHEMA)
            self.assertEqual(config["windows"], 24)
            policy = Policy(checkpoint)
            errors = []
            for value in (0.2, 0.8):
                observation = np.full((2, 4), value)
                first = policy.sample(observation, torch.Generator().manual_seed(0))
                second = policy.sample(observation, torch.Generator().manual_seed(0))
                np.testing.assert_allclose(first, second)
                errors.append(abs(float(first[0, 0]) - value))
        self.assertLess(max(errors), 0.15)


if __name__ == "__main__":
    unittest.main()
