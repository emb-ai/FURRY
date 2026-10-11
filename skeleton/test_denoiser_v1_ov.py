"""Тесты denoiser_v1_ov.LegDenoiserV1: parity с torch-fixture, сборка окна,
conf из NLL-головы, интеграция correct(). Без torch в тестовом процессе.

Запуск: python -m unittest test_denoiser_v1_ov
"""
import os
import unittest

import numpy as np

from denoiser_v1_ov import FEAT_DIM, LegDenoiserV1, T_WIN

ONNX = os.path.join("models", "denoiser_v1.onnx")
FIXTURE = ONNX + ".fixture.npz"


@unittest.skipUnless(os.path.exists(ONNX), f"нет {ONNX} (веса не коммитятся)")
class TestV1(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fx = np.load(FIXTURE)
        cls.deno = LegDenoiserV1(ONNX, mode="direct", seed=0)

    def test_parity_fixture(self):
        x0, x0d, ls = self.deno._infer(self.fx["x"], self.fx["t"], self.fx["win"])
        for got, ref, name in ((x0, self.fx["x0"], "x0"),
                               (x0d, self.fx["x0_direct"], "x0_direct"),
                               (ls, self.fx["log_sigma"], "log_sigma")):
            err = float(np.abs(got - ref).max())
            self.assertLess(err, 5e-3, f"{name}: max|d|={err}")

    def test_window_features_shape(self):
        self.assertEqual(FEAT_DIM, 153)
        self.assertEqual(T_WIN, 9)
        # окно паддится первым кадром до 9
        self.deno._win.clear()
        raw = {11: np.array([0.0, 0.0, 2.0]), 12: np.array([0.1, 0.0, 2.0]),
               13: np.array([0.0, 0.4, 2.0])}
        self.deno.observe(raw, {13: "line"}, {11: (100.0, 200.0)}, {11: 0.9})
        w = self.deno._window()
        self.assertEqual(w.shape, (1, T_WIN, FEAT_DIM))
        self.assertTrue(np.isfinite(w).all())

    def test_sample_conf_bounds(self):
        x0, conf = self.deno.sample()
        self.assertEqual(x0.shape, (18,))
        self.assertTrue(np.isfinite(x0).all())
        self.assertEqual(conf.shape, (6,))
        self.assertTrue(((0.05 <= conf) & (conf <= 0.95)).all())

    def test_correct_packet(self):
        raw = {11: np.array([0.0, 0.0, 2.0]), 12: np.array([0.1, 0.0, 2.0]),
               13: np.array([0.05, 0.4, 2.1]), 14: np.array([0.05, 0.4, 2.1])}
        out = self.deno.correct(raw, {}, {11: (100.0, 200.0)}, {11: 0.9}, {})
        self.assertIsNotNone(out)
        legs, confs, src = out
        self.assertEqual(src, "denoised_v1")
        self.assertEqual(sorted(legs), [11, 12, 13, 14, 15, 16])
        pelvis = (raw[11] + raw[12]) / 2
        np.testing.assert_allclose(legs[11] - pelvis,
                                   self.deno.sample()[0][:3] * 0.153, atol=1e-5)
        for c in confs.values():
            self.assertTrue(0.05 <= c <= 0.95)
        # без фьюжн-таза — отказ
        raw2 = {13: np.array([0.0, 0.4, 2.0])}
        self.assertIsNone(self.deno.correct(raw2, {}, {}, {}, {}))


if __name__ == "__main__":
    unittest.main()
