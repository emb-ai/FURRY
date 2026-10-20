"""Тесты denoiser_ov.LegDenoiser: parity ONNX/OpenVINO против torch-fixture
(эталоны посчитаны в export_denoiser_onnx.py, torch в тестах не нужен),
детерминизм direct, форма/конечность DDIM-сэмпла и интеграция correct().

Запуск: python -m unittest test_denoiser_ov
"""
import os
import unittest

import numpy as np

from denoiser_ov import COCO_LEG, LegDenoiser

ONNX = os.path.join("models", "denoiser_v0.onnx")
FIXTURE = ONNX + ".fixture.npz"


@unittest.skipUnless(os.path.exists(ONNX), f"нет {ONNX} (веса не коммитятся)")
class TestLegDenoiser(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fx = np.load(FIXTURE)
        cls.deno = LegDenoiser(ONNX, mode="direct", seed=0)

    def test_parity_fixture(self):
        """OV против эталона torch: |Δ| < 5e-3 по обоим выходам (fp32 MLP)."""
        x0, x0d = self.deno._infer(self.fx["x"], self.fx["t"], self.fx["cond"])
        for got, ref, name in ((x0, self.fx["x0"], "x0"),
                               (x0d, self.fx["x0_direct"], "x0_direct")):
            err = float(np.abs(got - ref).max())
            self.assertLess(err, 5e-3, f"{name}: max|Δ|={err}")

    def test_direct_deterministic(self):
        a = self.deno.direct(self.fx["cond"])
        b = self.deno.direct(self.fx["cond"])
        np.testing.assert_array_equal(a, b)

    def test_sample_obs_prev(self):
        cond = self.fx["cond"][:1]
        deno = LegDenoiser(ONNX, mode="obs", steps=5, seed=1)
        x = deno.sample(cond, x_init=deno.direct(cond))
        self.assertEqual(x.shape, (1, 18))
        self.assertTrue(np.isfinite(x).all())
        deno_p = LegDenoiser(ONNX, mode="prev", steps=5, seed=2)
        deno_p.correct(_pts2d(), _conf(), _smoothed())       # первый кадр — direct-init
        deno_p.correct(_pts2d(), _conf(), _smoothed())       # второй — от прошлого семпла
        self.assertIsNotNone(deno_p._prev)

    def test_correct_packet_semantics(self):
        deno = LegDenoiser(ONNX, mode="direct", seed=0)
        out = deno.correct(_pts2d(), _conf(), _smoothed())
        self.assertIsNotNone(out)
        legs, confs, src = out
        self.assertEqual(sorted(legs), COCO_LEG)
        self.assertEqual(src, "denoised")
        pelvis = (_smoothed()[11] + _smoothed()[12]) / 2
        # якорь: относительная геометрия = выходу модели, pelvis добавлен честно
        rel = legs[11] - pelvis
        x0d = deno.direct(_cond())[0]
        np.testing.assert_allclose(rel, x0d[:3] * deno.scale, atol=1e-6)
        for i in COCO_LEG:
            self.assertTrue(np.isfinite(legs[i]).all())
            self.assertTrue(0.0 < confs[i] <= 0.9)
        # без бёдер — отказ, а не фантазия
        sm = _smoothed()
        sm.pop(11), sm.pop(12)
        self.assertIsNone(deno.correct(_pts2d(), _conf(), sm))


def _pts2d():
    return {i: (200.0 + 40.0 * i, 300.0 + 25.0 * i) for i in range(11, 17)}


def _conf():
    return {i: 0.8 for i in range(11, 17)}


def _smoothed():
    return {11: np.array([0.01, 0.5, 2.4]), 12: np.array([-0.01, 0.5, 2.4]),
            13: np.array([0.05, 0.9, 2.2]), 14: np.array([-0.05, 0.9, 2.2])}


def _cond():
    c = np.zeros((1, 51), np.float32)
    for i in range(11, 17):
        c[0, i] = (200.0 + 40.0 * i) / 1280.0
        c[0, 17 + i] = (300.0 + 25.0 * i) / 720.0
        c[0, 34 + i] = 0.8
    return c


if __name__ == "__main__":
    unittest.main()
