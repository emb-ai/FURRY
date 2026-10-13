import unittest
import numpy as np
import camera_command_noise as cn
from build_command_bank import mirror_command

JOINTS = [f'{side}_{j}_joint' for side in ('left', 'right')
          for j in ('hip_pitch', 'hip_roll', 'hip_yaw', 'knee', 'ankle_pitch', 'ankle_roll')] + \
         ['waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint']


def white_model(sigma=.1, fs=50.):
    f = np.fft.rfftfreq(64, 1 / fs)
    S = np.zeros((len(f), cn.DIMS, cn.DIMS))
    S[:] = np.eye(cn.DIMS) * sigma ** 2 / (fs / 2)  # one-sided density of white noise
    return {'fs': fs, 'freqs': f.tolist(), 'csd_real': S.tolist(), 'csd_imag': np.zeros_like(S).tolist(),
            'session_bias': [np.zeros(cn.DIMS).tolist()]}


class CameraNoiseTests(unittest.TestCase):
    def test_mirror_matches_command_bank(self):
        e = np.random.default_rng(1).standard_normal((5, cn.DIMS))
        np.testing.assert_allclose(cn.mirror(e), mirror_command(e, JOINTS))
        np.testing.assert_allclose(cn.mirror(cn.mirror(e)), e)

    def test_residual_variance_matches_spectrum(self):
        gen = cn.CameraCommandNoise(white_model(), np.random.default_rng(2))
        x = gen.residual(50 * 400)
        np.testing.assert_allclose(x.std(0), .1, rtol=.05)

    def test_knee_never_below_clean_straight_leg(self):
        m = white_model(sigma=1.)
        gen = cn.CameraCommandNoise(m, np.random.default_rng(3))
        clean = np.zeros((500, 35)); clean[:, 9] = np.radians(1); clean[:, 15] = .5
        out = gen.apply(clean)
        self.assertTrue((out[:, 9] >= np.radians(1) - 1e-12).all())
        self.assertTrue((out[:, 15] >= np.radians(3) - 1e-12).all())
        np.testing.assert_array_equal(out[:, cn.DIMS:], clean[:, cn.DIMS:])

    def test_gate_is_bounded_and_dropout_free_sessions_exist(self):
        gen = cn.CameraCommandNoise(white_model(), np.random.default_rng(4))
        gates = [gen.gate(2000) for _ in range(40)]
        self.assertTrue(all(((g >= 0) & (g <= 1)).all() for g in gates))
        self.assertTrue(any(g.min() == 1 for g in gates))
        self.assertTrue(any(g.min() == 0 for g in gates))


if __name__ == '__main__':
    unittest.main()
