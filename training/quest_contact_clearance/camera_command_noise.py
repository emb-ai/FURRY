"""Camera-leg command error model for actor-input augmentation.

The error is measured in policy-command space: e = recorded command with
camera legs - the same frame re-solved by native GMR from raw Meta legs
(see fit()). Meta legs are generative, not ground truth, so e is the
camera/Meta disagreement. Treating all of it as camera error is the
conservative choice for robustness training.

Model, all in the first 21 command values (root vx vy z roll pitch yaw-rate,
12 leg joints, 3 waist joints; arms are unaffected by camera legs):
  - session bias: one measured session mean, randomly mirrored and scaled;
  - stationary Gaussian residual with the measured 21x21 cross-spectrum
    (camera depth wander is mostly 0.05-4 Hz, nothing above 4 Hz);
  - dropouts: camera legs off -> clean command, linear fade as in fusion;
  - knee flex is kept >= clamp(clean flex, 0, 3 deg) like CameraFusion Hinge().
The camera/Meta lag measured 0 +- 100 ms on a good network, so no delay term.
"""
import json
from pathlib import Path

import numpy as np

DIMS = 21
NAMES = ['vx', 'vy', 'z', 'roll', 'pitch', 'yaw_rate',
         'l_hip_pitch', 'l_hip_roll', 'l_hip_yaw', 'l_knee', 'l_ankle_pitch', 'l_ankle_roll',
         'r_hip_pitch', 'r_hip_roll', 'r_hip_yaw', 'r_knee', 'r_ankle_pitch', 'r_ankle_roll',
         'waist_yaw', 'waist_roll', 'waist_pitch']
KNEES = (9, 15)
_SWAP = list(range(6)) + list(range(12, 18)) + list(range(6, 12)) + list(range(18, 21))
_SIGN = np.array([1, -1, 1, -1, 1, -1] + [1, -1, -1, 1, 1, -1] * 2 + [-1, -1, 1], float)


def mirror(e):
    """Left/right reflection, same convention as build_command_bank.mirror_command."""
    return e[..., _SWAP] * _SIGN


def fit(pairs, fs=50., nperseg=512, min_on=0.1):
    """pairs: list of (camera[T,>=21], meta[T,>=21]) on a uniform fs grid, policy-held values.
    Frames where no leg differs by more than min_on degrees count as camera-off."""
    from scipy.signal import csd
    biases, spectra, weights, freqs, runs = [], [], [], None, []
    for cam, meta in pairs:
        e = (cam - meta)[:, :DIMS]
        on = np.degrees(np.abs(e[:, 6:18])).max(1) > min_on
        biases.append(e[on].mean(0))
        r = e - biases[-1]
        edges = np.flatnonzero(np.diff(on.astype(int))) + 1
        for seg in np.split(np.arange(len(on)), edges):
            runs.append((bool(on[seg[0]]), len(seg) / fs))
            if not on[seg[0]] or len(seg) < nperseg:
                continue
            S = np.empty((nperseg // 2 + 1, DIMS, DIMS), complex)
            for i in range(DIMS):
                for j in range(i, DIMS):
                    freqs, S[:, i, j] = csd(r[seg, i], r[seg, j], fs=fs, nperseg=nperseg)
                    S[:, j, i] = np.conj(S[:, i, j])
            spectra.append(S); weights.append(len(seg))
    w = np.array(weights, float) / sum(weights)
    S = np.tensordot(w, np.array(spectra), 1)
    return {'schema': 'camera-command-noise-v1', 'fs': fs, 'names': NAMES,
            'freqs': freqs.tolist(), 'csd_real': S.real.tolist(), 'csd_imag': S.imag.tolist(),
            'session_bias': [b.tolist() for b in biases],
            'runs': runs, 'segments_seconds': float(sum(weights) / fs)}


def save(model, path):
    Path(path).write_text(json.dumps(model))


def load(path):
    return json.loads(Path(path).read_text())


class CameraCommandNoise:
    """Generates one camera session of command errors and applies it to clean commands."""

    def __init__(self, model, rng, bias_scale=(.5, 1.5), noise_scale=1.,
                 on_mean_s=(2., 40.), off_mean_s=(.15, 2.), dropout_free=.3, fade_s=.16):
        self.m, self.rng = model, rng
        self.bias_scale, self.noise_scale = bias_scale, noise_scale
        self.on_mean_s, self.off_mean_s, self.dropout_free, self.fade_s = on_mean_s, off_mean_s, dropout_free, fade_s

    def residual(self, n):
        """Stationary Gaussian process with the fitted cross-spectral density."""
        fs = self.m['fs']; f0 = np.array(self.m['freqs'])
        S0 = np.array(self.m['csd_real']) + 1j * np.array(self.m['csd_imag'])
        N = 1 << int(np.ceil(np.log2(max(n, 2 * len(f0)))))
        f = np.fft.rfftfreq(N, 1 / fs)
        X = np.zeros((len(f), DIMS), complex)
        for k in range(1, len(f)):
            j = min(np.searchsorted(f0, f[k]), len(f0) - 1)
            lo = max(j - 1, 0); u = 0 if j == lo else (f[k] - f0[lo]) / (f0[j] - f0[lo])
            S = (1 - u) * S0[lo] + u * S0[j]
            val, vec = np.linalg.eigh(S)
            L = vec * np.sqrt(np.clip(val, 0, None) * N * fs / 2)
            z = (self.rng.standard_normal(DIMS) + 1j * self.rng.standard_normal(DIMS)) / np.sqrt(2)
            X[k] = L @ z
        return np.fft.irfft(X, N, axis=0)[:n]

    def gate(self, n):
        """Camera-on weight in [0,1] with fades, two-state renewal process."""
        fs = self.m['fs']
        if self.rng.random() < self.dropout_free:
            return np.ones(n)
        on_mean = np.exp(self.rng.uniform(*np.log(self.on_mean_s)))
        off_mean = np.exp(self.rng.uniform(*np.log(self.off_mean_s)))
        g, state, i = np.zeros(n), True, 0
        while i < n:
            d = max(1, int(self.rng.exponential(on_mean if state else off_mean) * fs))
            g[i:i + d] = float(state); i += d; state = not state
        k = max(1, int(self.fade_s * fs))
        return np.convolve(np.r_[np.full(k - 1, g[0]), g], np.ones(k) / k, 'valid')

    def session(self, n):
        bias = np.array(self.m['session_bias'][self.rng.integers(len(self.m['session_bias']))])
        if self.rng.random() < .5:
            bias = mirror(bias)
        e = bias * self.rng.uniform(*self.bias_scale) + self.noise_scale * self.residual(n)
        if self.rng.random() < .5:
            e = mirror(e)
        return e, self.gate(n)

    def apply(self, clean, e=None, gate=None):
        """clean: [T,>=21] commands on the model grid. Returns corrupted copy."""
        if e is None:
            e, gate = self.session(len(clean))
        out = clean.copy()
        out[:, :DIMS] += gate[:, None] * e
        for k in KNEES:
            out[:, k] = np.maximum(out[:, k], np.clip(clean[:, k], 0, np.radians(3)))
        return out
