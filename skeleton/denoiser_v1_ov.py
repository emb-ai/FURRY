"""
Лег-денойзер v1 для live-пайплайна: окно фьюжн-скелета -> чистые ноги + conf.

Отличия от v0-обёртки (denoiser_ov.py): вход — ОКНО из 9 кадров по 153 фичи
(фьюжн xyz root-relative + presence + src-флаги + 2D + conf), выходы —
x0/x0_direct/log_sigma; conf сустава честно выводится из предсказанной
неопределённости (NLL-голова): неуверенные позы (напр. глубокий сгиб, где
модель слаба) автоматически получают меньший вес в CameraFusion шлема.

Бекенд OpenVINO (torch в процессе запрещён, PIPELINE.md 6.10). DDIM — numpy.
Тест: python -m unittest test_denoiser_v1_ov
"""
import math
from collections import deque

import numpy as np

X_DIM, T_WIN, FEAT_DIM = 18, 9, 153
COCO_LEG = [11, 12, 13, 14, 15, 16]
IMG_W, IMG_H = 1280.0, 720.0
# нормировка root-relative (медиана таз-бедро оператора из L2-калибровки)
SCALE = 0.153
CONF_K = 6.0          # conf = exp(-CONF_K * sigma_норм); 0.05 -> ~0.74


def cosine_schedule(T: int, s: float = 0.008) -> np.ndarray:
    t = np.arange(T + 1) / T
    f = np.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    return np.clip(f / f[0], 1e-5, 1.0)


class LegDenoiserV1:
    """Окно последних кадров + OpenVINO-инференс + DDIM obs-режим."""

    def __init__(self, onnx_path: str, mode: str = "obs", steps: int = 5,
                 t0_frac: float = 0.3, scale: float = SCALE, T: int = 400,
                 conf_mode: str = "honest", seed: int | None = None):
        assert mode in ("direct", "obs")
        assert conf_mode in ("honest", "fixed")
        self.mode, self.steps, self.scale = mode, max(1, steps), scale
        self.conf_mode = conf_mode
        self.abar = cosine_schedule(T)
        self.t0 = int(t0_frac * T)
        self._rng = np.random.default_rng(seed)
        try:
            try:
                from openvino import Core
            except ImportError:
                from openvino.runtime import Core
            self._req = Core().compile_model(onnx_path, "CPU").create_infer_request()
            self._backend = "openvino"
        except ImportError:
            import onnxruntime as ort
            self._sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
            self._req = None
            self._backend = "onnxruntime"
        self._win = deque(maxlen=T_WIN)

    @property
    def backend(self):
        return self._backend

    def _infer(self, x, t, win):
        if self._req is not None:
            self._req.infer({"x": x, "t": t, "win": win}, share_inputs=True)
            return (self._req.get_tensor("x0").data.copy(),
                    self._req.get_tensor("x0_direct").data.copy(),
                    self._req.get_tensor("log_sigma").data.copy())
        r = self._sess.run(["x0", "x0_direct", "log_sigma"],
                           {"x": x, "t": t, "win": win})
        return r

    # ---- фичи кадра из состояния цикла pose_yolo --------------------------
    def observe(self, raw, sources, pts2d, confidence):
        """Собрать 153 фичи кадра и положить в окно.

        raw: {coco_idx: np.array(3)} — фьюжн-суставы, color frame, метры
        sources: {coco_idx: 'window'|'line'} от fuse_joints
        pts2d/confidence: словари YOLO-кейпоинтов (conf>0.5, как в pose_yolo)
        """
        fxyz = np.full((17, 3), np.nan, np.float32)
        for i, p in raw.items():
            fxyz[i] = p
        if np.isfinite(fxyz[11, 0]) and np.isfinite(fxyz[12, 0]):
            root = (fxyz[11] + fxyz[12]) / 2
        elif np.isfinite(fxyz[..., 0]).any():
            root = np.nanmean(fxyz, axis=0)
        else:
            return                                   # кадр без фьюжна — пропускаем окно
        rel = (fxyz - root) / self.scale
        present = np.isfinite(rel[..., 0])
        src = np.zeros(17, np.float32)
        src_line = np.zeros(17, np.float32)
        for i, s in sources.items():
            src[i] = 1.0 if s == "window" else 0.0
            src_line[i] = 1.0 if s == "line" else 0.0
        k2d = np.zeros((17, 2), np.float32)
        cf = np.zeros(17, np.float32)
        for i, (u, v) in pts2d.items():
            k2d[i] = (u / IMG_W, v / IMG_H)
            cf[i] = confidence.get(i, 0.0)
        self._win.append(np.concatenate([np.nan_to_num(rel).reshape(-1), present,
                                         src, src_line, k2d.reshape(-1), cf
                                         ]).astype(np.float32))

    def _window(self):
        """(1,T,153): паддинг первым доступным кадром, если окно не заполнено."""
        frames = list(self._win)
        while len(frames) < T_WIN:
            frames.insert(0, frames[0])
        return np.stack(frames)[None]

    def sample(self) -> tuple[np.ndarray, np.ndarray]:
        """-> (x0 (18,), conf (6,)). DDIM obs: старт с прямой оценки."""
        win = self._window()
        x0 = np.zeros((1, X_DIM), np.float32)
        _, x0d, log_s = self._infer(x0, np.zeros(1, np.float32), win)
        if self.mode == "direct":
            x0 = x0d
        else:
            x = x0d.copy()
            ts = np.linspace(self.t0, 1, self.steps).astype(np.int64)
            a0 = self.abar[self.t0]
            x = (np.sqrt(a0) * x0d
                 + np.sqrt(max(1 - a0, 1e-6))
                 * self._rng.normal(size=x0d.shape).astype(np.float32))
            for i, t in enumerate(ts):
                ab = self.abar[t]
                x0, _, _ = self._infer(x, np.full(1, t, np.float32), win)
                x0 = np.clip(x0, -8, 8)
                e = (x - np.sqrt(ab) * x0) / np.sqrt(max(1 - ab, 1e-6))
                ab_n = self.abar[ts[i + 1]] if i + 1 < len(ts) else 1.0
                x = (np.sqrt(ab_n) * x0
                     + np.sqrt(max(1 - ab_n, 1e-6)) * e).astype(np.float32)
        sigma = np.exp(log_s[0].reshape(6, 3)).mean(1)     # (6,) норм. единицы
        if self.conf_mode == "fixed":
            conf = np.full(6, 0.95)      # A/B-тест: камера всегда почти на 100%
        else:
            conf = np.exp(-CONF_K * sigma).clip(0.05, 0.95)
        return x0[0], conf

    # ---- интеграция в кадр pose_yolo ---------------------------------------
    def correct(self, raw, sources, pts2d, confidence, smoothed):
        """Обновить окно кадром и вернуть (legs, confs, 'denoised_v1') или None."""
        self.observe(raw, sources, pts2d, confidence)
        if not self._win:
            return None
        if not (np.isfinite(raw.get(11, [np.nan])[0]) and
                np.isfinite(raw.get(12, [np.nan])[0])):
            return None                              # таз-якорь нужен
        x0, conf = self.sample()
        pelvis = (np.asarray(raw[11], np.float64)
                  + np.asarray(raw[12], np.float64)) / 2.0
        legs, confs = {}, {}
        for k, idx in enumerate(COCO_LEG):
            legs[idx] = pelvis + x0[3 * k:3 * k + 3].astype(np.float64) * self.scale
            confs[idx] = float(conf[k])
        return legs, confs, "denoised_v1"
