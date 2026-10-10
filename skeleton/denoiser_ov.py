"""
Лег-денойзер для live-пайплайна: OpenVINO-обёртка над ONNX-экспортом
диффузионной модели (обучение — sirius_twist/train_denoiser_v0.py,
экспорт — export_denoiser_onnx.py). Тот же математический контракт:
cond (B,51) = u/1280, v/720, conf по 17 COCO-кейпоинтам;
выход (B,18) = 6 суставов ног [L/R hip, knee, ankle] root-relative,
в единицах scale (метры = выход x scale).

Режимы (обратная цепь диффузии стартует не из гауссианы):
  direct — один прогон прямой головы cond -> x0 (детерминированный);
  obs    — x_t0 из прямой оценки + шум, короткая цепь DDIM t0->0;
  prev   — как obs, но x_t0 из семпла прошлого кадра (авторегрессия).

Бекенд: OpenVINO (как у YOLO в pose_yolo — torch в этом процессе запрещён,
PIPELINE.md §6.10); фолбек onnxruntime, если openvino недоступен.
DDIM-цикл — numpy поверх одного MLP-вызова на шаг: микросекунды на CPU.

Тест: python -m unittest test_denoiser_ov  (parity против fixture torch).
"""
import math

import numpy as np

X_DIM, COND_DIM = 18, 51
COCO_LEG = [11, 12, 13, 14, 15, 16]   # порядок = [L-hip, R-hip, L-knee, R-knee, L-ankle, R-ankle]
IMG_W, IMG_H = 1280.0, 720.0


def cosine_schedule(T: int, s: float = 0.008) -> np.ndarray:
    """abar (T+1,), в точности как в обучении (train_denoiser_v0.py)."""
    t = np.arange(T + 1) / T
    f = np.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    return np.clip(f / f[0], 1e-5, 1.0)


class LegDenoiser:
    """Загрузка ONNX + DDIM-сэмплер + интеграция с кадром pose_yolo."""

    def __init__(self, onnx_path: str, mode: str = "obs", steps: int = 5,
                 t0_frac: float = 0.3, scale: float = 0.125, T: int = 400,
                 seed: int | None = None):
        assert mode in ("direct", "obs", "prev")
        self.mode, self.steps, self.scale = mode, max(1, steps), scale
        self.abar = cosine_schedule(T)
        self.t0 = int(t0_frac * T)
        self._prev = None
        self._rng = np.random.default_rng(seed)
        try:
            try:
                from openvino import Core          # openvino >= 2024
            except ImportError:
                from openvino.runtime import Core  # старые сборки
            self._req = Core().compile_model(onnx_path, "CPU").create_infer_request()
            self._backend = "openvino"
        except ImportError:
            import onnxruntime as ort
            self._sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
            self._req = None
            self._backend = "onnxruntime"

    @property
    def backend(self):
        return self._backend

    def _infer(self, x, t, cond):
        """-> (x0 (B,18), x0_direct (B,18))."""
        if self._req is not None:
            self._req.infer({"x": x, "t": t, "cond": cond}, share_inputs=True)
            return (self._req.get_tensor("x0").data.copy(),
                    self._req.get_tensor("x0_direct").data.copy())
        x0, x0d = self._sess.run(["x0", "x0_direct"],
                                 {"x": x, "t": t, "cond": cond})
        return x0, x0d

    def direct(self, cond: np.ndarray) -> np.ndarray:
        x0, _ = self._infer(np.zeros((len(cond), X_DIM), np.float32),
                            np.zeros(len(cond), np.float32), cond)
        return x0

    def sample(self, cond: np.ndarray, x_init: np.ndarray | None = None) -> np.ndarray:
        """DDIM (eta=0), как train_denoiser_v0.ddim_sample. x_init != None —
        старт не из гауссианы, а из оценки x0, «зашумленной» до уровня t0."""
        B = len(cond)
        t0 = self.t0 if x_init is not None else len(self.abar) - 1
        ts = np.linspace(t0, 1, self.steps).astype(np.int64)
        if x_init is None:
            x = self._rng.normal(size=(B, X_DIM)).astype(np.float32)
        else:
            a = self.abar[t0]
            x = (np.sqrt(a) * x_init
                 + np.sqrt(max(1.0 - a, 1e-6))
                 * self._rng.normal(size=(B, X_DIM))).astype(np.float32)
        x0 = x
        for i, t in enumerate(ts):
            ab_t = self.abar[t]
            x0, _ = self._infer(x, np.full(B, t, np.float32), cond)
            x0 = np.clip(x0, -8.0, 8.0)
            e_hat = (x - np.sqrt(ab_t) * x0) / np.sqrt(max(1.0 - ab_t, 1e-6))
            ab_next = self.abar[ts[i + 1]] if i + 1 < len(ts) else 1.0
            x = (np.sqrt(ab_next) * x0
                 + np.sqrt(max(1.0 - ab_next, 1e-6)) * e_hat).astype(np.float32)
        return x0

    # ---- интеграция в кадр pose_yolo ------------------------------------------
    def correct(self, pts2d: dict, confidence: dict, smoothed: dict):
        """Скорректировать ноги текущего кадра.

        pts2d/confidence — словари pose_yolo (только кейпоинты conf>0.5),
        smoothed — One-Euro-суставы {idx: np.array(3)} в camera frame.
        Возвращает (legs {11..16: np.array(3)}, conf {idx: float}, "denoised")
        или None, если якоря таза нет. Меняет внутреннее состояние в режиме
        prev (хранит прошлый семпл).
        """
        if 11 not in smoothed or 12 not in smoothed:
            return None
        cond = np.zeros((1, COND_DIM), np.float32)
        kp_conf = np.zeros(17, np.float32)
        for i, (u, v) in pts2d.items():
            cond[0, i] = u / IMG_W
            cond[0, 17 + i] = v / IMG_H
            kp_conf[i] = confidence.get(i, 0.0)
            cond[0, 34 + i] = kp_conf[i]
        hip_c = min(kp_conf[11], kp_conf[12])
        if hip_c <= 0.0:                       # бёдра не видны — cond пуст, не врём
            return None
        x0d = self.direct(cond)
        if self.mode == "direct":
            legs = x0d
        elif self.mode == "obs":
            legs = self.sample(cond, x_init=x0d)
        else:                                  # prev: цепь от прошлого семпла
            legs = self.sample(cond, x_init=self._prev if self._prev is not None else x0d)
            self._prev = legs
        pelvis = (np.asarray(smoothed[11], np.float64)
                  + np.asarray(smoothed[12], np.float64)) / 2.0
        out, confs = {}, {}
        for k, idx in enumerate(COCO_LEG):
            out[idx] = pelvis + legs[0, 3 * k:3 * k + 3].astype(np.float64) * self.scale
            c = kp_conf[idx] if kp_conf[idx] > 0 else hip_c
            confs[idx] = float(min(0.9, c))
        return out, confs, "denoised"
