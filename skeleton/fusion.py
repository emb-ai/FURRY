"""
Общий модуль depth-фьюжна v2 — один код для live-превью (pose_yolo.py)
и офлайн-бенчмарка/GT (analyze_bag.py). Бенчмарк обязан измерять тот же
пайплайн, что будет отдавать GT, поэтому фьюжн живёт здесь, а не в копиях
по скриптам.

fuse_joints(): 2D-кейпоинты {COCO-индекс: (u, v)} + карта глубины (м)
    -> {COCO-индекс: xyz (м)}:
  * дистанция человека — крупное окно на тазе, без гейта
  * foreground-гейт: медиана окна 9x9 только из полосы ±band от таза
    (фон физически недоступен)
  * пустой сустав -> line-median вдоль кости от родителя
  * deproject по интринсикам того потока, в чьих пикселях работаем:
    live подаёт depth, выровненный на color (пиксели color, color-интринсики);
    analyze на плейбэке .db3 сначала переводит кейпоинты color->depth
    (экстринсики в записи битые, rs.align не работает) и подаёт сырой depth
    (пиксели depth, depth-интринсики). Выход получается в системе координат
    потока интринсик; для длин костей/джиттера это не важно (жёсткое
    преобразование между ними).

Индексы кейпоинтов канонические COCO-17 (0 нос ... 16 правая лодыжка);
BLAZE2COCO канонизирует BlazePose-33 для бейзлайна MediaPipe.
"""
import numpy as np
import pyrealsense2 as rs

# родитель для line-sampling (кость, из которой добираем пустой сустав)
PARENT = {7: 5, 9: 7, 8: 6, 10: 8, 13: 11, 15: 13, 14: 12, 16: 14,
          5: 6, 6: 5, 11: 12, 12: 11, 0: 5}

# BlazePose-33 -> канонический COCO-17
BLAZE2COCO = {0: 0, 11: 5, 12: 6, 13: 7, 14: 8, 15: 9, 16: 10,
              23: 11, 24: 12, 25: 13, 26: 14, 27: 15, 28: 16}


def window_median(depth_m, u, v, person_d=None, win=4, band=0.5):
    """Медиана глубины в окне (2*win+1)^2, только из полосы человека."""
    h, w = depth_m.shape
    x, y = int(round(u)), int(round(v))
    x0, x1 = max(0, x - win), min(w, x + win + 1)
    y0, y1 = max(0, y - win), min(h, y + win + 1)
    patch = depth_m[y0:y1, x0:x1]
    mask = (patch > 0) & (np.abs(patch - person_d) < band) if person_d else (patch > 0)
    if not mask.any():
        return None
    return float(np.median(patch[mask]))


def line_median(depth_m, u0, v0, u1, v1, person_d, n=9, band=0.5):
    """Медиана вдоль отрезка родитель->сустав (поверхность конечности)."""
    us, vs = np.linspace(u0, u1, n), np.linspace(v0, v1, n)
    vals = []
    for u, v in zip(us, vs):
        h, w = depth_m.shape
        x, y = int(round(u)), int(round(v))
        if 0 <= x < w and 0 <= y < h:
            d = depth_m[y, x]
            if d > 0 and (person_d is None or abs(d - person_d) < band + 0.25):
                vals.append(float(d))
    return float(np.median(vals)) if vals else None


def person_distance(depth_m, pts2d):
    """Дистанция человека: крупное окно на середине таза (без гейта)."""
    if 11 in pts2d and 12 in pts2d:
        hu = (pts2d[11][0] + pts2d[12][0]) / 2
        vu = (pts2d[11][1] + pts2d[12][1]) / 2
        return window_median(depth_m, hu, vu, None, win=8)
    if pts2d:
        u, v = next(iter(pts2d.values()))
        return window_median(depth_m, u, v, None, win=8)
    return None


def fuse_joints(depth_m, pts2d, intr, band=0.5, sources=None):
    """2D-кейпоинты {coco: (u, v)} + глубина (м) -> {coco: xyz (м)}.

    Дистанция человека оценивается по тазу, суставы берутся гейт-медианой,
    пустые добираются line-median вдоль кости от родителя.
    """
    if not pts2d:
        return {}
    person_d = person_distance(depth_m, pts2d)
    raw = {}
    for i, (u, v) in pts2d.items():
        d = window_median(depth_m, u, v, person_d, band=band)
        source = "window"
        if d is None and i in PARENT and PARENT[i] in pts2d:
            pu, pv = pts2d[PARENT[i]]
            d = line_median(depth_m, pu, pv, u, v, person_d, band=band)
            source = "line"
        if d is not None:
            if sources is not None:
                sources[i] = source
            raw[i] = np.array(
                rs.rs2_deproject_pixel_to_point(intr, [float(u), float(v)], d),
                dtype=np.float32)
    return raw


class OneEuro:
    """Векторный One Euro filter: x — np.array(3,), t — секунды (monotonic).

    Частота берётся из фактического dt между вызовами, а не из номинала
    стрима: YOLO на CPU даёт 8-15 fps, и захардкоженные 30 Гц ломают
    cutoff'ы. dt > 0.5 c (провал/рестарт) — фильтр перезапускается.
    """

    def __init__(self, mincutoff=1.2, beta=0.06, dcutoff=1.0):
        self.mc, self.beta, self.dc = mincutoff, beta, dcutoff
        self.x_prev = None
        self.t_prev = None
        self.dx_prev = np.zeros(3)

    @staticmethod
    def _alpha(cutoff, freq):
        tau = 1.0 / (2 * np.pi * cutoff)
        te = 1.0 / freq
        return 1.0 / (1.0 + tau / te)

    def __call__(self, x, t):
        dt = None if (self.x_prev is None or self.t_prev is None) else t - self.t_prev
        if dt is None or not (0 < dt <= 0.5):  # первый вызов / немонотонное время / провал
            self.x_prev, self.t_prev = x.copy(), t
            return x.copy()
        freq = 1.0 / dt
        a_d = self._alpha(self.dc, freq)
        dx = (x - self.x_prev) * freq
        self.dx_prev = a_d * dx + (1 - a_d) * self.dx_prev
        cutoff = self.mc + self.beta * abs(self.dx_prev)
        a = self._alpha(cutoff, freq)
        self.x_prev = a * x + (1 - a) * self.x_prev
        self.t_prev = t
        return self.x_prev.copy()
