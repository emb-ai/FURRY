"""
Live-трекинг AprilTag 36h11 (QR-трекеры ног) с RealSense D435i.
Детекция -> solvePnP (IPPE_SQUARE) -> One-Euro -> отрисовка / CSV.

Что считается (фрейм камеры, стандарт OpenCV/RealSense color, метры):
  tvec — центр тега: X вправо, Y вниз, Z от камеры к оператору
  rvec — ориентация тега (Родригес). Оси маркера: X вправо, Y вверх вдоль
         конечности (при монтаже «читаемо», см. TAGS_GUIDE.md), Z из плоскости
         тега к камере.
Время — аппаратные таймстампы камеры (равномерный шаг); захват идёт в колбэке
SDK-потока через очередь кадров, поэтому запись получает все 30 кадров, а
отрисовка пропускается при отставании (backlog) и не влияет на запись.

Размеры тегов — TAG_SIZE_MM. После печати ИЗМЕРЬТЕ сторону чёрного квадрата
линейкой и впишите фактическое значение: ошибка 2 мм на 120-мм теге = ~1.7%
по дистанции (~3-4 см на 2.5 м).

Вид: одно окно 50/50 — слева кадр камеры с детекцией, справа 3D: теги как
ориентированные по rvec квадраты в фрейме камеры (метры) + хвосты-треки за
последние ~5 с; зелёная стрелка от центра тега — ось Y маркера (вдоль
конечности: сразу видно yaw стопы).

Управление: q/Esc — выход · e — фиксация экспозиции вкл/выкл · s — скриншот ·
← → ↑ ↓ — вращение 3D-вида · мышь в 3D: ЛКМ-тянуть — орбита, ПКМ/СКМ-тянуть —
сдвиг пространства, колесо — зум, двойной клик — сброс вида

Запуск:
  .venv/Scripts/python tag_tracker.py                   # live с камеры
  .venv/Scripts/python tag_tracker.py --selftest        # синтетика, без камеры
  .venv/Scripts/python tag_tracker.py --record out.csv  # + запись CSV (сырые позы)
  .venv/Scripts/python tag_tracker.py --record --nogui --duration 60  # запись 60 с без GUI
  .venv/Scripts/python tag_tracker.py --refine apriltag # точный рефайн (медленно, ~5 fps)
"""
import argparse
import csv
import os
import queue
import threading
import time
from collections import deque

import cv2
import cv2.aruco as ar
import numpy as np

# ---- конфиг -------------------------------------------------------------------
TAG_SIZE_MM = {i: 120.0 for i in (0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)}
# монтаж 10.10 (L2-сессия): ID01 не печатался; 00/02 - стопы спереди,
# 03-05 - левое бедро, 06-08 - правое бедро, 09 - пятка ПРАВОЙ (задняя плоскость),
# 10 - мировой якорь на стене (~45 гр между камерами), 11 - пояс (root).
# Новые ID просто дописывать сюда (размер - по линейке!).
ID2LABEL = {0: "L-foot", 2: "R-foot",
            3: "L-thigh-out", 4: "L-thigh", 5: "L-thigh-in",
            6: "R-thigh-in", 7: "R-thigh", 8: "R-thigh-out",
            9: "R-heel-back", 10: "world-wall", 11: "center"}
MIN_SIDE_PX = 36.0   # тег меньше — дальний/ложный декод (36 px = 4.5 px/модуль при 120 мм, ~2.8 м)

# сценарий записи (для --script): латиница — у Hershey-шрифтов OpenCV нет кириллицы
SCRIPT_BLOCKS = [
    ("statika 10s", 15),
    ("shagi na meste", 150),
    ("mahi vpered / v storony", 150),
    ("prisedaniya, vypady", 150),
    ("razvoroty stop, perenos vesa", 150),
    ("stoyka, VR-zhesty", 150),
    ("statika 10s", 15),
]


def draw_script_panel(disp, eff_now):
    """Панель сценария на кадре: текущий блок с обратным отсчётом. Возвращает idx."""
    ends = np.cumsum([d for _, d in SCRIPT_BLOCKS])
    idx = int(np.searchsorted(ends, eff_now))
    idx = min(idx, len(SCRIPT_BLOCKS) - 1)
    x0, y0, lh = 12, 46, 24
    w = 380
    h = lh * (len(SCRIPT_BLOCKS) + 1) + 10
    roi = disp[y0:y0 + h, x0:x0 + w]
    if roi.size:
        ov = roi.copy()
        cv2.rectangle(ov, (0, 0), (w, h), (24, 24, 24), -1)
        roi[:] = cv2.addWeighted(ov, 0.65, roi, 0.35, 0)
        cv2.rectangle(disp, (x0, y0), (x0 + w, y0 + h), (90, 90, 90), 1)
        total = ends[-1]
        cv2.putText(disp, f"SCRIPT {eff_now/60:04.1f}/{total/60:04.1f} min   [n]=next",
                    (x0 + 8, y0 + 18), 6, 0.55, (220, 220, 220), 1)
        y = y0 + lh + 10
        for k, (name, _) in enumerate(SCRIPT_BLOCKS):
            if k < idx:
                mark, col, txt = "+", (110, 110, 110), name
            elif k == idx:
                rem = max(0.0, ends[k] - eff_now)
                mark, col = ">", (80, 255, 80)
                txt = f"{name}  {int(rem // 60):02d}:{int(rem % 60):02d}"
            else:
                mark, col, txt = " ", (150, 150, 150), name
            cv2.putText(disp, f"{mark} {txt}", (x0 + 8, y), 6, 0.55, col, 1)
            y += lh
    return idx
TEST_TAG_IDS = {20, 21, 22, 23}               # настольные 60 мм: детектим, но не фильтруем

DICT = ar.getPredefinedDictionary(ar.DICT_APRILTAG_36H11)
P = ar.DetectorParameters()
P.cornerRefinementMethod = ar.CORNER_REFINE_SUBPIX   # 7 мс/кадр; APRILTAG-рефайн в ~5 раз
P.minMarkerPerimeterRate = 0.02                      # точнее, но 179 мс -> только офлайн (--refine apriltag)
DET = ar.ArucoDetector(DICT, P)

MODULES = DICT.markerSize + 2                 # 8 модулей с рамкой (для selftest)


# ---- геометрия ------------------------------------------------------------------
def obj_points(size_m: float) -> np.ndarray:
    """Углы маркера в его системе координат, порядок как у detectMarkers: TL, TR, BR, BL."""
    s = size_m / 2.0
    return np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], dtype=np.float32)


def estimate_pose(corner_px: np.ndarray, tag_id: int, K, dist):
    """solvePnP IPPE_SQUARE для плоского маркера. Возвращает (rvec, tvec) или None."""
    size_m = TAG_SIZE_MM.get(tag_id, 120.0) / 1000.0
    res = cv2.solvePnP(obj_points(size_m), corner_px, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not res[0]:
        return None
    return res[1], res[2]


# ---- фильтр ----------------------------------------------------------------------
class OneEuro:
    """One Euro Filter на 3-вектор: гасит джиттер, почти не лагая на движении."""

    def __init__(self, min_cutoff=1.7, beta=0.3, d_cutoff=1.0):
        self.mc, self.beta, self.dc = min_cutoff, beta, d_cutoff
        self.x_prev = None
        self.dx_prev = np.zeros(3)
        self.t_prev = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, t, x):
        if self.x_prev is None:
            self.x_prev, self.t_prev = x.copy(), t
            return x.copy()
        dt = max(t - self.t_prev, 1e-3)
        self.t_prev = t
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.dc, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev
        self.dx_prev = dx_hat
        a = self._alpha(self.mc + self.beta * np.abs(dx_hat), dt)
        x_hat = a * x + (1.0 - a) * self.x_prev
        self.x_prev = x_hat
        return x_hat


# ---- 3D-вид (ортопроекция с поворотом, стиль pose_realsense.py) ------------------
TRAIL_LEN = 150   # кадров в хвосте трека (~5 с при 30 fps)
PALETTE = [(80, 200, 60), (60, 120, 255), (255, 170, 70), (70, 170, 255),
           (200, 120, 255), (255, 220, 120), (120, 255, 230), (255, 120, 190),
           (180, 200, 60), (60, 200, 200), (200, 120, 120), (120, 180, 255)]


class Tags3D:
    """Правая половина окна: теги в 3D (фрейм камеры, метры) + треки."""

    def __init__(self, w=1280, h=720, scale_px=280.0):
        self.w, self.h, self.s = w, h, scale_px
        self.yaw, self.pitch = np.radians(-35), np.radians(18)
        self.zoom = 1.0                  # колесо мыши
        self.pan_x = self.pan_y = 0.0    # сдвиг вида, px
        self._drag = None

    def project(self, pts):
        """(N,3) метры, фрейм камеры (X вправо, Y ВНИЗ, Z от камеры) -> (N,2) px."""
        cy, sy = np.cos(self.yaw), np.sin(self.yaw)
        cp, sp = np.cos(self.pitch), np.sin(self.pitch)
        x1 = pts[:, 0] * cy + pts[:, 2] * sy
        z1 = -pts[:, 0] * sy + pts[:, 2] * cy
        y1 = pts[:, 1] * cp - z1 * sp
        s = self.s * self.zoom
        return np.stack([self.w / 2 + self.pan_x + x1 * s,
                         self.h * 0.55 + self.pan_y + y1 * s], axis=1)

    # ---- мышь: ЛКМ — орбита, ПКМ/СКМ — сдвиг, колесо — зум, 2xклик — сброс ----
    def bind(self, win):
        cv2.setMouseCallback(win, self._on_mouse)

    def reset_view(self):
        self.yaw, self.pitch = np.radians(-35), np.radians(18)
        self.zoom, self.pan_x, self.pan_y = 1.0, 0.0, 0.0

    def _on_mouse(self, event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            self._drag = ("orbit", x, y)
        elif event in (cv2.EVENT_RBUTTONDOWN, cv2.EVENT_MBUTTONDOWN):
            self._drag = ("pan", x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self._drag:
            mode, x0, y0 = self._drag
            dx, dy = x - x0, y - y0
            if mode == "orbit":
                self.yaw += dx * 0.005
                self.pitch += dy * 0.005
            else:
                self.pan_x += dx
                self.pan_y += dy
            self._drag = (mode, x, y)
        elif event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP, cv2.EVENT_MBUTTONUP):
            self._drag = None
        elif event == cv2.EVENT_LBUTTONDBLCLK:   # в OpenCV 5 без двойных суффиксов CLICK
            self.reset_view()
        elif event == cv2.EVENT_MOUSEWHEEL:
            self.zoom = float(np.clip(self.zoom * (1.15 if flags > 0 else 1 / 1.15),
                                      0.05, 40.0))

    def draw(self, poses, trails, info=""):
        """poses: id -> (tvec фильтр., rvec); trails: id -> deque[tvec]."""
        canvas = np.full((self.h, self.w, 3), (28, 28, 30), np.uint8)

        # пол-сетка — чуть ниже самого низкого из известных положений тегов
        ys = [tv[1] for tv, _ in poses.values()] + \
             [p[1] for tr in trails.values() for p in tr]
        y_g = (max(ys) + 0.10) if ys else 1.5
        for g in np.arange(-2.0, 2.01, 0.5):
            for a, b in (((g, y_g, 0.0), (g, y_g, 4.0)), ((-2.0, y_g, g), (2.0, y_g, g))):
                pa, pb = self.project(np.array([a, b]))
                cv2.line(canvas, tuple(pa.astype(int)), tuple(pb.astype(int)),
                         (55, 55, 58), 1)

        # камера в начале координат + направление взгляда (+Z)
        cam_p = self.project(np.zeros((1, 3)))[0].astype(int)
        z_p = self.project(np.array([[0.0, 0.0, 0.4]]))[0].astype(int)
        cv2.circle(canvas, tuple(cam_p), 6, (0, 0, 0), -1)
        cv2.circle(canvas, tuple(cam_p), 5, (230, 230, 230), -1)
        cv2.line(canvas, tuple(cam_p), tuple(z_p), (90, 90, 90), 1, cv2.LINE_AA)
        cv2.putText(canvas, "cam", (cam_p[0] - 18, cam_p[1] + 4), 6, 0.5,
                    (160, 160, 160), 1)

        # хвосты-треки
        for i, tr in trails.items():
            if len(tr) < 2:
                continue
            pts = self.project(np.stack(list(tr)))
            dim = tuple(int(c * 0.55) for c in PALETTE[i % len(PALETTE)])
            for k in range(len(pts) - 1):
                cv2.line(canvas, tuple(pts[k].astype(int)), tuple(pts[k + 1].astype(int)),
                         dim, 2, cv2.LINE_AA)

        # теги: ориентированный по rvec квадрат + ось Y маркера (вдоль конечности)
        for i, (tv, rv) in poses.items():
            size_m = TAG_SIZE_MM.get(i, 120.0) / 1000.0
            R, _ = cv2.Rodrigues(rv)
            corners = (R @ obj_points(size_m).T).T + tv
            quad = self.project(corners).astype(np.int32)
            color = PALETTE[i % len(PALETTE)]
            cv2.fillPoly(canvas, [quad], tuple(int(c * 0.35) for c in color))
            cv2.polylines(canvas, [quad], True, color, 2, cv2.LINE_AA)
            c0 = self.project(tv.reshape(1, 3))[0]
            tip = self.project((tv + R @ np.array([0.0, size_m * 0.8, 0.0])).reshape(1, 3))[0]
            cv2.line(canvas, tuple(c0.astype(int)), tuple(tip.astype(int)), (60, 255, 60), 2)
            cv2.putText(canvas, f"ID{i}", (int(c0[0]) - 16, int(c0[1]) - 12),
                        6, 0.6, color, 2)

        cv2.putText(canvas, "3D (m): L-drag orbit, R-drag pan, wheel zoom, dblclick reset",
                    (16, 26), 6, 0.55, (200, 200, 200), 1)
        for n, line in enumerate(info.splitlines()):
            cv2.putText(canvas, line, (16, self.h - 24 - n * 20), 6, 0.55, (150, 200, 150), 1)
        return canvas

    def rotate(self, k):
        """k — сырой код waitKey (стрелки: Windows 2424832.., GTK 81..84, unix 0..3)."""
        d = {2424832: "l", 2555904: "r", 2490368: "u", 2621440: "d",
             81: "l", 83: "r", 82: "u", 84: "d",
             2: "l", 3: "r", 0: "u", 1: "d"}.get(k)
        if d == "l":
            self.yaw -= np.radians(8)
        elif d == "r":
            self.yaw += np.radians(8)
        elif d == "u":
            self.pitch += np.radians(6)
        elif d == "d":
            self.pitch -= np.radians(6)


# ---- selftest: синтетический кадр без камеры --------------------------------------
def run_selftest():
    K = np.array([[920.0, 0, 640.0], [0, 920.0, 360.0], [0, 0, 1.0]])
    dist = np.zeros(5)
    # тег ID2 120 мм в точке (-0.25, -0.10, 2.0) м, наклонён на 20° вокруг Y камеры.
    # R_true: диагональ (1,-1,-1) — «читаемый» тег лицом к камере в конвенции
    # OpenCV (Y маркера вверх = -Y камеры), затем наклон.
    theta = np.deg2rad(20.0)
    R_tilt, _ = cv2.Rodrigues(np.array([0.0, theta, 0.0]))
    Rt = R_tilt @ np.diag([1.0, -1.0, -1.0])
    rvec_true, _ = cv2.Rodrigues(Rt)
    tvec_true = np.array([[-0.25], [-0.10], [2.0]])

    proj, _ = cv2.projectPoints(obj_points(0.120), rvec_true, tvec_true, K, dist)
    quad = proj.reshape(4, 2).astype(np.float32)

    # рендер с суперсэмплингом 4x + лёгкий блюр — края как у реальной оптики,
    # иначе субпиксельный рефайнмент цепляется за растеризацию (проверено: без
    # этого позиция уезжает на ~15 мм артефактно)
    ss = 4
    n = 960
    bits = ar.generateImageMarker(DICT, 2, MODULES)
    big = cv2.resize(bits, (n * ss, n * ss), interpolation=cv2.INTER_NEAREST)
    src = np.float32([[0, 0], [n * ss, 0], [n * ss, n * ss], [0, n * ss]])
    H = cv2.getPerspectiveTransform(src, quad * ss)
    frame4 = cv2.warpPerspective(big, H, (1280 * ss, 720 * ss),
                                 flags=cv2.INTER_NEAREST, borderValue=255)
    frame = cv2.GaussianBlur(cv2.resize(frame4, (1280, 720), interpolation=cv2.INTER_AREA),
                             (3, 3), 0.6)
    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    corners, ids, _ = DET.detectMarkers(frame)
    if ids is None or 2 not in ids.ravel():
        print("SELFTEST FAIL: тег не детектится")
        return 1
    k = int(np.flatnonzero(ids.ravel() == 2)[0])
    rvec, tvec = estimate_pose(corners[k], 2, K, dist)

    err_t = float(np.linalg.norm(tvec - tvec_true)) * 1000.0   # мм
    err_a = float(np.linalg.norm(cv2.Rodrigues(rvec)[0] - Rt, ord="fro"))
    print(f"SELFTEST: ID2 детект; ошибка позиции {err_t:.2f} мм, ориентации {err_a:.4f} (фробениус)")

    # 3D-канва: тег+трек должны отрисоваться (считаем пиксели в цветах тега)
    view = Tags3D()
    trail = deque([tvec.reshape(3)] * 10, maxlen=TRAIL_LEN)
    canvas = view.draw({2: (tvec.reshape(3), rvec.reshape(3))}, {2: trail}, "selftest")
    mask = np.zeros(canvas.shape[:2], bool)
    for c in PALETTE:
        mask |= (canvas == np.array(c)).all(axis=2)
    px = int(mask.sum())
    print(f"SELFTEST 3D: {px} px в цветах тега — " + ("OK" if px > 150 else "FAIL"))
    # порог 15 мм: SUBPIX на синтетике даёт ~11 мм (растеризация краёв завышает;
    # на реальной оптике краёв меньше, а систематику Z убирает калибровка размера).
    # --refine apriltag даёт ~2 мм, но 180 мс/кадр.
    ok = err_t < 15.0 and px > 150
    print("SELFTEST OK" if ok else "SELFTEST FAIL")
    return 0 if ok else 1


# ---- live -------------------------------------------------------------------------
def run_live(record_path: str | None, nogui: bool = False, duration: float = 0.0,
             video_path: str | None = None, script: bool = False):
    import pyrealsense2 as rs

    pipe = rs.pipeline()
    cfg = rs.config()
    # 1280x720: детект с человеком в кадре 29-34 мс на 1080p (= весь бюджет
    # кадра 33 мс -> очередь переполняется, окно голодает); на 720p втрое
    # дешевле. Условие: оператор не дальше ~2.5 м (тег 120 мм >= 5.5 px/модуль)
    cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

    # Захват в колбэке SDK-потока: копии кадров копятся в очередь, основной
    # поток обрабатывает их подряд. Обработка (~10 мс) успевает за кадром
    # (33 мс) -> запись получает ВСЕ 30 кадров; отрисовка пропускается, если
    # накопился backlog, и никогда не тормозит запись. Переполнение очереди
    # (=обработка не успевает) считается в dropped и печатается — не молчим.
    frame_q: queue.Queue = queue.Queue(maxsize=90)   # ~3 с буфера
    dropped = {"n": 0}

    def on_frame(f):
        # в колбэк приходит generic frame -> обязательно развернуть в frameset,
        # иначе get_color_frame() кидает исключение ПОВЕРХ SDK и молчит
        if not f.is_frameset():
            return
        fs = f.as_frameset()
        color = fs.get_color_frame()
        if color is None:
            return
        try:
            frame_q.put_nowait((np.asanyarray(color.get_data()).copy(),
                                fs.get_timestamp() * 1e-3))
        except queue.Full:
            dropped["n"] += 1

    profile = pipe.start(cfg, on_frame)

    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    K = np.array([[intr.fx, 0, intr.ppx],
                  [0, intr.fy, intr.ppy],
                  [0, 0, 1]], dtype=np.float64)
    dist = np.array(intr.coeffs, dtype=np.float64)
    print(f"color intrinsics: {intr.width}x{intr.height}, fx={intr.fx:.1f} fy={intr.fy:.1f} "
          f"pp=({intr.ppx:.1f},{intr.ppy:.1f}) model={intr.model}")

    color_sensor = None
    for s in profile.get_device().query_sensors():
        try:
            if "RGB" in s.get_info(rs.camera_info.name):
                color_sensor = s
        except Exception:
            pass
    fixed = {"on": False, "val": None}

    def toggle_exposure():
        if color_sensor is None:
            return
        if not fixed["on"]:
            fixed["val"] = color_sensor.get_option(rs.option.exposure)
            color_sensor.set_option(rs.option.enable_auto_exposure, 0)
            color_sensor.set_option(rs.option.exposure, fixed["val"])
            fixed["on"] = True
        else:
            color_sensor.set_option(rs.option.enable_auto_exposure, 1)
            fixed["on"] = False

    csv_f = csv_w = None
    if record_path:
        os.makedirs(os.path.dirname(record_path) or ".", exist_ok=True)
        csv_f = open(record_path, "w", newline="")
        csv_w = csv.writer(csv_f)
        csv_w.writerow(["t_s", "id", "tx", "ty", "tz", "rx", "ry", "rz", "seen"])
    video_w = None
    video_q = None
    video_thr = None
    video_drops = {"n": 0}
    if video_path:
        os.makedirs(os.path.dirname(video_path) or ".", exist_ok=True)
        video_w = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"MJPG"),
                                  30.0, (1280, 720))
        video_w.set(cv2.VIDEOWRITER_PROP_QUALITY, 70)   # ~2x меньше/быстрее
        video_q = queue.Queue(maxsize=450)              # буфер ~15 с

        def _video_loop():
            while True:
                item = video_q.get()
                if item is None:
                    break
                video_w.write(item)

        video_thr = threading.Thread(target=_video_loop, daemon=True)
        video_thr.start()
        print("video:", video_path)

    os.makedirs("tags_snapshots", exist_ok=True)
    filters: dict[int, OneEuro] = {}
    trails: dict[int, deque] = {}
    view = Tags3D()
    if not nogui:
        cv2.namedWindow("tags | 3D", cv2.WINDOW_NORMAL)
        view.bind("tags | 3D")
    t_cam0 = None      # первый аппаратный таймстамп, с (база времени записи)
    wall0 = time.perf_counter()
    n_done = 0
    fps_ema = 0.0
    silent = {"s": 0.0}   # сколько секунд камера молчит (для авто-реконнекта)
    dbg = {"render": 0, "det_ms": 0.0}   # телеметрия цикла (печать каждые 150 кадров)
    script_state = {"off": 0.0}          # сдвиг сценария (клавиша n — следующий блок)

    try:
        while True:
            try:
                color, t_cam = frame_q.get(timeout=2.0)
                silent["s"] = 0.0
            except queue.Empty:
                silent["s"] += 2.0
                if silent["s"] >= 6.0 and not nogui:
                    # D435i капризна к USB: после срыва молча — переподключаемся
                    print(f"камера молчит {silent['s']:.0f} с — реконнект...", flush=True)
                    try:
                        pipe.stop()
                    except Exception:
                        pass
                    time.sleep(1.0)
                    pipe.start(cfg, on_frame)   # интринсики не меняются
                    silent["s"] = 0.0
                elif silent["s"] % 10 == 0:
                    print(f"нет кадров {silent['s']:.0f} с (кабель/USB?)", flush=True)
                continue
            if t_cam0 is None:
                t_cam0 = t_cam
            now = t_cam - t_cam0          # время камеры: равномерный шаг 33 мс

            t_d0 = time.perf_counter()
            corners, ids, _ = DET.detectMarkers(color)
            dbg["det_ms"] = (time.perf_counter() - t_d0) * 1000
            poses: dict[int, tuple] = {}  # (фильтр., rvec) — для отрисовки
            raw: dict[int, tuple] = {}    # (сырой tvec, rvec) — для CSV
            if ids is not None:
                for c, i in zip(corners, ids.ravel()):
                    i = int(i)
                    if i not in TAG_SIZE_MM:
                        continue
                    quad = c.reshape(4, 2)
                    if np.linalg.norm(quad[1] - quad[0]) < MIN_SIDE_PX:
                        continue
                    est = estimate_pose(c, i, K, dist)
                    if est is None:
                        continue
                    rvec, tvec = est
                    raw[i] = (tvec.reshape(3).copy(), rvec.reshape(3).copy())
                    f = filters.setdefault(i, OneEuro())
                    tv = f(now, tvec.reshape(3).copy())
                    poses[i] = (tv, rvec.reshape(3).copy())

            if csv_w:
                for i in sorted(TAG_SIZE_MM):
                    if i in raw:
                        tv_r, rv = raw[i]
                        csv_w.writerow([f"{now:.4f}", i, *(f"{v:.4f}" for v in tv_r),
                                        *(f"{v:.4f}" for v in rv), 1])
                    else:
                        csv_w.writerow([f"{now:.4f}", i, "nan", "nan", "nan",
                                        "nan", "nan", "nan", 0])
            n_done += 1
            if n_done % 150 == 0:
                print(f"dbg: n={n_done} renders={dbg['render']} qsize={frame_q.qsize()} "
                      f"det={dbg['det_ms']:.0f}ms seen={len(raw)}", flush=True)
            if video_q is not None:    # энкод в отдельном потоке — кадр не теряем
                try:
                    video_q.put_nowait(color)
                except queue.Full:
                    video_drops["n"] += 1

            if nogui:
                if duration and now >= duration:
                    break
                if n_done % 300 == 0:
                    rate = n_done / (time.perf_counter() - wall0)
                    print(f"t={now:.0f}s | {rate:.1f} fps | seen "
                          f"{len(raw)}/{len(TAG_SIZE_MM)} | drop {dropped['n']}", flush=True)
                continue

            # отрисовка — каждый 2-й кадр (~15 fps); пропускаем показ ТОЛЬКО
            # если очередь почти полная (запись важнее). Мгновенный backlog
            # (1-5 кадров) — не причина морить окно: это нормальный буфер
            if frame_q.qsize() > 60 or n_done % 2:
                k = cv2.waitKey(1)          # держим окно отзывчивым без отрисовки
                if (k & 0xFF) in (ord("q"), 27):
                    break
                continue
            dbg["render"] += 1
            for i, (tv, _) in poses.items():
                trails.setdefault(i, deque(maxlen=TRAIL_LEN)).append(tv.copy())

            disp = color.copy()
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(disp, corners, ids)
                for i in poses:
                    k_id = np.flatnonzero(ids.ravel() == i)
                    if len(k_id):
                        cv2.drawFrameAxes(disp, K, dist,
                                          raw[i][1].reshape(3, 1), raw[i][0].reshape(3, 1),
                                          0.06, 2)
                        x, y = corners[k_id[0]].reshape(4, 2)[0]
                        cv2.putText(disp,
                                    f"ID{i} {ID2LABEL.get(i, '')} "
                                    f"z {raw[i][0][2]*100:.0f}|{poses[i][0][2]*100:.0f} cm",
                                    (int(x) - 40, int(y) - 12),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 200, 60), 2)

            dt = time.perf_counter() - wall0
            rate = (n_done / dt) if dt > 0 else 0.0
            fps_ema = 0.9 * fps_ema + 0.1 * rate
            cv2.putText(disp, f"{fps_ema:.1f} fps | {len(poses)}/{len(TAG_SIZE_MM)} tags"
                        f" | exp {'FIXED' if fixed['on'] else 'auto'}",
                        (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (235, 235, 235), 2)
            if script:
                draw_script_panel(disp, now + script_state["off"])

            canvas = view.draw(
                poses, trails,
                info=f"{fps_ema:.1f} fps | seen {len(poses)}/{len(TAG_SIZE_MM)} | "
                     f"exp {'FIXED' if fixed['on'] else 'auto'}")
            combined = np.hstack([disp, canvas])
            cv2.imshow("tags | 3D", combined)

            k = cv2.waitKey(1)
            key = k & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("e"):
                toggle_exposure()
            if key == ord("n") and script:
                # перескок на следующий блок сценария
                ends = np.cumsum([d for _, d in SCRIPT_BLOCKS])
                eff = now + script_state["off"]
                k = int(np.searchsorted(ends, eff))
                if k < len(SCRIPT_BLOCKS):
                    script_state["off"] += ends[k] - eff
            if key == ord("s"):
                p = f"tags_snapshots/frame_{n_done:06d}.png"
                cv2.imwrite(p, combined)
                print("saved", p)
            view.rotate(k)
    finally:
        pipe.stop()
        cv2.destroyAllWindows()
        wall = time.perf_counter() - wall0
        print(f"кадров обработано: {n_done} за {wall:.1f} с "
              f"({n_done / wall if wall > 0 else 0:.1f} fps), потеряно в очереди: {dropped['n']}")
        if csv_f:
            csv_f.close()
            print("csv:", record_path)
        if video_w is not None:
            video_q.put(None)
            video_thr.join(timeout=5.0)
            video_w.release()
            print(f"video закрыт (кадров потеряно для видео: {video_drops['n']})")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--selftest", action="store_true", help="синтетический тест без камеры")
    ap.add_argument("--record", nargs="?", const="AUTO", metavar="CSV",
                    help="писать СЫРЫЕ позы в CSV (имя можно не указывать)")
    ap.add_argument("--nogui", action="store_true",
                    help="без окон — максимум fps для записи (Ctrl+C стоп)")
    ap.add_argument("--duration", type=float, default=0.0, metavar="SEC",
                    help="длительность записи в сек (с --nogui; 0 = до Ctrl+C)")
    ap.add_argument("--refine", choices=["subpix", "apriltag"], default="subpix",
                    help="рефайн углов: subpix ~7 мс (live) | apriltag точнее в ~5 раз, "
                         "но ~180 мс/кадр — для точных статики/офлайн")
    ap.add_argument("--video", nargs="?", const="AUTO", metavar="AVI",
                    help="писать цветовой поток в MJPG-avi (~35 МБ/мин; "
                         "bag не вариант: 8 ГБ/мин по замеру)")
    ap.add_argument("--script", action="store_true",
                    help="панель сценария в окне: блоки с обратным отсчётом, n — следующий")
    args = ap.parse_args()
    if args.selftest:
        raise SystemExit(run_selftest())
    if args.refine == "apriltag":
        global DET
        P2 = ar.DetectorParameters()
        P2.cornerRefinementMethod = ar.CORNER_REFINE_APRILTAG
        P2.minMarkerPerimeterRate = 0.02
        DET = ar.ArucoDetector(DICT, P2)
    if args.record == "AUTO":
        args.record = time.strftime("tags_%Y%m%d_%H%M%S.csv")
    if getattr(args, "video", None) == "AUTO":
        args.video = time.strftime("tags_%Y%m%d_%H%M%S.avi")
    run_live(args.record, nogui=args.nogui, duration=args.duration,
             video_path=args.video, script=args.script)


if __name__ == "__main__":
    main()
