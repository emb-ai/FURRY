"""
Realtime позирование через RealSense D435i + MediaPipe Pose Landmarker.

Пайплайн (ядро будущего GT-рекордера для ретаргетинга):
  1) D435i: цвет 1280x720@30 + глубина, выровненная по цвету (rs.align)
  2) MediaPipe Pose Landmarker (VIDEO режим) -> 33 кейпоинта, 2D normalized
  3) Фьюжн: для каждого кейпоинта медиана глубины в окне 7x7 px
     -> rs2_deproject_pixel_to_point по интринсикам цвета -> метрический 3D (метры, фрейм камеры)
  4) Отрисовка: одно окно 50/50 — слева скелет поверх цвета, справа 3D-скелет
     (ортопроекция в метрах, вращается стрелками)

Управление:
  q / Esc  — выход
  ← → ↑ ↓  — вращение 3D-вида (yaw / pitch)
  w        — показывать нативные world landmarks MediaPipe (ghost) поверх фьюжн — видно, как шумит их Z

Запуск:  .venv/Scripts/python pose_realsense.py [lite|full|heavy]
"""
import sys
import time
from collections import defaultdict

import cv2
import numpy as np
import pyrealsense2 as rs
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_DIR = "models"
W2D = "D435i color | 3D pose"

# ---- Скелет BlazePose-33: связи и палитра по частям тела -------------------
CONNECTIONS = [
    # торс
    (11, 12), (11, 23), (12, 24), (23, 24),
    # левая рука
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    # правая рука
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    # лицо
    (0, 9), (9, 10), (10, 0), (7, 2), (8, 5),
    # левая нога
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    # правая нога
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
]
TORSO, LARM, RARM, LLEG, RLEG, FACE = range(6)
PART_COLOR = {  # BGR
    TORSO: (235, 235, 235), LARM: (255, 170, 70), RARM: (70, 170, 255),
    LLEG: (80, 200, 60), RLEG: (60, 120, 255), FACE: (200, 200, 120),
}
IDX2PART = {i: FACE for i in range(11)}
IDX2PART.update({i: LARM for i in (11, 13, 15, 17, 19, 21)})
IDX2PART.update({i: RARM for i in (12, 14, 16, 18, 20, 22)})
IDX2PART.update({i: LLEG for i in (23, 25, 27, 29, 31)})
IDX2PART.update({i: RLEG for i in (24, 26, 28, 30, 32)})
KEY_JOINTS = {0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28}  # крупные суставы — GT-набор


def part_of(i, j):
    if IDX2PART[i] == IDX2PART[j]:
        return IDX2PART[i]
    return TORSO


# ---- Depth-фьюжн ------------------------------------------------------------
def fuse_point(depth_data: np.ndarray, u: float, v: float, intr, scale: float,
               win: int = 3, max_m: float = 5.0):
    """Медиана глубины в окне (2*win+1)^2 вокруг пикселя + deproject -> метры (фрейм камеры)."""
    h, w = depth_data.shape
    x, y = int(round(u)), int(round(v))
    x0, x1 = max(0, x - win), min(w, x + win + 1)
    y0, y1 = max(0, y - win), min(h, y + win + 1)
    patch = depth_data[y0:y1, x0:x1].astype(np.float32)
    patch = patch[(patch > 0)]
    if patch.size == 0:  # дыра в глубине — пробуем окно шире
        x0, x1 = max(0, x - 7), min(w, x + 8)
        y0, y1 = max(0, y - 7), min(h, y + 8)
        patch = depth_data[y0:y1, x0:x1].astype(np.float32)
        patch = patch[(patch > 0)]
        if patch.size == 0:
            return None
    d = float(np.median(patch)) * scale
    if not (0.1 < d < max_m):
        return None
    pt = rs.rs2_deproject_pixel_to_point(intr, [u, v], d)
    return np.array(pt, dtype=np.float32)


# ---- 3D-рендер (ортопроекция с поворотом) ----------------------------------
class View3D:
    def __init__(self, w=1280, h=720, scale_px=280.0):
        self.w, self.h, self.s = w, h, scale_px
        self.yaw, self.pitch = np.radians(-25), np.radians(12)

    def project(self, pts):
        """pts: (N,3) метры, фрейм камеры (X вправо, Y ВНИЗ, Z от камеры).
        Возвращает (N,2) пиксели. Y уже направлена вниз -> на экран без инверсии."""
        cy, sy = np.cos(self.yaw), np.sin(self.yaw)
        cp, sp = np.cos(self.pitch), np.sin(self.pitch)
        x1 = pts[:, 0] * cy + pts[:, 2] * sy
        z1 = -pts[:, 0] * sy + pts[:, 2] * cy
        y1 = pts[:, 1] * cp - z1 * sp
        return np.stack([self.w / 2 + x1 * self.s, self.h * 0.55 + y1 * self.s], axis=1)

    def draw_skeleton(self, pts, show_ghost=False, ghost=None, info=""):
        """pts: dict idx->xyz (метры). Возвращает canvas BGR."""
        canvas = np.zeros((self.h, self.w, 3), np.uint8)
        canvas[:] = (28, 28, 30)
        if not pts:
            cv2.putText(canvas, "no pose", (20, 40), 6, 0.9, (120, 120, 120), 2)
            return canvas

        pelvis = (pts[23] + pts[24]) / 2 if 23 in pts and 24 in pts else next(iter(pts.values()))
        rel = {i: p - pelvis for i, p in pts.items()}
        idx = sorted(rel)
        arr = np.stack([rel[i] for i in idx])
        pos_of = {i: k for k, i in enumerate(idx)}
        proj = self.project(arr)

        # пол-сетка на уровне стоп (в pelvis-относительных координатах)
        feet = [rel[i][1] for i in (27, 28, 29, 30, 31, 32) if i in rel]
        y_g = max(feet) + 0.05 if feet else 0.95
        grid = []
        for g in np.arange(-1.5, 1.51, 0.25):
            grid += [((g, y_g, -1.5), (g, y_g, 1.5)), ((-1.5, y_g, g), (1.5, y_g, g))]
        for a, b in grid:
            pa, pb = self.project(np.array([a, b]))
            cv2.line(canvas, tuple(pa.astype(int)), tuple(pb.astype(int)), (55, 55, 58), 1)

        # ghost: нативные world landmarks MediaPipe (для сравнения Z)
        if show_ghost and ghost is not None:
            gp = {i: g for i, g in ghost.items() if i in rel}
            if gp:
                gi = sorted(gp)
                garr = np.stack([gp[i] for i in gi])
                gpos_of = {i: k for k, i in enumerate(gi)}
                gproj = self.project(garr)
                for i, j in CONNECTIONS:
                    if i in gpos_of and j in gpos_of:
                        cv2.line(canvas, tuple(gproj[gpos_of[i]].astype(int)),
                                 tuple(gproj[gpos_of[j]].astype(int)), (90, 90, 90), 1, cv2.LINE_AA)

        # связи
        for i, j in CONNECTIONS:
            if i in pos_of and j in pos_of:
                cv2.line(canvas, tuple(proj[pos_of[i]].astype(int)), tuple(proj[pos_of[j]].astype(int)),
                         PART_COLOR[part_of(i, j)], 4, cv2.LINE_AA)
        # суставы
        for i, k in pos_of.items():
            p = tuple(proj[k].astype(int))
            r, c = (7, (255, 255, 255)) if i in KEY_JOINTS else (4, PART_COLOR[IDX2PART[i]])
            cv2.circle(canvas, p, r + 2, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(canvas, p, r, c, -1, cv2.LINE_AA)

        cv2.putText(canvas, "depth-fused 3D (m), pelvis=0", (16, 26), 6, 0.55, (200, 200, 200), 1)
        for n, line in enumerate(info.splitlines()):
            cv2.putText(canvas, line, (16, self.h - 60 + n * 22), 6, 0.55, (150, 200, 150), 1)
        return canvas

    def rotate(self, key):
        if key == 81 or key == 2: self.yaw -= np.radians(8)   # ← (unix: 2)
        elif key == 83 or key == 3: self.yaw += np.radians(8) # →
        elif key == 82 or key == 0: self.pitch += np.radians(6)  # ↑
        elif key == 84 or key == 1: self.pitch -= np.radians(6)  # ↓


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "full"
    model_path = f"{MODEL_DIR}/pose_landmarker_{model_name}.task"
    print(f"Модель: {model_path}")

    # --- RealSense (конфиг; само подключение — в open_camera, с авто-реконнектом) ---
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

    # --- MediaPipe ---
    opts = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO, num_poses=1)
    landmarker = vision.PoseLandmarker.create_from_options(opts)

    view = View3D()
    smoothed = {}          # EMA-сглаженные 3D-точки
    EMA = 0.5
    show_ghost = False
    t0, frames = time.monotonic(), 0
    fps = 0.0
    ts_ms = 0

    cv2.namedWindow(W2D, cv2.WINDOW_NORMAL)

    def open_camera():
        """(Пере)подключение к камере с ожиданием, пока она появится в системе."""
        while True:
            try:
                p = rs.pipeline()
                prof = p.start(cfg)
                return p, prof
            except Exception as e:
                print(f"[!] Камера недоступна ({e}). Жду переподключения... (q — выход)", flush=True)
                for _ in range(10):  # ~2 c опроса, окна остаются отзывчивыми
                    k = cv2.waitKey(200) & 0xFF
                    if k in (ord('q'), 27) or cv2.getWindowProperty(W2D, cv2.WND_PROP_VISIBLE) < 1:
                        raise SystemExit(0)

    pipeline = None
    try:
        while True:
            if pipeline is None:
                pipeline, profile = open_camera()
                align = rs.align(rs.stream.color)
                depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
                color_intr = (profile.get_stream(rs.stream.color)
                              .as_video_stream_profile().get_intrinsics())
                print(f"USB: {profile.get_device().get_info(rs.camera_info.usb_type_descriptor)}, "
                      f"depth scale {depth_scale*1000:.1f} мм/ед.")
            try:
                fs = pipeline.wait_for_frames(timeout_ms=5000)
            except RuntimeError as e:
                print(f"[!] Камера отвалилась: {e}", flush=True)
                try:
                    pipeline.stop()
                except RuntimeError:
                    pass
                pipeline = None
                continue
            fs = align.process(fs)
            color_frame = fs.get_color_frame()
            depth_frame = fs.get_depth_frame()
            if not color_frame or not depth_frame:
                continue
            color = np.asanyarray(color_frame.get_data())          # BGR uint8
            depth = np.asanyarray(depth_frame.get_data())          # uint16

            ts_ms = max(ts_ms + 1, int(time.monotonic() * 1000))
            res = landmarker.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB,
                         data=cv2.cvtColor(color, cv2.COLOR_BGR2RGB)), ts_ms)

            pts3d, ghost = {}, {}
            if res.pose_landmarks:
                lms = res.pose_landmarks[0]
                H, W = color.shape[:2]
                px = {i: (l.x * W, l.y * H) for i, l in enumerate(lms) if l.visibility > 0.5}
                for i, j in CONNECTIONS:
                    if i in px and j in px:
                        cv2.line(color, tuple(map(int, px[i])), tuple(map(int, px[j])),
                                 PART_COLOR[part_of(i, j)], 2, cv2.LINE_AA)
                for i, (u, v) in px.items():
                    r, c = (6, (255, 255, 255)) if i in KEY_JOINTS else (3, PART_COLOR[IDX2PART[i]])
                    cv2.circle(color, (int(u), int(v)), r, c, -1, cv2.LINE_AA)

                # depth-фьюжн -> метрический 3D
                for i, l in enumerate(lms):
                    if l.visibility < 0.5:
                        continue
                    p = fuse_point(depth, l.x * W, l.y * H, color_intr, depth_scale)
                    if p is not None:
                        pts3d[i] = p
                # нативные world landmarks для сравнения (origin в тазу, метры)
                if res.pose_world_landmarks:
                    wl = res.pose_world_landmarks[0]
                    ghost = {i: np.array([w.x, w.y, w.z], np.float32)
                             for i, w in enumerate(wl) if wl[i].visibility > 0.5}

            # EMA-сглаживание (только для отрисовки)
            for i, p in pts3d.items():
                smoothed[i] = p if i not in smoothed else EMA * p + (1 - EMA) * smoothed[i]
            smoothed = {i: p for i, p in smoothed.items() if i in pts3d}

            frames += 1
            dt = time.monotonic() - t0
            if dt > 1:
                fps, t0, frames = frames / dt, time.monotonic(), 0

            pelvis = smoothed.get(23) if 23 in smoothed else None
            info = (f"fps {fps:4.1f} | joints 3D {len(smoothed)}/33 | "
                    f"pelvis {'-' if pelvis is None else f'{pelvis[2]:.2f} m'}")
            cv2.putText(color, info, (12, 30), 6, 0.7, (60, 255, 60), 2)
            cv2.putText(color, f"model={model_name}  q-quit  w-ghost", (12, 58), 6, 0.55, (200, 200, 200), 1)

            # одно окно 50/50: цвет со скелетом | 3D-скелет (2560x720)
            combined = np.hstack([color, view.draw_skeleton(smoothed, show_ghost, ghost, info)])
            cv2.imshow(W2D, combined)
            if cv2.getWindowProperty(W2D, cv2.WND_PROP_VISIBLE) < 1:
                print("Окно закрыто — выходим.", flush=True)
                break
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key == ord('w'):
                show_ghost = not show_ghost
            else:
                view.rotate(key)
    finally:
        if pipeline is not None:
            try:
                pipeline.stop()
            except RuntimeError:
                pass
        landmarker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
