"""
Анализ качества depth-фьюжна на записи .db3.

Начиная с v2 считает тем же кодом, что live-превью (фьюжн в fusion.py):
те же медианы/гейт/line-sampling/SDK-фильтры — таблица «до/после» измеряет
настоящий пайплайн, а не его офлайн-копию. Отличия от live только ожидаемые:

  * 2D-детектор выбирается аргументом: yolo (как в live v2) | mediapipe
    (бейзлайн v1, BlazePose-33 канонизируется в COCO-17)
  * фазы и dt — по реальным таймстампам бага (frame.get_timestamp), а не по
    номеру кадра: дропнутые кадры больше не сдвигают границы STATIC/T-POSE
  * --no-temporal: A/B пиксельного временного сглаживания (temporal-EMA
    может мазать глубину быстрых конечностей; для GT сглаживать лучше в 3D)

Метрики:
  1. Вариативность длин костей (статика/движение): медиана / std / CV%
  2. Джиттер в статике: скорость суставов (см/с) по dt из таймстампов
  3. Дропауты: % кадров без каждого ключевого сустава
  4. Длины костей по T-POSE (калибровка скелета)

Запуск:
  .venv/Scripts/python analyze_bag.py data/benchmark3.db3
  .venv/Scripts/python analyze_bag.py data/benchmark3.db3 mediapipe
  .venv/Scripts/python analyze_bag.py data/benchmark3.db3 yolo m --no-temporal
"""
import argparse
import time

import cv2
import numpy as np
import pyrealsense2 as rs

from fusion import BLAZE2COCO, fuse_joints

# (имя, сустав A, сустав B) — канонические COCO-17 индексы
BONES = [
    ("плечи (L-R)", 5, 6), ("торс", 5, 11),
    ("плечо Л", 5, 7), ("предплечье Л", 7, 9),
    ("плечо П", 6, 8), ("предплечье П", 8, 10),
    ("бёдра (L-R)", 11, 12),
    ("бедро Л", 11, 13), ("голень Л", 13, 15),
    ("бедро П", 12, 14), ("голень П", 14, 16),
]
KEY_JOINTS = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

p = argparse.ArgumentParser(description="Бенчмарк depth-фьюжна по записи .db3")
p.add_argument("bag", help="путь к .db3")
p.add_argument("model", nargs="?", default="yolo", choices=["yolo", "mediapipe"],
               help="2D-детектор: yolo (v2, live) | mediapipe (бейзлайн v1)")
p.add_argument("size", nargs="?", default="s", choices=["s", "m", "l"],
               help="размер yolo11{size}-pose (для model=yolo)")
p.add_argument("--no-temporal", action="store_true",
               help="выключить temporal+hole_filling (A/B пиксельного сглаживания)")
args = p.parse_args()

# ---- 2D-детектор --------------------------------------------------------------
if args.model == "yolo":
    from ultralytics import YOLO
    model = YOLO(f"models/yolo11{args.size}-pose.pt")
    print(f"YOLO11{args.size}-pose загружен", flush=True)

    def detect(color, ts_ms):
        res = model.predict(color, imgsz=640, conf=0.3, classes=0,
                            device="cpu", verbose=False)[0]
        pts = {}
        if res.keypoints is not None and len(res.keypoints):
            kxy = res.keypoints.xy.cpu().numpy()
            kc = (res.keypoints.conf.cpu().numpy()
                  if res.keypoints.conf is not None else np.ones(kxy.shape[:-1]))
            xy, conf = (kxy[0], kc[0]) if kxy.ndim == 3 else (kxy, kc)
            for i in range(min(17, len(xy))):
                if conf[i] > 0.5:
                    pts[i] = (float(xy[i][0]), float(xy[i][1]))
        return pts

    def detect_close():
        pass
else:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    landmarker = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path="models/pose_landmarker_full.task"),
        running_mode=vision.RunningMode.VIDEO, num_poses=1))

    def detect(color, ts_ms):
        res = landmarker.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB,
                     data=cv2.cvtColor(color, cv2.COLOR_BGR2RGB)), ts_ms)
        pts = {}
        if res.pose_landmarks:
            h, w = color.shape[:2]
            for blaze_i, coco_i in BLAZE2COCO.items():
                l = res.pose_landmarks[0][blaze_i]
                if l.visibility > 0.5:
                    pts[coco_i] = (l.x * w, l.y * h)
        return pts

    def detect_close():
        landmarker.close()

# ---- RealSense плейбэк --------------------------------------------------------
cfg = rs.config()
cfg.enable_device_from_file(args.bag, repeat_playback=False)
cfg.enable_stream(rs.stream.color, rs.format.any, 30)
cfg.enable_stream(rs.stream.depth, rs.format.any, 30)

pipe = rs.pipeline()
profile = pipe.start(cfg)
profile.get_device().as_playback().set_real_time(False)  # читаем так быстро, как успеваем
depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
color_intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
depth_intr = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()

# Экстринсики в .db3 не сериализуются (на плейбэке мусор ~1e10, align из-за этого
# отдаёт нулевую глубину) — берём заводскую калибровку живой камеры (константа устройства).
C2D_ROT = [0.9999944567680359, 0.0033040582202374935, -0.00040445956983603537,
           -0.003306071273982525, 0.9999816417694092, -0.005082169082015753,
           0.00038766037323512137, 0.005083478055894375, 0.999987006187439]
C2D_TRA = [-0.014619573950767517, -0.00022538429766427726, -0.00030640410841442645]
D2C_ROT = [0.9999944567680359, -0.003306071273982525, 0.00038766037323512137,
           0.0033040582202374935, 0.9999816417694092, 0.005083478055894375,
           -0.00040445956983603537, -0.005082169082015753, 0.999987006187439]
D2C_TRA = [0.014620114117860794, 0.00017548960749991238, 0.0003132132987957448]
c2d, d2c = rs.extrinsics(), rs.extrinsics()
c2d.rotation, c2d.translation = C2D_ROT, C2D_TRA
d2c.rotation, d2c.translation = D2C_ROT, D2C_TRA


def to_depth_pixels(pts2d, d_buf):
    """color-пиксели -> depth-пиксели (align на плейбэке не работает)."""
    out = {}
    for i, (u, v) in pts2d.items():
        try:
            dp = rs.rs2_project_color_pixel_to_depth_pixel(
                d_buf, depth_scale, 0.1, 5.0, depth_intr, color_intr, c2d, d2c,
                [float(u), float(v)])
        except Exception:
            continue
        out[i] = (float(dp[0]), float(dp[1]))
    return out


# SDK depth-фильтры — та же цепочка, что в live v2 (pose_yolo.py). Разница:
# live фильтрует depth, уже выровненный на color, здесь — сырой depth; при
# 1280x720 и дистанции 2-3 м это практически то же окно (параллакс ~15 px).
spatial = rs.spatial_filter()
spatial.set_option(rs.option.filter_magnitude, 2)
spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
spatial.set_option(rs.option.filter_smooth_delta, 20)
chain = [spatial]
if not args.no_temporal:
    temporal = rs.temporal_filter()
    temporal.set_option(rs.option.holes_fill, 3)
    chain += [temporal, rs.hole_filling_filter()]

tag = (f"yolo11{args.size}-pose" if args.model == "yolo" else "mediapipe-full")
print(f"Баг: {args.bag}\nДетектор: {tag} | temporal-фильтр: "
      f"{'выкл' if args.no_temporal else 'вкл'}\n", flush=True)

# ---- проход по записи ----------------------------------------------------------
frames = []          # (t_sec, {coco_idx: xyz})
n = 0
ts0, ts_prev, last_color_ts = None, 0.0, None
t_wall = time.monotonic()
try:
    while True:
        try:
            fs = pipe.wait_for_frames(timeout_ms=5000)
        except RuntimeError:
            break  # конец файла
        color_f, depth_f = fs.get_color_frame(), fs.get_depth_frame()
        if not color_f or not depth_f:
            continue
        n += 1
        # Плейбэк .db3 подставляет ДУБЛИКАТ прошлого color-кадра к осиротевшему
        # depth-кадру (в записи color терялся): тот же 2D-скелет x другой depth
        # с dt~0 => выбросы скорости. Дедуп по color-таймстампу.
        ts_raw = float(color_f.get_timestamp())
        if ts_raw == last_color_ts:
            continue
        last_color_ts = ts_raw
        # реальный таймстамп бага (мс): фазы и dt — по нему; монотонность
        # поддерживаем принудительно (VIDEO-режиму MediaPipe нужна строгая)
        ts_ms = max(ts_raw, ts_prev + 1.0)
        ts_prev = ts_ms
        if ts0 is None:
            ts0 = ts_ms

        d_buf = depth_f.get_data()  # сырой буфер: проекция color->depth по нему
        for flt in chain:
            depth_f = flt.process(depth_f)
        depth_m = np.asanyarray(depth_f.get_data()).astype(np.float32) * depth_scale
        color = np.asanyarray(color_f.get_data())

        pts2d = detect(color, int(ts_ms))
        raw = fuse_joints(depth_m, to_depth_pixels(pts2d, d_buf), depth_intr)
        frames.append(((ts_ms - ts0) / 1000.0, raw))
        if n % 150 == 0:
            print(f"  обработано {n} кадров ({time.monotonic() - t_wall:.0f} c)", flush=True)
finally:
    pipe.stop()
    detect_close()

print(f"\nВсего кадров: {len(frames)} (из {n} прочитанных)")
if len(frames) >= 2:
    span = frames[-1][0] - frames[0][0]
    print(f"Длительность по таймстампам: {span:.1f} с -> фактический "
          f"{(len(frames) - 1) / span:.1f} fps при записи 30")

# фазы — по времени записи (как в оверлее record_bag.py), не по номеру кадра
def phase(t):
    if t < 5:
        return "ready"
    if t < 20:
        return "static"
    if t < 25:
        return "tpose"
    return "free"

# ---------- 1. длины костей ----------
def bone_len(fr, a, b):
    if a in fr and b in fr:
        return float(np.linalg.norm(fr[a] - fr[b])) * 100  # см
    return None

print("\n=== Длины костей (см) ===")
print(f"{'кость':<14}{'медиана':>9}{'std(static)':>12}{'CV%':>7}{'std(free)':>10}")
calib = {}
for name, a, b in BONES:
    ls_static = [x for x in (bone_len(frames[k][1], a, b)
                             for k in range(len(frames)) if phase(frames[k][0]) == "static") if x]
    ls_free = [x for x in (bone_len(frames[k][1], a, b)
                           for k in range(len(frames)) if phase(frames[k][0]) == "free") if x]
    if not ls_static:
        print(f"{name:<14}  нет данных")
        continue
    med, std = np.median(ls_static), np.std(ls_static)
    calib[name] = med
    cv = 100 * std / med if med else 0
    std_free = np.std(ls_free) if ls_free else float("nan")
    print(f"{name:<14}{med:>9.1f}{std:>12.2f}{cv:>7.1f}{std_free:>10.2f}")

# ---------- 2. джиттер в статике (dt — реальный, из таймстампов) ----------
speeds, pelvis_xy = [], []
for (ta, f0), (tb, f1) in zip(frames, frames[1:]):
    dt = tb - ta
    if phase(ta) != "static" or dt <= 0 or len(f0) < 4:
        continue
    for j in set(f0) & set(f1):
        speeds.append(np.linalg.norm(f1[j] - f0[j]) / dt * 100)  # см/с
    if 11 in f0 and 12 in f0 and 11 in f1 and 12 in f1:
        pelvis_xy.append(((f0[11] + f0[12]) / 2)[:2])
if speeds:
    print("\n=== Джиттер (фаза STATIC) ===")
    print(f"средняя скорость суставов: {np.mean(speeds):.1f} см/с (медиана {np.median(speeds):.1f})")
    print(f"95-перцентиль скорости:    {np.percentile(speeds, 95):.1f} см/с")
    if pelvis_xy:
        print(f"разброс таза (std, см):   X={np.std([q[0] for q in pelvis_xy]) * 100:.2f} "
              f"Y={np.std([q[1] for q in pelvis_xy]) * 100:.2f}")

# ---------- 3. дропауты ----------
print("\n=== Дропауты ключевых суставов (весь ролик) ===")
for j in KEY_JOINTS:
    miss = sum(1 for _, f in frames if j not in f)
    print(f"  сустав {j:>2}: отсутствует в {100 * miss / max(1, len(frames)):5.1f}% кадров")
jpf = np.array([len(f) for _, f in frames])
print(f"суставов на кадр: медиана {np.median(jpf):.0f}, "
      f"min {jpf.min()}, max {jpf.max()} (из 17)")

# ---------- 4. T-POSE ----------
print("\n=== T-POSE длины (калибровка) ===")
for name, a, b in BONES:
    ls = [x for x in (bone_len(frames[k][1], a, b)
                      for k in range(len(frames)) if phase(frames[k][0]) == "tpose") if x]
    if ls:
        print(f"  {name:<14} {np.median(ls):6.1f} см")
