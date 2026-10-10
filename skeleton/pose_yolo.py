"""
v2 пайплайна позинга: YOLO11-pose + умный depth-фьюжн + One-Euro.

Апгрейды над pose_realsense.py (по итогам бенчмарка benchmark3):
  1) Кейпоинты: YOLO11s-pose (COCO-17) вместо MediaPipe BlazePose-33
  2) Depth-фильтры SDK: spatial (edge-preserving) + temporal + hole_filling
  3) Foreground-гейт: глубина сустава берётся только из полосы ±0.5 м от
     дистанции человека (таза) — фон физически недоступен (лечит «руки 65-87 см»)
  4) Line-sampling: пустой сустав добирается медианой вдоль кости от родителя
  5) One-Euro на сустав (адаптивный jitter/lag) вместо EMA

Фьюжн (гейт-медиана, line-sampling) и One-Euro — общие с офлайн-бенчмарком
analyze_bag.py, код в fusion.py: бенчмарк v2 измеряет ровно этот пайплайн.
One-Euro берёт фактический dt (CPU даёт 8-15 fps, не 30) и не сбрасывает
состояние при дропауте сустава.

Управление: q/Esc — выход, крестик — выход, стрелки — вращение 3D,
c — зафиксировать камеру по AprilTag и позе контроллера (нужен --ws).
Запуск: .venv/Scripts/python pose_yolo.py [s|m|l] [full|min|off]
  второй аргумент — режим depth-фильтров (min по умолчанию, см. main()).
  --backend torch|openvino|auto (auto: OV-экспорт, если есть) и --imgsz —
  для A/B скорости на конкретной машине; infer и age видны в [тайминг].
Превью — в полразмера панелей (экономия ~15 мс/кадр на imshow).
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO

from camera_open import latest_frames
from fusion import OneEuro, fuse_joints
from marker_frame import MarkerSession, to_stage_relative

W2D = "YOLO pose | 3D (smart fusion)"

# ---- COCO-17 скелет ---------------------------------------------------------
CONNECTIONS = [
    (5, 6), (5, 11), (6, 12), (11, 12),              # торс
    (5, 7), (7, 9), (6, 8), (8, 10),                 # руки
    (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6),  # лицо
    (11, 13), (13, 15), (12, 14), (14, 16),          # ноги
]
TORSO, LARM, RARM, LLEG, RLEG, FACE = range(6)
PART_COLOR = {  # BGR
    TORSO: (235, 235, 235), LARM: (255, 170, 70), RARM: (70, 170, 255),
    LLEG: (80, 200, 60), RLEG: (60, 120, 255), FACE: (200, 200, 120),
}
IDX2PART = {i: FACE for i in range(5)}
IDX2PART.update({i: LARM for i in (5, 7, 9)})
IDX2PART.update({i: RARM for i in (6, 8, 10)})
IDX2PART.update({i: LLEG for i in (11, 13, 15)})
IDX2PART.update({i: RLEG for i in (12, 14, 16)})
KEY_JOINTS = {0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}


def part_of(i, j):
    return IDX2PART[i] if IDX2PART[i] == IDX2PART[j] else TORSO


# ---- 3D-рендер (COCO-версия) --------------------------------------------------
class View3D:
    def __init__(self, w=1280, h=720, scale_px=280.0):
        self.w, self.h, self.s = w, h, scale_px
        self.yaw, self.pitch = np.radians(-25), np.radians(12)

    def project(self, pts):
        cy, sy = np.cos(self.yaw), np.sin(self.yaw)
        cp, sp = np.cos(self.pitch), np.sin(self.pitch)
        x1 = pts[:, 0] * cy + pts[:, 2] * sy
        z1 = -pts[:, 0] * sy + pts[:, 2] * cy
        y1 = pts[:, 1] * cp - z1 * sp
        return np.stack([self.w / 2 + x1 * self.s, self.h * 0.55 + y1 * self.s], axis=1)

    def draw(self, pts, info=""):
        canvas = np.full((self.h, self.w, 3), 28, np.uint8)
        if not pts:
            cv2.putText(canvas, "no pose", (20, 40), 6, 0.9, (120, 120, 120), 2)
            return canvas
        pelvis = (pts[11] + pts[12]) / 2 if 11 in pts and 12 in pts else next(iter(pts.values()))
        rel = {i: p - pelvis for i, p in pts.items()}
        idx = sorted(rel)
        arr = np.stack([rel[i] for i in idx])
        pos_of = {i: k for k, i in enumerate(idx)}
        proj = self.project(arr)
        feet = [rel[i][1] for i in (15, 16) if i in rel]
        y_g = max(feet) + 0.05 if feet else 0.9
        for g in np.arange(-1.5, 1.51, 0.25):
            for a, b in (((g, y_g, -1.5), (g, y_g, 1.5)), ((-1.5, y_g, g), (1.5, y_g, g))):
                pa, pb = self.project(np.array([a, b]))
                cv2.line(canvas, tuple(pa.astype(int)), tuple(pb.astype(int)), (55, 55, 58), 1)
        for i, j in CONNECTIONS:
            if i in pos_of and j in pos_of:
                cv2.line(canvas, tuple(proj[pos_of[i]].astype(int)), tuple(proj[pos_of[j]].astype(int)),
                         PART_COLOR[part_of(i, j)], 4, cv2.LINE_AA)
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
        if key in (81, 2): self.yaw -= np.radians(8)
        elif key in (83, 3): self.yaw += np.radians(8)
        elif key in (82, 0): self.pitch += np.radians(6)
        elif key in (84, 1): self.pitch -= np.radians(6)


# ---- main ---------------------------------------------------------------------
def load_model(size, backend="auto"):
    """Платформенный выбор бекенда (см. bench_onnx.py — тайминги):
      Windows/Linux: OpenVINO-экспорт (x2.2 к torch-cpu), фолбек .pt+cpu;
      macOS:         CoreML .mlpackage (GPU/ANE), фолбек .pt+mps.
    Возвращает (модель, device|None): OV/CoreML игнорируют device."""
    pt = f"models/yolo11{size}-pose.pt"
    if sys.platform == "darwin":
        ml = f"models/yolo11{size}-pose.mlpackage"
        if os.path.isdir(ml):
            print(f"YOLO11{size}-pose, бекенд CoreML: {ml}", flush=True)
            return YOLO(ml), None
        print(f"YOLO11{size}-pose, бекенд torch-mps: {pt} "
              f"(для CoreML: yolo export model={pt} format=coreml)", flush=True)
        return YOLO(pt), "mps"
    ov = f"models/yolo11{size}-pose_openvino_model"
    if backend == "openvino" and not os.path.isdir(ov):
        raise SystemExit(f"нет {ov}: yolo export model={pt} format=openvino")
    if backend != "torch" and os.path.isdir(ov):
        print(f"YOLO11{size}-pose, бекенд OpenVINO: {ov}", flush=True)
        return YOLO(ov), None
    print(f"YOLO11{size}-pose, бекенд torch-cpu: {pt} "
          f"(экспортируй OpenVINO для x2.2: yolo export format=openvino)", flush=True)
    return YOLO(pt), "cpu"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("size", nargs="?", choices=["n", "s", "m", "l", "x"], default="s")
    parser.add_argument("filters", nargs="?", choices=["full", "min", "off"], default="min")
    parser.add_argument("--backend", choices=["auto", "torch", "openvino"], default="auto")
    parser.add_argument("--imgsz", type=int, default=640, help="вход YOLO, px (кратно 32)")
    parser.add_argument("--ws", action="store_true")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--log", help="JSONL of downlink and headset uplink")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--seconds", type=float, default=0)
    parser.add_argument("--record-rgbd", help="New directory for unfiltered aligned RGB-D frames")
    parser.add_argument("--record-rgbd-seconds", type=float, default=60)
    parser.add_argument("--marker-m", type=float, default=0.16,
                        help="сторона чёрного квадрата AprilTag, метры")
    parser.add_argument("--marker-id", type=int, default=0)
    parser.add_argument("--marker-hand", choices=["right", "left"], default="right")
    parser.add_argument("--session", help="JSON связки камера→STAGE: загрузить, если файл есть, и писать сюда по клавише c")
    args = parser.parse_args()
    size, filt = args.size, args.filters
    server = None
    rgbd_recorder = None
    session = MarkerSession(args.marker_m, args.marker_id, args.marker_hand)
    if args.session and os.path.isfile(args.session):
        session.load(args.session)
        print(f"Связка камера→STAGE загружена из {args.session}", flush=True)
    model, device = load_model(size, args.backend)

    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

    # depth-фильтры SDK (stateful, создаём один раз)
    spatial = rs.spatial_filter()
    spatial.set_option(rs.option.filter_magnitude, 2)
    spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
    spatial.set_option(rs.option.filter_smooth_delta, 20)
    temporal = rs.temporal_filter()
    temporal.set_option(rs.option.holes_fill, 3)
    holes = rs.hole_filling_filter()
    # Цепочка по режимам (тайминг на кадр, i5-12450H, 1280x720):
    #   full = spatial+temporal+holes ~45 мс — максимум качества (джиттер x0.5);
    #   min  = только spatial          ~18 мс — баланс (по умолчанию);
    #   off  = без фильтров            ~0 мс  — гейт-медиана фьюжна берёт на
    #          себя большую часть работы (A/B в analyze: разница мала)
    chain = [] if filt == "off" else [spatial]
    if filt == "full":
        chain += [temporal, holes]
    print(f"depth-фильтры: {filt}", flush=True)

    def open_camera():
        while True:
            if args.seconds and time.monotonic() - started >= args.seconds:
                raise TimeoutError("Camera did not become available")
            try:
                from camera_open import open_pipeline
                return open_pipeline(rs, cfg)
            except Exception as e:
                print(f"[!] Камера недоступна ({e}). Жду переподключения... (q — выход)", flush=True)
                for _ in range(10):
                    if args.headless:
                        time.sleep(.2)
                        continue
                    k = cv2.waitKey(200) & 0xFF
                    if k in (ord('q'), 27) or cv2.getWindowProperty(W2D, cv2.WND_PROP_VISIBLE) < 1:
                        raise SystemExit(0)

    if not args.headless:
        cv2.namedWindow(W2D, cv2.WINDOW_NORMAL)
    # превью в полразмера: hstack двух 720p-панелей + imshow стоили ~20 мс/кадр
    view = View3D(w=640, h=360, scale_px=140)
    filters = {}
    t0, frames = time.monotonic(), 0
    fps = 0.0
    pipeline = None
    marker_K = marker_dist = None
    started = time.monotonic()
    sequence = 0
    stage_t = {}  # имя стадии -> суммарные мс за секундное окно (раз в сек в консоль)
    age_sum = dropped_sum = 0.0  # возраст кадра на выходе из wait и выброшенные старые кадры

    def tic(name, t):
        now = time.perf_counter()
        stage_t[name] = stage_t.get(name, 0.0) + (now - t) * 1000
        return now
    try:
        if args.record_rgbd:
            from rgbd_recording import RGBDRecorder
            rgbd_recorder = RGBDRecorder(args.record_rgbd, args.record_rgbd_seconds)
        if args.ws:
            from skeleton_server import SkeletonServer, packet
            backend = "torch-" + device if device else ("coreml" if sys.platform == "darwin" else "openvino")
            server = SkeletonServer(args.host, args.port, args.log)
        while True:
            if args.seconds and time.monotonic() - started >= args.seconds:
                break
            if pipeline is None:
                filters.clear()
                pipeline, profile = open_camera()
                align = rs.align(rs.stream.color)
                depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
                color_intr = (profile.get_stream(rs.stream.color)
                              .as_video_stream_profile().get_intrinsics())
                marker_K = np.array([[color_intr.fx, 0, color_intr.ppx],
                                     [0, color_intr.fy, color_intr.ppy],
                                     [0, 0, 1]], np.float64)
                marker_dist = np.asarray(color_intr.coeffs, np.float64)
                print(f"USB: {profile.get_device().get_info(rs.camera_info.usb_type_descriptor)}", flush=True)
            t = time.perf_counter()
            try:
                fs, dropped = latest_frames(pipeline, timeout_ms=5000)
            except RuntimeError as e:
                print(f"[!] Камера отвалилась: {e}", flush=True)
                try:
                    pipeline.stop()
                except RuntimeError:
                    pass
                pipeline = None
                continue
            host_ms = time.time_ns() // 1_000_000
            frame_timestamp, clock = host_ms, "host_after_wait"
            color_clock = fs.get_color_frame()
            if color_clock:
                domain = color_clock.get_frame_timestamp_domain()
                if domain in (rs.timestamp_domain.global_time, rs.timestamp_domain.system_time):
                    frame_timestamp = int(color_clock.get_timestamp())
                    clock = str(domain).split(".")[-1]
            # host_after_wait: возраст кадра неизвестен (0), t позже реального снимка
            timing = {"clock": clock, "frame_age_ms": host_ms - frame_timestamp, "dropped": dropped}
            age_sum += timing["frame_age_ms"]
            dropped_sum += dropped
            t = tic("wait", t)
            fs = align.process(fs)
            t = tic("align", t)
            color_f, depth_f = fs.get_color_frame(), fs.get_depth_frame()
            if not color_f or not depth_f:
                continue
            if rgbd_recorder:
                rgbd_recorder.submit(np.asanyarray(color_f.get_data()),
                    np.asanyarray(depth_f.get_data()).astype(np.float32) * depth_scale,
                    [color_intr.fx, color_intr.fy, color_intr.ppx, color_intr.ppy],
                    frame_timestamp, sequence + 1,
                    {"distortion_model": str(color_intr.model),
                     "distortion_coeffs": np.array(color_intr.coeffs),
                     "timestamp_domain": str(color_f.get_frame_timestamp_domain())})
            # depth-фильтры SDK по выбранному режиму
            for flt in chain:
                depth_f = flt.process(depth_f)
            t = tic("filters", t)
            color = np.asanyarray(color_f.get_data())
            depth_m = np.asanyarray(depth_f.get_data()).astype(np.float32) * depth_scale
            if marker_K is not None:
                session.observe(color, marker_K, marker_dist)
            t = tic("to_numpy", t)

            res = model.predict(color, imgsz=args.imgsz, conf=0.3, classes=0,
                                **({"device": device} if device else {}), verbose=False)[0]
            t = tic("infer", t)
            raw, pts2d, sources, confidence = {}, {}, {}, {}
            if res.keypoints is not None and len(res.keypoints):
                kxy = res.keypoints.xy.cpu().numpy()
                kconf = (res.keypoints.conf.cpu().numpy()
                         if res.keypoints.conf is not None else np.ones(kxy.shape[:-1]))
                xy, conf = (kxy[0], kconf[0]) if kxy.ndim == 3 else (kxy, kconf)
                for i in range(min(17, len(xy))):
                    if conf[i] > 0.5:
                        confidence[i] = float(conf[i])
                        pts2d[i] = (float(xy[i][0]), float(xy[i][1]))

                # 3D: фьюжн общий с analyze_bag.py (fusion.py)
                raw = fuse_joints(depth_m, pts2d, color_intr, sources=sources)
                t = tic("fuse", t)

                # отрисовка 2D
                for i, j in CONNECTIONS:
                    if i in pts2d and j in pts2d:
                        cv2.line(color, tuple(map(int, pts2d[i])), tuple(map(int, pts2d[j])),
                                 PART_COLOR[part_of(i, j)], 2, cv2.LINE_AA)
                for i, (u, v) in pts2d.items():
                    r, c = (6, (255, 255, 255)) if i in KEY_JOINTS else (3, PART_COLOR[IDX2PART[i]])
                    cv2.circle(color, (int(u), int(v)), r, c, -1, cv2.LINE_AA)
            t = tic("draw", t)

            # One-Euro на сустав: фактический dt, состояние переживает дропаут
            t_now = time.monotonic()
            for i, p in raw.items():
                filters.setdefault(i, OneEuro())(p, t_now)
            smoothed = {i: f.x_prev.copy() for i, f in filters.items() if i in raw}

            sequence += 1
            if server:
                wire, wire_frame, pelvis = smoothed, "camera", None
                if session.locked:
                    server.record("camera", packet(smoothed, confidence, sources, frame_timestamp,
                                                   sequence, fps, backend, timing=timing))
                    converted = to_stage_relative(smoothed, session.R, session.t)
                    wire_frame = "pelvis-relative"
                    if converted:
                        wire, pelvis_xyz = converted
                        pelvis = {"p": pelvis_xyz, "conf": min(confidence.get(11, 0), confidence.get(12, 0))}
                    else:
                        wire, pelvis = {}, {"p": [0, 0, 0], "conf": 0}
                server.publish(packet(wire, confidence, sources, frame_timestamp,
                                      sequence, fps, backend, wire_frame, pelvis, timing))
            frames += 1
            dt = time.monotonic() - t0
            if dt > 1:
                n = frames
                fps, t0, frames = n / dt, time.monotonic(), 0
                br = " ".join(f"{k}:{v / max(1, n):.0f}ms"
                              for k, v in sorted(stage_t.items(), key=lambda kv: -kv[1]))
                print(f"    [тайминг] {br} | age:{age_sum / max(1, n):.0f}ms ({clock}) "
                      f"drop:{dropped_sum / max(1, n):.1f}", flush=True)
                stage_t.clear()
                age_sum = dropped_sum = 0.0
            if args.headless:
                continue
            if session.corners is not None:
                quad = session.corners.astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(color, [quad], True, (0, 220, 255), 2, cv2.LINE_AA)
            pelvis = smoothed.get(11)
            marker = "STAGE" if session.locked else ("маркер" if session.corners is not None else "нет маркера")
            info = (f"fps {fps:4.1f} | joints {len(smoothed)}/17 | "
                    f"pelvis {'-' if pelvis is None else f'{pelvis[2]:.2f} m'} | {marker}")
            cv2.putText(color, info, (12, 30), 6, 0.7, (60, 255, 60), 2)
            cv2.putText(color, "c — зафиксировать камеру контроллером", (12, 86),
                        6, 0.55, (200, 200, 200), 1)
            cv2.putText(color, f"yolo11{size}-pose  q-quit", (12, 58), 6, 0.55, (200, 200, 200), 1)

            cv2.imshow(W2D, np.hstack([cv2.resize(color, (640, 360)),
                                       view.draw(smoothed, info)]))
            if cv2.getWindowProperty(W2D, cv2.WND_PROP_VISIBLE) < 1:
                print("Окно закрыто — выходим.", flush=True)
                break
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            if key == ord('c'):
                if server is None:
                    print("Фиксация маркера нужна с --ws: шлем присылает позу контроллера.", flush=True)
                else:
                    ok, reason = session.try_lock(server.latest_uplink())
                    print(reason, flush=True)
                    if ok:
                        path = args.session or "session.json"
                        session.save(path)
                        print(f"Связка записана в {path}", flush=True)
            view.rotate(key)
            tic("show", t)
    finally:
        if rgbd_recorder:
            rgbd_recorder.close()
        if pipeline is not None:
            try:
                pipeline.stop()
            except RuntimeError:
                pass
        if server:
            server.close()
        if not args.headless:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
