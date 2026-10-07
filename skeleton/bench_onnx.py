"""
Бенчмарк бекендов инференса YOLO11s-pose на кадрах из записи .db3.

Сравнивает (одинаковый пре/постпроцессинг ultralytics, отличается только
исполнение графа):
  1. torch-cpu (.pt) — базовый бейзлайн
  2. onnxruntime-cpu (.onnx) и openvino (_openvino_model/) — Windows/Linux
  3. torch-mps (.pt) и coreml (.mlpackage) — macOS (OpenVINO под Apple
     Silicon нет)

Без pyrealsense2 (macOS: колёс нет) бенчит один кадр из data/_bench_frame.npy
N раз — тайминг бекенда, не пайплайна. Плюс sanity: кейпоинты кадра 0
с conf>0.5 должны совпадать между бекендами.

Запуск:
  python bench_onnx.py [n_frames] [all|torch|ort|ov|mps|coreml]   # кадров 200
"""
import os
import sys
import time

import numpy as np

try:
    import pyrealsense2 as rs
except ImportError:  # macOS: колёс нет (PIPELINE.md §macOS) — кадр из .npy
    rs = None
from ultralytics import YOLO

N_FRAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 200
ONLY = sys.argv[2] if len(sys.argv) > 2 else "all"
BAG = "data/benchmark3.db3"
MAC = sys.platform == "darwin"

frames = []
if rs is not None:
    # кадры из бага: оба стрима + set_real_time(False) — проверенный рецепт
    # analyze_bag.py (color-only обрывается на ~32 кадрах); вьюшки — только .copy()
    cfg = rs.config()
    cfg.enable_device_from_file(BAG, repeat_playback=False)
    cfg.enable_stream(rs.stream.color, rs.format.any, 30)
    cfg.enable_stream(rs.stream.depth, rs.format.any, 30)
    pipe = rs.pipeline()
    profile = pipe.start(cfg)
    profile.get_device().as_playback().set_real_time(False)
    while len(frames) < N_FRAMES:
        try:
            fs = pipe.wait_for_frames(timeout_ms=5000)
        except RuntimeError:
            break
        c, d = fs.get_color_frame(), fs.get_depth_frame()
        if c and d:
            frames.append(np.asanyarray(c.get_data()).copy())
    pipe.stop()
    src = f"баг {BAG}"
else:
    frames = [np.load("data/_bench_frame.npy")] * N_FRAMES
    src = "data/_bench_frame.npy (один кадр x N; pyrealsense2 нет)"
print(f"кадров: {len(frames)} из {src} {frames[0].shape}")

PRED_ARGS = dict(imgsz=640, conf=0.3, classes=0, verbose=False)


def run_backend(name, model_path, device_arg):
    model = YOLO(model_path)
    for i in range(10):  # прогрев
        model.predict(frames[i % len(frames)], device=device_arg, **PRED_ARGS)
    times, kp0, conf0 = [], None, None
    for i, f in enumerate(frames):
        t0 = time.perf_counter()
        r = model.predict(f, device=device_arg, **PRED_ARGS)[0]
        times.append(time.perf_counter() - t0)
        if i == 0 and r.keypoints is not None and len(r.keypoints):
            kp0 = r.keypoints.xy.cpu().numpy()[0]
            conf0 = (r.keypoints.conf.cpu().numpy()[0]
                     if r.keypoints.conf is not None else np.ones(len(kp0)))
    times = np.array(times) * 1000
    print(f"{name:<26} медиана {np.median(times):6.1f} мс | p95 {np.percentile(times, 95):6.1f} | "
          f"max {times.max():6.1f} | fps {1000.0 / np.median(times):5.1f}")
    return np.median(times), kp0, conf0


def sanity(name_a, kp_a, c_a, name_b, kp_b, c_b):
    if kp_a is None or kp_b is None:
        print(f"sanity {name_a} vs {name_b}: нет детекции в одном из бекендов")
        return
    m = (c_a > 0.5) & (c_b > 0.5)
    if not m.any():
        print(f"sanity {name_a} vs {name_b}: нет общих уверенных кейпоинтов")
        return
    d = np.abs(kp_a[m] - kp_b[m])
    ok = np.median(d) < 1.0 and d.max() < 4.0
    print(f"sanity {name_a} vs {name_b}: {m.sum()} кт, медиана {np.median(d):.2f} px, "
          f"max {d.max():.2f} px — {'ок' if ok else 'РАСХОЖДЕНИЕ'}")


print()
results = {}
if ONLY in ("all", "torch"):
    results["torch"] = run_backend("torch-cpu (.pt)", "models/yolo11s-pose.pt", "cpu")
if ONLY in ("all", "mps") and MAC:
    results["mps"] = run_backend("torch-mps (.pt)", "models/yolo11s-pose.pt", "mps")
if ONLY in ("all", "ort") and not MAC:
    results["ort"] = run_backend("onnxruntime (.onnx)", "models/yolo11s-pose.onnx", None)
if ONLY in ("all", "ov") and not MAC and os.path.isdir("models/yolo11s-pose_openvino_model"):
    results["ov"] = run_backend("openvino (IR)",
                                "models/yolo11s-pose_openvino_model/", None)
if ONLY in ("all", "coreml") and MAC and os.path.isdir("models/yolo11s-pose.mlpackage"):
    results["coreml"] = run_backend("coreml (.mlpackage)",
                                    "models/yolo11s-pose.mlpackage", None)

if len(results) > 1:
    print()
    base = results["torch"][0]
    for k, (t, _, _) in results.items():
        if k != "torch":
            print(f"ускорение vs torch: {k} -> x{base / t:.2f}")
    if "ov" in results:
        sanity("torch", *results["torch"][1:], "ov", *results["ov"][1:])
    elif "ort" in results:
        sanity("torch", *results["torch"][1:], "ort", *results["ort"][1:])
