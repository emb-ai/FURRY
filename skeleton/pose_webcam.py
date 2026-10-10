"""2D-траектория BlazePose-33 с UVC-камеры ноутбука.

Метрический фьюжн pose_yolo.py требует RealSense D435i. Встроенная камера
отдаёт только RGB, поэтому лог — пиксели и visibility, не метры камеры.
World-координаты MediaPipe сюда не пишутся: их масштаб не сенсорный.
"""
import argparse
import json
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)
NAMES = (
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer", "left_ear", "right_ear",
    "mouth_left", "mouth_right", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_pinky",
    "right_pinky", "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle",
    "right_ankle", "left_heel", "right_heel", "left_foot_index",
    "right_foot_index",
)
LEG = (
    "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle",
    "right_ankle", "left_heel", "right_heel", "left_foot_index",
    "right_foot_index",
)
CONNECTIONS = (
    (11, 12), (11, 23), (12, 24), (23, 24),
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
)
LEG_IDX = {23, 24, 25, 26, 27, 28, 29, 30, 31, 32}


def ensure_model(path: Path) -> Path:
    if path.is_file() and path.stat().st_size > 0:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"скачиваю {path.name}", flush=True)
    urllib.request.urlretrieve(MODEL_URL, path)
    return path


def joints_from_landmarks(landmarks, width, height):
    joints = {}
    for i, name in enumerate(NAMES):
        if i >= len(landmarks):
            break
        mark = landmarks[i]
        vis = None if mark.visibility is None else float(mark.visibility)
        presence = None if mark.presence is None else float(mark.presence)
        joints[name] = {
            "uv": [round(float(mark.x) * width, 2), round(float(mark.y) * height, 2)],
            "visibility": None if vis is None else round(vis, 4),
            "presence": None if presence is None else round(presence, 4),
        }
    return joints


def path_length(frames, name):
    pts = []
    for frame in frames:
        joint = frame["joints"].get(name)
        if joint and (joint["visibility"] or 0) >= 0.5:
            pts.append(joint["uv"])
    if len(pts) < 2:
        return 0.0
    arr = np.asarray(pts, dtype=np.float64)
    return float(np.linalg.norm(np.diff(arr, axis=0), axis=1).sum())


def summarize(frames, width, height):
    n = len(frames)
    posed = [f for f in frames if f["joints"]]
    legs = {}
    for name in LEG:
        seen = sum(
            1 for f in frames
            if ((f["joints"].get(name) or {}).get("visibility") or 0) >= 0.5
        )
        legs[name] = {
            "visible_fraction": 0 if n == 0 else round(seen / n, 3),
            "path_px": round(path_length(frames, name), 1),
        }
    return {
        "frames": n,
        "posed_frames": len(posed),
        "size": [width, height],
        "frame": "image",
        "units": "pixels",
        "legs": legs,
        "nose_path_px": round(path_length(frames, "nose"), 1),
    }


def draw_pose(frame, joints):
    h, w = frame.shape[:2]

    def xy(name):
        joint = joints.get(name)
        if not joint or (joint["visibility"] or 0) < 0.5:
            return None
        u, v = joint["uv"]
        if not (0 <= u < w and 0 <= v < h):
            return None
        return int(u), int(v)

    for i, j in CONNECTIONS:
        a, b = xy(NAMES[i]), xy(NAMES[j])
        if a and b:
            color = (80, 220, 80) if i in LEG_IDX and j in LEG_IDX else (235, 235, 235)
            cv2.line(frame, a, b, color, 2, cv2.LINE_AA)
    for name in NAMES:
        point = xy(name)
        if point:
            color = (80, 220, 80) if name in LEG else (255, 255, 255)
            cv2.circle(frame, point, 4, color, -1, cv2.LINE_AA)
    return frame


def open_camera(device, width, height):
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    if not cap.isOpened():
        raise RuntimeError(f"не открылась камера {device}")
    return cap


def main():
    parser = argparse.ArgumentParser(description="Запись 2D-траектории позы с веб-камеры")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--seconds", type=float, default=20)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "pose_landmarker_full.task")
    parser.add_argument("--preview-jpg", type=Path, default=None)
    args = parser.parse_args()
    if (args.output / "trajectory.jsonl").exists():
        raise SystemExit(
            f"уже есть {args.output / 'trajectory.jsonl'}; укажите новый --output")

    from mediapipe.tasks.python.core import base_options as base_options_lib
    from mediapipe.tasks.python.vision import pose_landmarker as landmarker_lib
    from mediapipe.tasks.python.vision.core import image as image_lib
    from mediapipe.tasks.python.vision.core import vision_task_running_mode as mode_lib

    model = ensure_model(args.model)
    options = landmarker_lib.PoseLandmarkerOptions(
        base_options=base_options_lib.BaseOptions(model_asset_path=str(model)),
        running_mode=mode_lib.VisionTaskRunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = landmarker_lib.PoseLandmarker.create_from_options(options)
    cap = open_camera(args.device, args.width, args.height)
    ok, frame = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError("камера открылась, но кадр не пришёл")
    height, width = frame.shape[:2]
    args.output.mkdir(parents=True, exist_ok=True)
    trajectory = args.output / "trajectory.jsonl"
    preview = args.output / "preview.avi"
    writer = cv2.VideoWriter(
        str(preview), cv2.VideoWriter_fourcc(*"MJPG"), 15, (width, height))
    manifest = {
        "device": "USB2.0 HD UVC WebCam",
        "node": f"/dev/video{args.device}",
        "size": [width, height],
        "fourcc": "MJPG",
        "detector": "mediapipe-pose-landmarker-full",
        "model": str(model),
        "frame": "image",
        "units": "pixels",
        "note": "Нет сенсора глубины. Это не метрические XYZ камеры и не GT ног.",
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    show = True
    frames = []
    sequence = 0
    last_ts = -1
    t0 = time.monotonic()
    annotated = None
    try:
        with trajectory.open("w", encoding="utf-8") as log:
            while time.monotonic() - t0 < args.seconds:
                ok, frame = cap.read()
                if not ok:
                    continue
                rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                stamp = int((time.monotonic() - t0) * 1000)
                if stamp <= last_ts:
                    stamp = last_ts + 1
                last_ts = stamp
                started = time.perf_counter()
                result = landmarker.detect_for_video(
                    image_lib.Image(image_lib.ImageFormat.SRGB, rgb), stamp)
                infer_ms = (time.perf_counter() - started) * 1000
                landmarks = result.pose_landmarks[0] if result.pose_landmarks else []
                joints = joints_from_landmarks(landmarks, width, height)
                sequence += 1
                row = {
                    "t": time.time_ns() // 1_000_000,
                    "seq": sequence,
                    "t_video_ms": stamp,
                    "infer_ms": round(infer_ms, 1),
                    "frame": "image",
                    "size": [width, height],
                    "joints": joints,
                }
                log.write(json.dumps(row, ensure_ascii=False) + "\n")
                frames.append(row)
                draw_pose(frame, joints)
                visible_legs = sum(
                    1 for name in LEG
                    if ((joints.get(name) or {}).get("visibility") or 0) >= 0.5)
                cv2.putText(
                    frame,
                    f"seq {sequence}  legs {visible_legs}/{len(LEG)}  {infer_ms:.0f} ms",
                    (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 255, 40), 2)
                if writer.isOpened():
                    writer.write(frame)
                if sequence == 1 or sequence % 15 == 0:
                    annotated = frame.copy()
                if show:
                    try:
                        cv2.imshow("webcam pose", frame)
                        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                            break
                    except cv2.error:
                        show = False
    finally:
        cap.release()
        writer.release()
        landmarker.close()
        if show:
            cv2.destroyAllWindows()

    summary = summarize(frames, width, height)
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if annotated is not None:
        target = args.preview_jpg or (args.output / "annotated.jpg")
        target.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(target), annotated)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"траектория: {trajectory}", flush=True)


if __name__ == "__main__":
    main()
