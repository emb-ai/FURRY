"""
Экстракция L2-датасета из .db3 бага: один проход -> l2_dataset/<стем>.npz.

На каждый color-кадр (дедуп по таймстампу, см. analyze_bag.py §6.8):
  - YOLO 2D кейпоинты (COCO-17, OpenVINO) — вход v1 + для оффсетов
  - AprilTag 36h11 (subpix) с интриниками бага — GT-сырьё
  - depth-фьюжн 3D-скелет (fusion.fuse_joints): depth сырой, пиксели
    проецируются color->depth через заводские экстринсики (§6.3: align на
    плейбэке не работает), фильтры SDK те же, что в live (min-режим)

Выход: t_ms (n,), kpts2d (n,17,2) nan-если-нет, conf2d (n,17),
fusion_xyz (n,17,3) nan, fusion_conf (n,17), fusion_src (n,17) int8
(0 window, 1 line, 2 missing), tag_tvec/tag_rvec (n,K,3) nan,
tag_seen (n,K), tag_side_px (n,K), fx/fy/cx/cy бага.

Запуск: .venv/Scripts/python l2_extract.py <path.bag> [ещё.bag ...]
"""
import os
import sys
import time

import cv2
import cv2.aruco as ar
import numpy as np
import pyrealsense2 as rs

from tag_tracker import TAG_SIZE_MM, estimate_pose

OUT_DIR = "l2_dataset"
TAG_IDS = sorted(TAG_SIZE_MM)
# заводские экстринсики живой D435i (color<->depth) — из analyze_bag.py §6.3
C2D_ROT = [0.9999816417694092, 0.0033040582202374935, -0.00040445956983603537,
           -0.003309598368960917, 0.9999395012857759, 0.005083478455894375,
           0.0004213851907745725, -0.005082169882025753, 0.999987006187439]
C2D_TRA = [0.014620114117860794, -0.00017548960749991238, 0.0003132132987957448]
D2C_ROT = [0.9999816417694092, -0.0033040582202374935, 0.00040445956983603537,
           0.0033040582202374935, 0.9999816417694092, 0.005083478455894375,
           -0.00040445956983603537, -0.005082169882025753, 0.999987006187439]
D2C_TRA = [0.014620114117860794, 0.00017548960749991238, 0.0003132132987957448]

# детекторы
P = ar.DetectorParameters()
P.cornerRefinementMethod = ar.CORNER_REFINE_SUBPIX
DET = ar.ArucoDetector(ar.getPredefinedDictionary(ar.DICT_APRILTAG_36H11), P)


def load_yolo():
    from ultralytics import YOLO
    for cand in ("models/yolo11s-pose_openvino_model",
                 "models/yolo11s-pose.onnx", "models/yolo11s-pose.pt"):
        if os.path.exists(cand):
            print("YOLO:", cand, flush=True)
            return YOLO(cand)
    raise SystemExit("нет YOLO-модели")


def extract(bag: str, yolo):
    stem = os.path.splitext(os.path.basename(bag))[0]
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f"{stem}.npz")
    part = out + ".part.npz"

    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device_from_file(bag, False)
    profile = pipe.start(cfg)
    pb = profile.get_device().as_playback()
    pb.set_real_time(False)

    s = profile.get_streams()[0]           # любой стрим даёт профильное инфо
    color_prof = (profile.get_stream(rs.stream.color)
                  .as_video_stream_profile())
    depth_prof = (profile.get_stream(rs.stream.depth)
                  .as_video_stream_profile())
    color_intr = color_prof.get_intrinsics()
    depth_intr = depth_prof.get_intrinsics()
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    K = np.array([[color_intr.fx, 0, color_intr.ppx],
                  [0, color_intr.fy, color_intr.ppy], [0, 0, 1]], np.float64)
    dist = np.array(color_intr.coeffs, np.float64)
    c2d, d2c = rs.extrinsics(), rs.extrinsics()
    c2d.rotation, c2d.translation = C2D_ROT, C2D_TRA
    d2c.rotation, d2c.translation = D2C_ROT, D2C_TRA

    spatial = rs.spatial_filter()
    spatial.set_option(rs.option.filter_magnitude, 2)
    spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
    spatial.set_option(rs.option.filter_smooth_delta, 20)

    # fusion.py берём из FURRY/skeleton (в sirius_twist лежит старая копия без sources)
    import sys as _sys
    _sys.path.insert(0, "C:/Users/denis/ml/self_assistant/FURRY/skeleton")
    from fusion import fuse_joints

    rows = {"t": [], "k2d": [], "c2d": [], "fxyz": [], "fsrc": [],
            "tv": [], "rv": [], "seen": [], "px": []}
    # резюм: чекпоинт каждые 2000 кадров (атомарно), при старте продолжаем с него
    # резюм: берём самый полный из part/tmp (tmp мог остаться от падения на replace)
    cand = [p for p in (part, part + ".tmp.npz") if os.path.exists(p)]
    if cand:
        best = max(cand, key=lambda p: os.path.getsize(p))
        zp = np.load(best, allow_pickle=True)
        keymap = {"t": "t", "k2d": "kpts2d", "c2d": "conf2d", "fxyz": "fusion_xyz",
                  "fsrc": "fusion_src", "tv": "tag_tvec", "rv": "tag_rvec",
                  "seen": "tag_seen", "px": "tag_px"}
        for k, nk in keymap.items():
            rows[k] = list(zp[nk])
        zp.close()
        rel_ms = float(rows["t"][-1]) - float(rows["t"][0])   # seek относителен файлу
        pb.seek(__import__("datetime").timedelta(milliseconds=rel_ms + 1.0))
        print(f"резюм с чекпоинта: {len(rows['t'])} кадров, +{rel_ms/1000:.0f} с от начала",
              flush=True)
    n, last_ts, ts_prev = 0, None, 0.0
    if rows["t"]:
        ts_prev = rows["t"][-1]
        last_ts = ts_prev          # дедуп против пересечения с чекпоинтом
    t0 = time.perf_counter()

    def save(path, final=False):
        t = np.array(rows["t"], np.float64)
        tmp = path + ".tmp.npz"
        np.savez_compressed(
            tmp, t=t, kpts2d=np.stack(rows["k2d"]), conf2d=np.stack(rows["c2d"]),
            fusion_xyz=np.stack(rows["fxyz"]),
            fusion_src=np.stack(rows["fsrc"]),
            tag_tvec=np.stack(rows["tv"]), tag_rvec=np.stack(rows["rv"]),
            tag_seen=np.stack(rows["seen"]), tag_px=np.stack(rows["px"]),
            tag_ids=np.array(TAG_IDS), final=final,
            fx=color_intr.fx, fy=color_intr.fy, cx=color_intr.ppx, cy=color_intr.ppy)
        for _ in range(6):                      # файл может держать AV/индексатор
            try:
                os.replace(tmp, path)
                return
            except PermissionError:
                time.sleep(1.0)
        print(f"[!] чекпоинт не заменился ({path}) — продолжаю, следующий попробует снова",
              flush=True)
    try:
        while True:
            try:
                fs = pipe.wait_for_frames(timeout_ms=5000)
            except RuntimeError:
                break
            color_f, depth_f = fs.get_color_frame(), fs.get_depth_frame()
            if not color_f or not depth_f:
                continue
            n += 1
            ts_raw = float(color_f.get_timestamp())
            if ts_raw == last_ts:            # дедуп дублей color (§6.8)
                continue
            last_ts = ts_raw
            ts = max(ts_raw, ts_prev + 1.0)
            ts_prev = ts

            d_buf = depth_f.get_data()       # сырой буфер для проекции пикселей
            depth_f = spatial.process(depth_f)
            depth_m = np.asanyarray(depth_f.get_data()).astype(np.float32) * depth_scale
            color = np.asanyarray(color_f.get_data())

            # YOLO 2D
            kpts = np.full((17, 2), np.nan, np.float32)
            conf = np.zeros(17, np.float32)
            res = yolo(color, verbose=False, conf=0.25)[0]
            if res.keypoints is not None and len(res.keypoints.xy):
                k = res.keypoints
                xy = k.xy[0].cpu().numpy() if k.xy.ndim == 3 else k.xy.cpu().numpy()
                cc = (k.conf[0].cpu().numpy() if k.conf is not None
                      else np.ones(len(xy)))
                m = min(17, len(xy))
                kpts[:m], conf[:m] = xy[:m], cc[:m]
            keep = conf > 0.5
            pts2d = {i: (float(kpts[i, 0]), float(kpts[i, 1]))
                     for i in range(17) if keep[i]}

            # depth-фьюжн (обход align): пиксели color -> depth, фьюжн в depth-интринсиках
            dpx = {}
            for i, (u, v) in pts2d.items():
                try:
                    dp = rs.rs2_project_color_pixel_to_depth_pixel(
                        d_buf, depth_scale, 0.1, 5.0, depth_intr, color_intr,
                        c2d, d2c, [u, v])
                    dpx[i] = (float(dp[0]), float(dp[1]))
                except Exception:
                    continue
            raw, sources = {}, {}
            if dpx:
                raw = fuse_joints(depth_m, dpx, depth_intr, sources=sources)
                # точки в depth-фрейме -> color-фрейм (в live выравнивание делает align)
                Rd = np.array(d2c.rotation, np.float64).reshape(3, 3)
                td = np.array(d2c.translation, np.float64)
                raw = {i: Rd @ p + td for i, p in raw.items()}

            # AprilTag
            tv = np.full((len(TAG_IDS), 3), np.nan, np.float32)
            rv = np.full((len(TAG_IDS), 3), np.nan, np.float32)
            seen = np.zeros(len(TAG_IDS), bool)
            px = np.zeros(len(TAG_IDS), np.float32)
            corners, ids, _ = DET.detectMarkers(color)
            if ids is not None:
                for c_, i in zip(corners, ids.ravel()):
                    i = int(i)
                    if i not in TAG_SIZE_MM:
                        continue
                    q = c_.reshape(4, 2)
                    side = float(np.linalg.norm(q[1] - q[0]))
                    if side < 30:
                        continue
                    est = estimate_pose(c_, i, K, dist)
                    if est:
                        k = TAG_IDS.index(i)
                        rv[k], tv[k] = est[0].reshape(3), est[1].reshape(3)
                        seen[k], px[k] = True, side

            fxyz = np.full((17, 3), np.nan, np.float32)
            fsrc = np.full(17, 2, np.int8)
            for i, p in raw.items():
                fxyz[i] = p
                fsrc[i] = {"window": 0, "line": 1}.get(sources.get(i, "window"), 2)

            rows["t"].append(ts)
            rows["k2d"].append(kpts); rows["c2d"].append(conf)
            rows["fxyz"].append(fxyz); rows["fsrc"].append(fsrc)
            rows["tv"].append(tv); rows["rv"].append(rv)
            rows["seen"].append(seen); rows["px"].append(px)
            if len(rows["t"]) % 300 == 0:
                rate = (len(rows["t"]) - (0 if not os.path.exists(part) else 0)) / max(time.perf_counter() - t0, 1e-9)
                print(f"  {bag.split('/')[-1]}: {len(rows['t'])} кадров, "
                      f"{rate:.1f} fps", flush=True)
            if len(rows["t"]) % 2000 == 0:
                save(part)
    finally:
        pipe.stop()

    save(out, final=True)
    if os.path.exists(part):
        os.remove(part)
    span = (rows["t"][-1] - rows["t"][0]) / 1000 if len(rows["t"]) > 1 else 0
    seen_pct = {int(i): round(100 * float(rows["seen"][:, k].mean()), 0)
                for k, i in enumerate(TAG_IDS)}
    print(f"{out}: {len(rows['t'])} кадров, {span:.0f} с, теги seen%: {seen_pct}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    yolo = load_yolo()
    for bag in sys.argv[1:]:
        print(f"=== {bag}", flush=True)
        extract(bag.replace("/c/", "C:/"), yolo)
