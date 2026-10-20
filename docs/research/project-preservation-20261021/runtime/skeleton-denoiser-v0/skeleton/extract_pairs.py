"""
Экстрактор пар для денойзера скелета: avi (+ CSV тегов той же сессии) -> npz.

На каждый обрабатываемую пару кадров:
  1) YOLO11s-pose (COCO-17): 2D-кейпоинты + conf   [шумный вход денойзера]
  2) AprilTag 36h11 с APRILTAG-рефайном (~180 мс/кадр, офлайн можно):
     6DoF-позы тегов                                  [GT-якоря ног, точнее live-CSV]

Тайминг: t_s по кадрам из CSV (таймстампы камеры), иначе idx/30.

Выход <имя>_pairs.npz: t, kpts_xy (N,17,2), kpts_conf (N,17),
  tag_tvec (N,K,3), tag_rvec (N,K,3), tag_seen (N,K), tag_ids, K-матрица.

Запуск:
  .venv/Scripts/python extract_pairs.py tags_20261009_172221.avi --csv tags_20261009_172221.csv
  --stride 2        теги детектить каждые 2-й кадр (вдвое быстрее, YOLO — все кадры)
  --max-frames 300  только первые N кадров (отладка/выборка)
  --refine subpix   быстрый рефайн (7 мс) вместо точного apriltag (180 мс)
"""
import argparse
import os
import time

import cv2
import cv2.aruco as ar
import numpy as np

from tag_tracker import DICT, TAG_SIZE_MM, obj_points

# интринсики D435i color 1280x720 (фабричные, из live-сессий; при смене камеры -- править)
FX, FY, CX, CY = 907.9, 907.3, 657.7, 372.3
K = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1.0]])
DIST = np.zeros(5)
MIN_SIDE_PX = 36.0


def make_detector(refine: str) -> ar.ArucoDetector:
    P = ar.DetectorParameters()
    P.cornerRefinementMethod = (ar.CORNER_REFINE_APRILTAG if refine == "apriltag"
                                else ar.CORNER_REFINE_SUBPIX)
    P.minMarkerPerimeterRate = 0.02
    return ar.ArucoDetector(DICT, P)


def detect_tags(frame, det):
    """-> {id: (tvec(3,), rvec(3,))} по конфигу TAG_SIZE_MM."""
    corners, ids, _ = det.detectMarkers(frame)
    out = {}
    if ids is None:
        return out
    for c, i in zip(corners, ids.ravel()):
        i = int(i)
        if i not in TAG_SIZE_MM:
            continue
        q = c.reshape(4, 2)
        if np.linalg.norm(q[1] - q[0]) < MIN_SIDE_PX:
            continue
        size_m = TAG_SIZE_MM[i] / 1000.0
        res = cv2.solvePnP(obj_points(size_m), c, K, DIST, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if res[0]:
            out[i] = (res[2].reshape(3), res[1].reshape(3))
    return out


def main():
    ap = argparse.ArgumentParser(description="avi+csv -> npz пар (YOLO кейпоинты + теги)")
    ap.add_argument("avi")
    ap.add_argument("--csv", help="CSV тегов той же сессии (таймстампы кадров)")
    ap.add_argument("--stride", type=int, default=1, help="теги каждые N-й кадр")
    ap.add_argument("--max-frames", type=int, default=0, help="0 = все")
    ap.add_argument("--refine", choices=["apriltag", "subpix"], default="apriltag")
    ap.add_argument("--conf", type=float, default=0.5, help="YOLO conf-порог")
    ap.add_argument("--start", type=int, default=0, help="продолжить с кадра N (resume)")
    ap.add_argument("--checkpoint", type=int, default=3000,
                    help="сохранять промежуточный npz каждые N кадров")
    args = ap.parse_args()

    from ultralytics import YOLO
    yolo = YOLO("models/yolo11s-pose.pt")
    det = make_detector(args.refine)

    cap = cv2.VideoCapture(args.avi)
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ts = None
    if args.csv:
        d = np.atleast_1d(np.genfromtxt(args.csv, delimiter=",", names=True))
        ts = np.array(sorted({round(float(r["t_s"]), 3) for r in d}))
        if len(ts) != n_total:
            print(f"[!] CSV-кадров {len(ts)} != AVI-кадров {n_total} — тайминг беру из видео (idx/30)")
            ts = None

    tag_ids = sorted(TAG_SIZE_MM)
    k_id = {i: k for k, i in enumerate(tag_ids)}
    kp_xy_arr = np.full((n_total, 17, 2), np.nan, np.float32)
    kp_cf_arr = np.zeros((n_total, 17), np.float32)
    person_seen = np.zeros(n_total, bool)
    tv_arr = np.full((n_total, len(tag_ids), 3), np.nan, np.float32)
    rv_arr = np.full((n_total, len(tag_ids), 3), np.nan, np.float32)
    seen_arr = np.zeros((n_total, len(tag_ids)), bool)

    out = args.avi.rsplit(".", 1)[0] + "_pairs.npz"

    def save(n_proc, final=False):
        """Промежуточный/финальный сейв (атомарно: tmp -> replace)."""
        t = (ts[:n_proc] if ts is not None
             else np.arange(n_proc) / 30.0)
        tmp = out + ".tmp.npz"
        np.savez_compressed(
            tmp,
            t=t, n_proc=n_proc, final=final,
            kpts_xy=kp_xy_arr[:n_proc], kpts_conf=kp_cf_arr[:n_proc],
            person_seen=person_seen[:n_proc],
            tag_tvec=tv_arr[:n_proc], tag_rvec=rv_arr[:n_proc],
            tag_seen=seen_arr[:n_proc],
            tag_ids=np.array(tag_ids), fx=FX, fy=FY, cx=CX, cy=CY,
        )
        for attempt in range(5):                   # Windows: лок (антивирус/индексатор)
            try:
                os.replace(tmp, out)
                break
            except PermissionError:
                if attempt == 4:
                    print("[!] не заменить чекпоинт (файл залочен) — "
                          "продолжаю, следующий чекпоинт попробует снова", flush=True)
                else:
                    time.sleep(1.0)

    t0, i = time.perf_counter(), 0
    if args.start > 0 and os.path.exists(out):     # resume: поднять чекпоинт
        with np.load(out) as ck:                   # хендл закрыть, иначе лок на replace
            n0 = int(ck["n_proc"])
            if n0 != args.start:
                print(f"[!] чекпоинт {n0} кадров != --start {args.start}: продолжаю с {n0}")
                args.start = n0
            kp_xy_arr[:n0] = ck["kpts_xy"]
            kp_cf_arr[:n0] = ck["kpts_conf"]
            if "person_seen" in ck:
                person_seen[:n0] = ck["person_seen"]
            tv_arr[:n0] = ck["tag_tvec"]
            rv_arr[:n0] = ck["tag_rvec"]
            seen_arr[:n0] = ck["tag_seen"]
        print(f"resume: загружено {n0} кадров из чекпоинта")
    while True:
        if i < args.start:                     # resume: прокрутить без декода
            if not cap.grab():
                break
            i += 1
            continue
        ok, frame = cap.read()
        if not ok or (args.max_frames and i >= args.max_frames):
            break
        res = yolo(frame, verbose=False, conf=args.conf)
        if res and res[0].keypoints is not None and len(res[0].keypoints.xy):
            xy = res[0].keypoints.xy[0].cpu().numpy()      # один оператор
            cc = (res[0].keypoints.conf[0].cpu().numpy()
                  if res[0].keypoints.conf is not None else np.ones(len(xy)))
            kp_xy_arr[i, :len(xy)] = xy
            kp_cf_arr[i, :len(cc)] = cc
            person_seen[i] = True

        if i % args.stride == 0:
            for tid, (tv, rv) in detect_tags(frame, det).items():
                tv_arr[i, k_id[tid]] = tv
                rv_arr[i, k_id[tid]] = rv
                seen_arr[i, k_id[tid]] = True

        i += 1
        done = i - args.start
        if done % 300 == 0:
            rate = done / (time.perf_counter() - t0)
            print(f"{i}/{n_total} кадров, {rate:.1f} fps, "
                  f"ETA {(n_total - i) / rate / 60:.0f} мин", flush=True)
        if done % args.checkpoint == 0:
            save(i)
            print(f"чекпоинт сохранён ({i} кадров)", flush=True)
    cap.release()

    save(i, final=True)
    dur = ((ts[i - 1] - ts[0]) if (ts is not None and i > 1)
           else (i - 1) / 30.0)
    print(f"\n{out}: {i} кадров ({dur:.0f} с), "
          f"YOLO-скелет в {100*person_seen[:i].mean():.0f}% кадров, теги: "
          + ", ".join(f"ID{j}={100*seen_arr[:i, k].mean():.0f}%"
                      for k, j in enumerate(tag_ids)))


if __name__ == "__main__":
    main()
