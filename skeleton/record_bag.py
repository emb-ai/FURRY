"""
Запись .bag с D435i: color 1280x720@30 + depth (выровнены при воспроизведении).

Сценарий записи показывается оверлеем в окне превью:
  0-5 c   GET READY (встань в 2-3 м, целиком в кадр)
  5-20 c  STATIC    (стой ровно, руки вдоль тела)
  20-25 c T-POSE    (руки горизонтально)
  25-35 c FREE      (свободное движение: шаги, присед, махи)

Запуск: .venv/Scripts/python record_bag.py [длительность_сек] [путь.bag]
"""
import os
import sys
import time

import cv2
import numpy as np
import pyrealsense2 as rs

DUR = float(sys.argv[1]) if len(sys.argv) > 1 else 35.0
PATH = sys.argv[2] if len(sys.argv) > 2 else os.path.join("data", f"rec_{time.strftime('%H%M%S')}.bag")
os.makedirs(os.path.dirname(PATH), exist_ok=True)


def phase(t):
    if t < 5:
        return f"GET READY {5 - t:.0f}...", (0, 200, 255)
    if t < 20:
        return "STATIC (stand still)", (0, 255, 0)
    if t < 25:
        return "T-POSE", (255, 200, 0)
    return "FREE (move!)", (255, 120, 0)


cfg = rs.config()
cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
cfg.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
cfg.enable_record_to_file(PATH)

pipe = rs.pipeline()
profile = pipe.start(cfg)
n = 0
t0 = time.monotonic()
try:
    while True:
        fs = pipe.wait_for_frames(timeout_ms=5000)
        color = fs.get_color_frame()
        depth = fs.get_depth_frame()  # ОБЯЗАТЕЛЬНО забирать — иначе в .db3 запишутся нули
        if not color or not depth:
            continue
        n += 1
        t = time.monotonic() - t0
        if t >= DUR:
            break
        img = np.asanyarray(color.get_data())
        txt, col = phase(t)
        cv2.rectangle(img, (0, 0), (720, 90), (0, 0, 0), -1)
        cv2.putText(img, txt, (20, 45), 6, 1.1, col, 2)
        cv2.putText(img, f"REC {t:5.1f}/{DUR:.0f} s   frames {n}", (20, 75), 6, 0.7, (255, 255, 255), 1)
        cv2.imshow("REC", img)
        if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
            break
finally:
    pipe.stop()
    cv2.destroyAllWindows()

size_mb = os.path.getsize(PATH) / 1e6
print(f"Записано: {PATH}\nКадров: {n}, длительность {DUR:.0f} c, размер {size_mb:.0f} МБ", flush=True)
