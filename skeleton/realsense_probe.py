"""Проба доступных стрим-конфигураций на текущем USB-подключении."""
import pyrealsense2 as rs

combos = [
    ("color only 640x480@30",  [(rs.stream.color, 640, 480, rs.format.bgr8, 30)]),
    ("color only 1280x720@30", [(rs.stream.color, 1280, 720, rs.format.bgr8, 30)]),
    ("depth only 640x480@30",  [(rs.stream.depth, 640, 480, rs.format.z16, 30)]),
    ("depth only 848x480@30",  [(rs.stream.depth, 848, 480, rs.format.z16, 30)]),
    ("both 640x480@15",        [(rs.stream.color, 640, 480, rs.format.bgr8, 15),
                                (rs.stream.depth, 640, 480, rs.format.z16, 15)]),
]
for name, streams in combos:
    try:
        pipe = rs.pipeline()
        cfg = rs.config()
        for s in streams:
            cfg.enable_stream(*s)
        profile = pipe.start(cfg)
        fs = pipe.wait_for_frames(timeout_ms=4000)
        pipe.stop()
        print(f"OK   {name}")
    except Exception as e:
        print(f"FAIL {name}: {e}")
