"""Проверка RealSense D435i: перечисление устройства + захват кадров."""
import numpy as np
import pyrealsense2 as rs

# 1) Перечисление устройств
ctx = rs.context()
devices = ctx.query_devices()
print(f"Найдено устройств: {len(devices)}")
for dev in devices:
    print(f"  Модель:   {dev.get_info(rs.camera_info.name)}")
    print(f"  Серийник: {dev.get_info(rs.camera_info.serial_number)}")
    print(f"  FW:       {dev.get_info(rs.camera_info.firmware_version)}")
    print(f"  USB:      {dev.get_info(rs.camera_info.usb_type_descriptor)}")

# 2) Захват 30 кадров, проверка цвет+глубина
pipeline = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
cfg.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
profile = pipeline.start(cfg)

depth_sensor = profile.get_device().first_depth_sensor()
print(f"\nDepth scale: {depth_sensor.get_depth_scale():.6f} м/единица")

n = 0
depth_valid_pct = []
try:
    while n < 30:
        fs = pipeline.wait_for_frames(timeout_ms=5000)
        color = fs.get_color_frame()
        depth = fs.get_depth_frame()
        if not color or not depth:
            continue
        n += 1
        if n % 10 == 0:
            d = np.asanyarray(depth.get_data()).astype(np.float32)
            valid = (d > 0) & (d < 10_000)  # мм, < 10 м
            depth_valid_pct.append(100.0 * valid.mean())
            print(f"Кадр {n}: color {color.get_width()}x{color.get_height()}, "
                  f"depth валиден {depth_valid_pct[-1]:.1f}%, "
                  f"медиана глубины {np.median(d[valid]) / 1000:.2f} м")
finally:
    pipeline.stop()

print(f"\nИтог: {n} кадров получено. Глубина валидна в среднем на "
      f"{np.mean(depth_valid_pct):.1f}% пикселей. Камера работает.")
