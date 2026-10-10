"""Release partially opened RealSense streams before retrying."""
# Кадров в архиве сенсора: по умолчанию 16. При 6-15 fps обработки длинная
# очередь отдаёт старые кадры (эпизод 10.10: ~120 мс скрытого лага), короткая
# — дропает лишние, нам нужен самый свежий.
FRAMES_QUEUE_SIZE = 2


def open_pipeline(rs, config):
    pipeline=rs.pipeline()
    # Размер очереди читается при старте стрима, поэтому до start().
    try:
        device=config.resolve(rs.pipeline_wrapper(pipeline)).get_device()
        for sensor in device.query_sensors():
            if sensor.supports(rs.option.frames_queue_size):
                sensor.set_option(rs.option.frames_queue_size,FRAMES_QUEUE_SIZE)
    except Exception as exc:
        print(f'Frame queue size unchanged: {exc}',flush=True)
    try:
        profile=pipeline.start(config)
    except BaseException:
        try:pipeline.stop()
        except Exception:pass
        raise
    # Failure of optional clock configuration must not abandon an open stream.
    for sensor in profile.get_device().query_sensors():
        try:
            if sensor.supports(rs.option.global_time_enabled):
                sensor.set_option(rs.option.global_time_enabled,1)
        except Exception as exc:
            print(f'Global camera clock unavailable: {exc}; using timestamp fallback',flush=True)
    return pipeline,profile


def latest_frames(pipeline, timeout_ms=5000):
    """Блокирующе ждёт кадр, затем забирает всё, что уже накопилось: старые
    framesets выбрасываются, чтобы обработка шла по самому свежему кадру."""
    frames=pipeline.wait_for_frames(timeout_ms=timeout_ms)
    dropped=0
    while True:
        newer=pipeline.poll_for_frames()
        if not newer:
            return frames,dropped
        frames=newer
        dropped+=1
