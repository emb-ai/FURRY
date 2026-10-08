"""Release partially opened RealSense streams before retrying."""
def open_pipeline(rs, config):
    pipeline=rs.pipeline()
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
