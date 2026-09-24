"""Settings for the existing high-density E2E test, independent of hardware."""

import os


def qualification_options():
    prefix = "SIMANEAT_APPS_TEST_HD_"
    def number(name, default, minimum, maximum):
        raw = os.environ.get(prefix + name, str(default))
        value = int(raw)
        if not minimum <= value <= maximum:
            raise ValueError(f"{prefix}{name} must be in [{minimum}, {maximum}], got {raw}")
        return value
    streams = number("STREAMS", 16, 1, 80)
    fps = number("SOURCE_FPS", 30, 1, 120)
    frames = number("MEASURE_FRAMES", 5000, streams, 100000000)
    profile = os.environ.get(prefix + "PROFILE", "config.yaml")
    if profile not in ("config.yaml", "config-24x720p20fps.yaml", "config-48x720p10fps.yaml"):
        raise ValueError(f"{prefix}PROFILE must name a shipped density profile")
    overrides = {}
    for name, key in (("DECODER_BUFFERS", "decoder_buffers"),
                      ("INPUT_BUFFERS", "decoder_input_buffers")):
        if prefix + name in os.environ:
            overrides[key] = number(name, 1, 1, 64 if name == "DECODER_BUFFERS" else 2147483647)
    return streams, fps, frames, profile, overrides
