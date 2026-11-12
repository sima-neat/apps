"""Show how near each point of a scene is, from two cameras, with LightStereo-S.

A left and a right camera feed one source graph, which forwards each camera's video to its own
Insight channel. Each loop takes the newest frame of both, joins them side by side for the model,
and sends the colored disparity to a third channel. Disparity is how far a point shifts between
the two views: the larger it is, the nearer the point.
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "common" / "config.yaml"
SOURCE_TYPES = ("rtsp", "usb", "mipi")
RTSP_CODECS = ("h264", "h265", "mjpeg")
# The source graph's two outputs, in the order the model takes the views. Insight shows them on
# the first two channels and the depth map on the third.
VIEWS = ("left", "right")
DEPTH_CHANNEL = len(VIEWS)
SOURCE_TIMEOUT_MS = 20000
MODEL_TIMEOUT_MS = 10000
# The model preprocesses, infers and converts its output on different processors. Three frames
# in flight keep all of them busy; one frame at a time leaves two of them idle.
MODEL_FRAMES_IN_FLIGHT = 3

# Imported in main(), so the configuration loads and reports its errors without them.
cv2 = None
np = None
pyneat = None


@dataclass(frozen=True)
class AppConfig:
    model_path: str
    source_type: str
    cameras: tuple[str, str]  # The left and right camera: RTSP URLs, USB IDs or MIPI names.
    codec: str
    tcp: bool
    latency_ms: int
    width: int
    height: int
    fps: int
    frames: int
    max_disparity: float
    save_dir: str
    save_every: int
    insight_host: str
    video_port: int
    bitrate_kbps: int


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="YAML configuration")
    return parser.parse_args(argv)


def section(raw: dict, key: str) -> dict:
    value = raw.get(key) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping")
    return value


def setting(raw: dict, key: str, default):
    """raw[key], or the default when it is missing. The value has the default's type."""
    value = raw.get(key)
    if value is None:
        return default
    if isinstance(default, float) and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    if type(value) is not type(default):
        raise ValueError(f"{key} must be of type {type(default).__name__}")
    return value


def choice(value: str, choices: tuple[str, ...], key: str) -> str:
    if value.lower() not in choices:
        raise ValueError(f"{key} must be {', '.join(choices[:-1])} or {choices[-1]}")
    return value.lower()


def validate_config(cfg: AppConfig) -> None:
    source = f"source.{cfg.source_type}"
    camera_key = "url" if cfg.source_type == "rtsp" else "camera"
    if not cfg.model_path:
        raise ValueError("model.path must be set")
    if not all(cfg.cameras):
        raise ValueError(f"{source}.left_{camera_key} and {source}.right_{camera_key} must be set")
    if cfg.cameras[0] == cfg.cameras[1]:
        raise ValueError(f"{source}.left_{camera_key} and {source}.right_{camera_key} must differ")
    if cfg.source_type != "rtsp" and min(cfg.width, cfg.height, cfg.fps) <= 0:
        raise ValueError(f"{source}.width, {source}.height and {source}.fps must be > 0")
    if cfg.latency_ms < 0:
        raise ValueError("source.rtsp.latency_ms must be >= 0")
    if cfg.fps < 0:
        raise ValueError(f"{source}.fps must be >= 0")
    if cfg.frames < 0:
        raise ValueError("inference.frames must be >= 0")
    if cfg.max_disparity <= 0:
        raise ValueError("output.max_disparity must be > 0")
    if cfg.save_every < 0:
        raise ValueError("output.save_every must be >= 0")
    if not cfg.insight_host:
        raise ValueError("output.insight.host must be set")
    if not 0 < cfg.video_port <= 65535 - DEPTH_CHANNEL:
        raise ValueError(f"output.insight.video_port must be in [1, {65535 - DEPTH_CHANNEL}]")
    if cfg.bitrate_kbps <= 0:
        raise ValueError("output.insight.bitrate_kbps must be > 0")


def load_app_config(path: Path) -> AppConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("config root must be a mapping")
    source = section(raw, "source")
    output = section(raw, "output")
    insight = section(output, "insight")

    # Only the section that source.type names is read; the others may hold placeholders.
    source_type = choice(setting(source, "type", "rtsp"), SOURCE_TYPES, "source.type")
    selected = section(source, source_type)
    camera_key = "url" if source_type == "rtsp" else "camera"
    cfg = AppConfig(
        model_path=setting(section(raw, "model"), "path", ""),
        source_type=source_type,
        cameras=(
            setting(selected, f"left_{camera_key}", ""),
            setting(selected, f"right_{camera_key}", ""),
        ),
        codec=choice(setting(selected, "codec", "h264"), RTSP_CODECS, "source.rtsp.codec"),
        tcp=setting(selected, "tcp", True),
        latency_ms=setting(selected, "latency_ms", 100),
        width=setting(selected, "width", 0),
        height=setting(selected, "height", 0),
        fps=setting(selected, "fps", 0),
        frames=setting(section(raw, "inference"), "frames", 0),
        max_disparity=setting(output, "max_disparity", 96.0),
        save_dir=setting(output, "save_dir", ""),
        save_every=setting(output, "save_every", 0),
        insight_host=setting(insight, "host", ""),
        video_port=setting(insight, "video_port", 9000),
        bitrate_kbps=setting(insight, "bitrate_kbps", 8000),
    )
    validate_config(cfg)
    return cfg


def stream_fps(cfg: AppConfig) -> int:
    """The configured frame rate, or, when it is 0, the one the left RTSP stream reports. The
    encoder needs it before the first frame."""
    if cfg.fps > 0:
        return cfg.fps
    capture = cv2.VideoCapture(cfg.cameras[0])
    fps = round(capture.get(cv2.CAP_PROP_FPS) or 0) if capture.isOpened() else 0
    capture.release()
    if fps <= 0:
        raise RuntimeError(
            "could not read the frame rate of source.rtsp.left_url; set source.rtsp.fps"
        )
    return fps


def rtsp_codec(cfg: AppConfig):
    return {
        "h264": pyneat.RtspCodec.H264,
        "h265": pyneat.RtspCodec.H265,
        "mjpeg": pyneat.RtspCodec.MJPEG,
    }[cfg.codec]


def rtsp_source_options(cfg: AppConfig, url: str, fps: int):
    """Receives a camera's encoded stream without decoding it."""
    options = pyneat.RtspEncodedInputOptions()
    options.url = url
    options.tcp = cfg.tcp
    options.latency_ms = cfg.latency_ms
    options.source_fps = fps
    options.codec = rtsp_codec(cfg)
    options.auto_caps_from_stream = cfg.codec != "mjpeg"
    return options


def rtsp_decode_options(cfg: AppConfig, fps: int):
    options = pyneat.SimaDecodeOptions()
    options.type = {
        "h264": pyneat.SimaDecodeType.H264,
        "h265": pyneat.SimaDecodeType.H265,
        "mjpeg": pyneat.SimaDecodeType.MJPEG,
    }[cfg.codec]
    options.raw_output = True
    options.dec_fps = fps
    return options


def usb_camera_options(cfg: AppConfig, camera_id: str):
    options = pyneat.UsbCameraInputOptions()
    options.camera_name = camera_id
    options.width = cfg.width
    options.height = cfg.height
    options.framerate_num = cfg.fps
    return options


def usb_decode_options(cfg: AppConfig, decoded_format):
    options = pyneat.SimaDecodeOptions()
    options.type = pyneat.SimaDecodeType.MJPEG
    options.raw_output = True
    options.out_format = decoded_format
    options.dec_width = cfg.width
    options.dec_height = cfg.height
    options.dec_fps = cfg.fps
    return options


def mipi_camera_options(cfg: AppConfig, camera_name: str):
    options = pyneat.CameraInputOptions()
    options.camera_name = camera_name
    options.width = cfg.width
    options.height = cfg.height
    options.framerate_num = cfg.fps
    options.format = "NV12"
    return options


def camera_graphs(cfg: AppConfig, view: str, camera_name: str, fps: int):
    """One camera as two graphs: the camera's video as it arrives, and the decoder that turns it
    into frames. A MIPI camera sends raw frames, so its second graph is empty."""
    camera = pyneat.Graph(f"{view}_camera")
    decoder = pyneat.Graph(f"{view}_decoder")
    if cfg.source_type == "rtsp":
        camera.add(pyneat.groups.rtsp_encoded_input(rtsp_source_options(cfg, camera_name, fps)))
        decoder.add(pyneat.nodes.sima_decode(rtsp_decode_options(cfg, fps)))
    elif cfg.source_type == "usb":
        usb_camera = usb_camera_options(cfg, camera_name)
        # Checks the camera and its mode once; most webcams send 4:2:2 JPEG, which decodes to NV16.
        info = pyneat.usb_camera_info(usb_camera)
        camera.add(pyneat.nodes.usb_camera_input(usb_camera, info))
        decoder.add(pyneat.nodes.sima_decode(usb_decode_options(cfg, info.decoded_format)))
    else:
        camera.add(pyneat.nodes.camera_input(mipi_camera_options(cfg, camera_name)))
    return camera, decoder


def h264_encoding_sender_options(cfg: AppConfig, fps: int):
    """H.264 encoding of raw frames for Insight."""
    encode = pyneat.SimaEncodeOptions()
    encode.fps = fps
    options = pyneat.VideoSenderOptions.from_raw(encode)
    options.encoder.bitrate_kbps = cfg.bitrate_kbps
    return options


def camera_sender_options(cfg: AppConfig, channel: int, fps: int):
    """Sends a camera's video to its Insight channel. RTSP and USB cameras send encoded video,
    which is forwarded as it is; raw MIPI frames are encoded first."""
    if cfg.source_type == "rtsp":
        options = pyneat.VideoSenderOptions.passthrough(rtsp_codec(cfg))
    elif cfg.source_type == "usb":
        options = pyneat.VideoSenderOptions.passthrough(pyneat.RtspCodec.MJPEG)
    else:
        options = h264_encoding_sender_options(cfg, fps)
    options.host = cfg.insight_host
    options.video_port_base = cfg.video_port
    options.channel = channel
    return options


def build_source_run(cfg: AppConfig, fps: int):
    """One graph with two branches per camera::

        left camera  --+-- Insight channel 0
                       +-- decode -- Output("left")
        right camera --+-- Insight channel 1
                       +-- decode -- Output("right")

    Each output keeps only its newest frame, so a pull never returns a frame that waited.
    """
    graph = pyneat.Graph("source")
    # Insight's branch keeps the newest video only, so a slow network never delays the decoder.
    newest_video = pyneat.GraphLinkOptions()
    newest_video.policy = pyneat.GraphLinkPolicy.RealtimeLatestByStream
    for channel, (view, camera_name) in enumerate(zip(VIEWS, cfg.cameras)):
        camera, decoder = camera_graphs(cfg, view, camera_name, fps)
        decoder.add(pyneat.nodes.output(view, pyneat.OutputOptions.latest()))
        video = pyneat.Graph(f"{view}_video")
        video.add(pyneat.groups.video_sender(camera_sender_options(cfg, channel, fps)))
        graph.connect(camera, decoder)
        graph.connect(camera, video, newest_video)

    run_options = pyneat.RunOptions()
    run_options.preset = pyneat.RunPreset.Realtime
    run_options.queue_depth = 1
    run_options.overflow_policy = pyneat.OverflowPolicy.KeepLatest
    run_options.output_memory = pyneat.OutputMemory.ZeroCopy
    return graph, graph.build(run_options)


def frame_bgr(tensor) -> np.ndarray:
    """A decoded NV12 or NV16 frame as BGR pixels the application owns.

    Decoders pad rows and planes, so each plane is read at its own offset and stride. MJPEG with
    4:2:2 chroma, as most webcams send, decodes to NV16; every other chroma row of it makes NV12.
    """
    width, height = tensor.width(), tensor.height()
    payload = np.frombuffer(tensor.copy_payload_bytes(), np.uint8)
    planes = {plane.role: plane for plane in tensor.planes}
    nv16 = tensor.image_format() == pyneat.PixelFormat.NV16
    if not nv16 and not tensor.is_nv12():
        raise RuntimeError(f"unsupported decoded frame format {tensor.image_format()}")

    def plane(role, rows):
        layout = planes[role]
        return np.ndarray(
            (rows, width),
            np.uint8,
            buffer=payload,
            offset=layout.byte_offset,
            strides=(layout.strides_bytes[0], 1),
        )

    nv12 = np.empty((height * 3 // 2, width), np.uint8)
    nv12[:height] = plane(pyneat.PlaneRole.Y, height)
    nv12[height:] = (
        plane(pyneat.PlaneRole.UV, height)[::2] if nv16 else plane(pyneat.PlaneRole.UV, height // 2)
    )
    return cv2.cvtColor(nv12, cv2.COLOR_YUV2BGR_NV12)


def sample_tensor(sample):
    return sample.tensor if sample.kind == pyneat.SampleKind.Tensor else sample.tensors[0]


def pull_pair(source_run) -> tuple[np.ndarray, np.ndarray]:
    """The newest left and right frame. The cameras share no clock, so the two can be up to about
    one frame interval apart."""
    tensors = []
    for view in VIEWS:
        sample = source_run.pull(view, SOURCE_TIMEOUT_MS)
        if sample is None:
            raise RuntimeError(f"timed out waiting for a frame from the {view} camera")
        part = sample
        if sample.fields:
            part = next(field for field in sample.fields if field.stream_label == view)
        tensors.append(sample_tensor(part))
    left, right = (frame_bgr(tensor) for tensor in tensors)
    if left.shape != right.shape:
        raise RuntimeError(
            f"the cameras must send the same frame size: left is {left.shape[1]}x{left.shape[0]}, "
            f"right is {right.shape[1]}x{right.shape[0]}"
        )
    return left, right


def model_options(width: int, height: int):
    """LightStereo-S on a side-by-side BGR frame: Neat stretches it to the model input, so each
    half lands on one view, converts it to RGB and applies the ImageNet normalization."""
    options = pyneat.ModelOptions()
    options.preprocess.kind = pyneat.InputKind.Image
    options.preprocess.enable = pyneat.AutoFlag.On
    options.preprocess.input_max_width = width
    options.preprocess.input_max_height = height
    options.preprocess.resize.enable = pyneat.AutoFlag.On
    options.preprocess.resize.mode = pyneat.ResizeMode.Stretch
    options.preprocess.color_convert.enable = pyneat.AutoFlag.On
    options.preprocess.color_convert.input_format = pyneat.PreprocessColorFormat.BGR
    options.preprocess.color_convert.output_format = pyneat.PreprocessColorFormat.RGB
    options.preprocess.normalize.enable = pyneat.AutoFlag.On
    options.preprocess.preset = pyneat.NormalizePreset.ImageNet
    return options


def stereo_tensor(left: np.ndarray, right: np.ndarray):
    """The model input: the left view beside the right view, in one frame."""
    return pyneat.Tensor.from_numpy(
        np.hstack([left, right]),
        copy=True,
        image_format=pyneat.PixelFormat.BGR,
        memory=pyneat.TensorMemory.EV74,
    )


def disparity_heatmap(disparity: np.ndarray, max_disparity: float) -> np.ndarray:
    """The disparity map as a BGR image, from black at 0 to the brightest color at max_disparity
    and above. The scale is fixed, so a color means the same nearness in every frame."""
    levels = cv2.convertScaleAbs(disparity, alpha=255.0 / max_disparity)
    return cv2.applyColorMap(levels, cv2.COLORMAP_INFERNO)


def build_depth_run(cfg: AppConfig, width: int, height: int, fps: int):
    """Encodes the BGR depth map to H.264 and sends it to its Insight channel."""
    input_options = pyneat.InputOptions()
    input_options.payload_type = pyneat.PayloadType.Image
    input_options.format = pyneat.Format.BGR
    input_options.width = width
    input_options.height = height
    input_options.depth = 3
    input_options.fps_n = fps
    input_options.fps_d = 1

    sender_options = h264_encoding_sender_options(cfg, fps)
    sender_options.host = cfg.insight_host
    sender_options.video_port_base = cfg.video_port
    sender_options.channel = DEPTH_CHANNEL

    graph = pyneat.Graph("depth")
    graph.add(pyneat.nodes.input(input_options))
    graph.add(pyneat.groups.video_sender(sender_options))
    return graph, graph.build([heatmap_tensor(np.zeros((height, width, 3), np.uint8))])


def heatmap_tensor(heatmap: np.ndarray):
    return pyneat.Tensor.from_numpy(
        heatmap, copy=True, image_format=pyneat.PixelFormat.BGR, memory=pyneat.TensorMemory.CPU
    )


def push_pair(runner, left: np.ndarray, right: np.ndarray) -> None:
    """Starts the model on a pair. pull_disparity returns the results in the order of the pushes."""
    if not runner.push([stereo_tensor(left, right)]):
        raise RuntimeError("the model did not accept a frame")


def pull_disparity(runner) -> np.ndarray:
    """The oldest pair's result: one disparity per pixel of the left view, in pixels of the
    model's view."""
    sample = runner.pull(MODEL_TIMEOUT_MS)
    if sample is None:
        raise RuntimeError("timed out waiting for the model")
    return np.squeeze(sample_tensor(sample).to_numpy(copy=True))


def save_depth_map(path: Path, heatmap: np.ndarray) -> None:
    if not cv2.imwrite(str(path), heatmap):
        raise RuntimeError(f"could not save {path}")


def run(cfg: AppConfig, stopped: Callable[[], bool]) -> int:
    runs = []
    try:
        fps = stream_fps(cfg)
        _source_graph, source_run = build_source_run(cfg, fps)
        runs.append(source_run)
        left, right = pull_pair(source_run)
        height, width = left.shape[:2]
        print(f"source={cfg.source_type} cameras={width}x{height}@{fps}", flush=True)

        model = pyneat.Model(cfg.model_path, model_options(2 * width, height))
        runner = model.build([stereo_tensor(left, right)])
        runs.append(runner)
        # The first disparity map sets the size of the depth map Insight receives.
        push_pair(runner, left, right)
        map_height, map_width = pull_disparity(runner).shape
        _depth_graph, depth_run = build_depth_run(cfg, map_width, map_height, fps)
        runs.append(depth_run)

        frames = 0
        in_flight = 0
        started = time.monotonic()
        while not stopped() and (cfg.frames == 0 or frames < cfg.frames):
            push_pair(runner, *pull_pair(source_run))
            in_flight += 1
            if in_flight < MODEL_FRAMES_IN_FLIGHT:
                continue
            heatmap = disparity_heatmap(pull_disparity(runner), cfg.max_disparity)
            in_flight -= 1
            if not depth_run.push([heatmap_tensor(heatmap)]):
                raise RuntimeError("could not send the depth map to Insight")
            if cfg.save_dir and cfg.save_every and frames % cfg.save_every == 0:
                save_depth_map(Path(cfg.save_dir) / f"depth_{frames:06d}.jpg", heatmap)
            frames += 1

        depth_fps = frames / (time.monotonic() - started) if frames else 0.0
        print(
            f"frames={frames} depth_fps={depth_fps:.1f} "
            f"insight={cfg.insight_host}:{cfg.video_port}-{cfg.video_port + DEPTH_CHANNEL}",
            flush=True,
        )
        return 0
    finally:
        for started_run in reversed(runs):
            started_run.close()


def main(argv: list[str] | None = None) -> int:
    global cv2, np, pyneat

    args = parse_args(argv)
    try:
        cfg = load_app_config(args.config)
    except (OSError, ValueError, yaml.YAMLError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    if not Path(cfg.model_path).is_file():
        print(f"Error: model does not exist: {cfg.model_path}", file=sys.stderr)
        return 2
    if cfg.save_dir:
        Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)

    import cv2
    import numpy as np
    import pyneat

    stop_requested = False

    def request_stop(_signum, _frame) -> None:
        nonlocal stop_requested
        stop_requested = True

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, request_stop)
    try:
        return run(cfg, lambda: stop_requested)
    except Exception as error:  # noqa: BLE001
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
