"""Match a live stream's SuperPoint keypoints to a reference frame with LightGlue or SuperGlue.

The first frame with 512 keypoints becomes the reference. Every later frame runs SuperPoint, the
selected matcher matches its keypoints to the reference's, and Insight receives the reference on
the left and the live frame on the right, with one line per match colored by its score.
"""

from __future__ import annotations

import argparse
import collections
import os
import queue
import signal
import sys
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "common" / "config.yaml"
# Both matchers take exactly 512 keypoints per image and have no padding mask.
KEYPOINTS = 512
# SuperPoint's input size, which both matchers' keypoint normalization assumes.
MODEL_INPUT_SIZE = (640, 480)
SOURCE_TIMEOUT_MS = 20000
MODEL_TIMEOUT_MS = 10000
# Frames each model runner may hold, so that its stages overlap: on the board, SuperPoint
# alone runs at 11.6 ms per frame one at a time and 7.4 ms with two frames in flight.
RUNNER_QUEUE_DEPTH = 2
# Frames waiting between two stages; a full queue holds back the stage before it.
STAGE_QUEUE_DEPTH = 2
# Frames matched and drawn at once, each on its own worker. On the board one frame takes up to
# about 50 ms of CPU: drawing 430 anti-aliased lines, or SuperGlue's Sinkhorn iterations plus
# drawing. OpenCV and numpy run without holding the GIL, so frames proceed in parallel: a 30 fps
# stream with 430 matches per frame is sent at 28 fps this way, and at 13 fps one at a time.
RENDER_WORKERS = 3
# The summary's inlier ratio samples every tenth frame: RANSAC takes about 18 ms.
INLIER_SAMPLE_INTERVAL = 10
# Frames the displayed rate is measured over.
FPS_WINDOW = 30
# LightGlue's learned keypoint projection (positional_encoder.projector.weight, [32, 2]) in
# ETH-CVG/lightglue_superpoint revision 5f5f626e, Apache-2.0. Each row turns a normalized
# keypoint into one rotary phase.
LIGHTGLUE_PROJECTION = (
    (0.07869736105203629, 0.6576962471008301),
    (0.30121079087257385, -1.3067657947540283),
    (-0.17319072782993317, 0.3640412390232086),
    (-1.8221570253372192, 0.07125379890203476),
    (4.831599235534668, 0.024766402319073677),
    (-0.6115880608558655, -0.5418711304664612),
    (0.5497006177902222, -0.7080969214439392),
    (0.9728097915649414, 0.8288111686706543),
    (-0.18647703528404236, 0.2699703872203827),
    (1.3076542615890503, -1.4809051752090454),
    (-0.19871585071086884, 2.5542619228363037),
    (-1.342088222503662, -1.5769871473312378),
    (-0.023420853540301323, -0.2613159120082855),
    (-0.5988006591796875, -4.0408616065979),
    (0.4600417912006378, -0.07377243041992188),
    (-2.12711763381958, -1.0835118293762207),
    (-0.6925885677337646, 1.3105987310409546),
    (0.25808796286582947, -0.06581223011016846),
    (0.09479046612977982, -0.44248539209365845),
    (-1.2112597227096558, -1.857155680656433),
    (-0.24696263670921326, -0.0016804899787530303),
    (-2.5080809593200684, 3.7625091075897217),
    (-1.390500783920288, 0.7762705087661743),
    (0.3077039122581482, -1.0518392324447632),
    (-1.6019573211669922, -0.010330487042665482),
    (0.5500544309616089, 2.5014572143554688),
    (-1.0771361589431763, 0.19658993184566498),
    (0.25206997990608215, -0.5227173566818237),
    (-0.5797532200813293, 0.07100092619657516),
    (-0.36304882168769836, 1.7311886548995972),
    (0.554409921169281, 1.0818008184432983),
    (0.4458959400653839, 0.6291354298591614),
)
SOURCE_TYPES = ("rtsp", "http", "usb", "mipi")
MATCHER_TYPES = ("lightglue", "superglue")
RTSP_CODECS = ("h264", "h265", "mjpeg")
# Composed frames this large or larger need H.264 level 5.1; smaller ones fit level 4.0.
LEVEL_4_MAX_MACROBLOCKS = 8192

cv2 = None
np = None
pyneat = None


@dataclass(frozen=True)
class AppConfig:
    superpoint_path: str
    matcher_type: str
    matcher_path: str
    match_threshold: float
    sinkhorn_iterations: int
    source_type: str
    # The key that names the source in errors: source.<type>.url or source.<type>.camera.
    source_key: str
    source_url: str
    source_codec: str
    tcp: bool
    latency_ms: int
    ssl_strict: bool
    # source.usb and source.mipi: the camera and the exact mode it captures.
    camera: str
    source_width: int
    source_height: int
    # 0 probes the stream; USB and MIPI cameras take it from their mode.
    source_fps: int
    frames: int
    insight_host: str
    video_port: int
    bitrate_kbps: int
    save_dir: str
    save_every: int


@dataclass(frozen=True)
class Features:
    """One frame's 512 SuperPoint keypoints, in pixels of the model's input."""

    keypoints: object  # float32 [512, 2], (x, y)
    scores: object  # float32 [512]
    descriptors: object  # float32 [512, 256], unit length


@dataclass(frozen=True)
class Matches:
    """Keypoint pairs, as indices into the reference's and the live frame's keypoints."""

    reference: object  # int [M]
    live: object  # int [M]
    scores: object  # float32 [M]


@dataclass(frozen=True)
class Matcher:
    """The selected matcher's CPU steps around its model."""

    name: str
    # One image's model inputs from its features, in the model's input order.
    image_inputs: Callable[[Features], list]
    # The matches in the model's output scores.
    match: Callable[[object], Matches]


@dataclass(frozen=True)
class Frame:
    """A frame's BGR pixels and SuperPoint features; `features` is None when it has fewer than
    512 keypoints."""

    bgr: object
    features: Features | None


@dataclass(frozen=True)
class StageLinks:
    """The queues and events the three stages share."""

    captured: queue.Queue  # stage 1 to 2: each frame's BGR pixels
    matched: queue.Queue  # stage 2 to 3: each Frame, then None after the last
    failed: threading.Event  # a stage raised; the others stop
    captured_all: threading.Event  # stage 1 pushed its last frame


class Rendered(NamedTuple):
    """A composed frame, its match count (None when not matched) and its sampled inlier ratio."""

    canvas: object
    match_count: int | None
    inlier_ratio: float | None


@dataclass
class MatchStats:
    """What the publisher sent, for the summary line and the displayed rate."""

    frames: int = 0
    matched_frames: int = 0
    total_matches: int = 0
    inlier_sum: float = 0.0
    inlier_samples: int = 0
    first_sent_s: float = 0.0
    last_sent_s: float = 0.0
    recent_sent_s: collections.deque = field(
        default_factory=lambda: collections.deque(maxlen=FPS_WINDOW)
    )

    def record(self, rendered: Rendered) -> None:
        now = time.perf_counter()
        if not self.frames:
            self.first_sent_s = now
        self.frames += 1
        self.last_sent_s = now
        self.recent_sent_s.append(now)
        if rendered.match_count is not None:
            self.matched_frames += 1
            self.total_matches += rendered.match_count
        if rendered.inlier_ratio is not None:
            self.inlier_sum += rendered.inlier_ratio
            self.inlier_samples += 1

    def recent_fps(self) -> float:
        times = self.recent_sent_s
        return (len(times) - 1) / (times[-1] - times[0]) if len(times) > 1 else 0.0

    def summary(self) -> str:
        average = self.total_matches / self.matched_frames if self.matched_frames else 0.0
        inliers = self.inlier_sum / self.inlier_samples if self.inlier_samples else 0.0
        elapsed = self.last_sent_s - self.first_sent_s
        fps = (self.frames - 1) / elapsed if elapsed > 0 else 0.0
        return (
            f"frames={self.frames} matched={self.matched_frames} "
            f"skipped={self.frames - self.matched_frames} average_matches={average:.1f} "
            f"inlier_ratio={inliers:.2f} output_fps={fps:.1f}"
        )


def load_runtime_dependencies() -> None:
    global cv2, np, pyneat
    # The app runs frames in parallel on its own threads, so OpenCV and the OpenBLAS it bundles
    # for cv2.gemm each use one thread. Their idle workers otherwise spin at full CPU between
    # calls and take every core of the board.
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    import cv2 as cv2_module
    import numpy as np_module
    import pyneat as pyneat_module

    cv2, np, pyneat = cv2_module, np_module, pyneat_module
    cv2.setNumThreads(1)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="YAML configuration")
    return parser.parse_args(argv)


def section(raw: dict, key: str) -> dict:
    value = raw.get(key) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping")
    return value


def string_or(raw: dict, key: str, default: str = "") -> str:
    value = raw.get(key, default)
    if value is None:
        return default
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    return value


def int_or(raw: dict, key: str, default: int) -> int:
    value = raw.get(key, default)
    if value is None:
        return default
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{key} must be an integer")
    return value


def float_or(raw: dict, key: str, default: float) -> float:
    value = raw.get(key, default)
    if value is None:
        return default
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{key} must be numeric")
    return float(value)


def bool_or(raw: dict, key: str, default: bool) -> bool:
    value = raw.get(key, default)
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be true or false")
    return value


def choice(value: str, choices: tuple[str, ...], key: str) -> str:
    lowered = value.lower()
    if lowered not in choices:
        raise ValueError(f"{key} must be {', '.join(choices[:-1])} or {choices[-1]}")
    return lowered


def validate_config(cfg: AppConfig) -> None:
    if not cfg.superpoint_path:
        raise ValueError("superpoint.path must be set")
    if not cfg.matcher_path:
        raise ValueError(f"matcher.{cfg.matcher_type}.path must be set")
    if not 0.0 <= cfg.match_threshold < 1.0:
        raise ValueError(f"matcher.{cfg.matcher_type}.match_threshold must be in [0, 1)")
    if cfg.sinkhorn_iterations <= 0:
        raise ValueError("matcher.superglue.sinkhorn_iterations must be > 0")
    if cfg.source_type in ("rtsp", "http") and not cfg.source_url:
        raise ValueError(f"{cfg.source_key} must be set")
    if cfg.source_type == "usb" and not cfg.camera:
        raise ValueError("source.usb.camera must be set")
    if cfg.source_type in ("usb", "mipi"):
        prefix = f"source.{cfg.source_type}"
        if cfg.source_width <= 0 or cfg.source_height <= 0 or cfg.source_fps <= 0:
            raise ValueError(f"{prefix}.width, {prefix}.height and {prefix}.fps must be > 0")
    if cfg.latency_ms < 0:
        raise ValueError("source.rtsp.latency_ms must be >= 0")
    if cfg.source_fps < 0:
        raise ValueError(f"source.{cfg.source_type}.fps must be >= 0")
    if cfg.frames < 0:
        raise ValueError("inference.frames must be >= 0")
    if not cfg.insight_host:
        raise ValueError("output.insight.host must be set")
    if not 0 < cfg.video_port <= 65535:
        raise ValueError("output.insight.video_port must be in [1, 65535]")
    if cfg.bitrate_kbps <= 0:
        raise ValueError("output.insight.bitrate_kbps must be > 0")
    if cfg.save_every < 0:
        raise ValueError("output.save_every must be >= 0")


def load_app_config(path: Path) -> AppConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("config root must be a mapping")
    superpoint = section(raw, "superpoint")
    matcher = section(raw, "matcher")
    source = section(raw, "source")
    inference = section(raw, "inference")
    output = section(raw, "output")
    insight = section(output, "insight")

    # Only the sections that source.type and matcher.type name are read; the others may hold
    # placeholders.
    matcher_type = choice(string_or(matcher, "type", "lightglue"), MATCHER_TYPES, "matcher.type")
    selected_matcher = section(matcher, matcher_type)
    source_type = choice(string_or(source, "type", "rtsp"), SOURCE_TYPES, "source.type")
    selected_source = section(source, source_type)
    camera_source = source_type in ("usb", "mipi")
    codec = string_or(selected_source, "codec", "h264") if source_type == "rtsp" else "mjpeg"

    cfg = AppConfig(
        superpoint_path=string_or(superpoint, "path"),
        matcher_type=matcher_type,
        matcher_path=string_or(selected_matcher, "path"),
        match_threshold=float_or(
            selected_matcher, "match_threshold", 0.1 if matcher_type == "lightglue" else 0.2
        ),
        sinkhorn_iterations=int_or(selected_matcher, "sinkhorn_iterations", 20),
        source_type=source_type,
        source_key=f"source.{source_type}.{'camera' if camera_source else 'url'}",
        source_url=string_or(selected_source, "url"),
        source_codec=choice(codec, RTSP_CODECS, "source.rtsp.codec"),
        tcp=bool_or(selected_source, "tcp", True),
        latency_ms=int_or(selected_source, "latency_ms", 100),
        ssl_strict=bool_or(selected_source, "ssl_strict", True),
        camera=string_or(selected_source, "camera"),
        source_width=int_or(selected_source, "width", 0),
        source_height=int_or(selected_source, "height", 0),
        source_fps=int_or(selected_source, "fps", 0),
        frames=int_or(inference, "frames", 0),
        insight_host=string_or(insight, "host"),
        video_port=int_or(insight, "video_port", 9000),
        bitrate_kbps=int_or(insight, "bitrate_kbps", 8000),
        save_dir=string_or(output, "save_dir"),
        save_every=int_or(output, "save_every", 0),
    )
    validate_config(cfg)
    return cfg


def stream_fps(cfg: AppConfig) -> int:
    """The configured frame rate, or, when it is 0, the one an RTSP or HTTP stream reports. The
    encoder needs it before the first frame."""
    if cfg.source_fps > 0:
        return cfg.source_fps
    capture = cv2.VideoCapture(cfg.source_url)
    fps = round(capture.get(cv2.CAP_PROP_FPS) or 0) if capture.isOpened() else 0
    capture.release()
    if fps <= 0:
        raise RuntimeError(
            f"could not read the frame rate of {cfg.source_key}; set source.{cfg.source_type}.fps"
        )
    return fps


def rtsp_source_options(cfg: AppConfig, fps: int):
    options = pyneat.RtspDecodedInputOptions()
    options.url = cfg.source_url
    options.tcp = cfg.tcp
    options.latency_ms = cfg.latency_ms
    options.source_fps = fps
    options.codec = {
        "h264": pyneat.RtspCodec.H264,
        "h265": pyneat.RtspCodec.H265,
        "mjpeg": pyneat.RtspCodec.MJPEG,
    }[cfg.source_codec]
    options.auto_caps_from_stream = cfg.source_codec != "mjpeg"
    return options


def http_source_options(cfg: AppConfig, fps: int):
    options = pyneat.HttpMjpegDecodedInputOptions()
    options.url = cfg.source_url
    options.ssl_strict = cfg.ssl_strict
    options.source_fps = fps
    return options


def usb_camera_options(cfg: AppConfig):
    options = pyneat.UsbCameraInputOptions()
    options.camera_name = cfg.camera
    options.width = cfg.source_width
    options.height = cfg.source_height
    options.framerate_num = cfg.source_fps
    return options


def usb_decode_options(cfg: AppConfig, decoded_format):
    options = pyneat.SimaDecodeOptions()
    options.type = pyneat.SimaDecodeType.MJPEG
    options.raw_output = True
    options.out_format = decoded_format
    options.dec_width = cfg.source_width
    options.dec_height = cfg.source_height
    options.dec_fps = cfg.source_fps
    return options


def mipi_camera_options(cfg: AppConfig):
    options = pyneat.CameraInputOptions()
    if cfg.camera:
        options.camera_name = cfg.camera
    options.width = cfg.source_width
    options.height = cfg.source_height
    options.framerate_num = cfg.source_fps
    options.format = "NV12"
    return options


def build_source_run(cfg: AppConfig, fps: int):
    """The decoded frames of the selected source, newest first: a slow consumer skips frames."""
    graph = pyneat.Graph("source")
    if cfg.source_type == "rtsp":
        graph.add(pyneat.groups.rtsp_decoded_input(rtsp_source_options(cfg, fps)))
    elif cfg.source_type == "http":
        graph.add(pyneat.groups.http_mjpeg_decoded_input(http_source_options(cfg, fps)))
    elif cfg.source_type == "usb":
        camera = usb_camera_options(cfg)
        # Checks the camera and its mode once; most webcams send 4:2:2 JPEG, which decodes to NV16.
        info = pyneat.usb_camera_info(camera)
        graph.add(pyneat.nodes.usb_camera_input(camera, info))
        graph.add(pyneat.nodes.sima_decode(usb_decode_options(cfg, info.decoded_format)))
    else:
        graph.add(pyneat.nodes.camera_input(mipi_camera_options(cfg)))
    newest = pyneat.OutputOptions.latest()
    newest.max_buffers = 2
    graph.add(pyneat.nodes.output("frame", newest))

    run_options = pyneat.RunOptions()
    run_options.preset = pyneat.RunPreset.Realtime
    run_options.queue_depth = 2
    run_options.overflow_policy = pyneat.OverflowPolicy.KeepLatest
    run_options.output_memory = pyneat.OutputMemory.ZeroCopy
    return graph, graph.build(run_options)


def superpoint_options(width: int, height: int):
    """SuperPoint on the frame's luma: Neat stretches it to the model input and scales it to
    [0, 1]. The LightGlueV1 decode (threshold 0.0005, NMS radius 4, 4-pixel border) is the one
    both matchers' keypoints were extracted with, keeping the best 512 keypoints."""
    options = pyneat.ModelOptions()
    options.preprocess.kind = pyneat.InputKind.Image
    options.preprocess.enable = pyneat.AutoFlag.On
    options.preprocess.input_max_width = width
    options.preprocess.input_max_height = height
    options.preprocess.resize.enable = pyneat.AutoFlag.On
    options.preprocess.resize.mode = pyneat.ResizeMode.Stretch
    options.preprocess.color_convert.enable = pyneat.AutoFlag.On
    options.preprocess.color_convert.input_format = pyneat.PreprocessColorFormat.GRAY8
    options.preprocess.color_convert.output_format = pyneat.PreprocessColorFormat.GRAY8
    options.preprocess.normalize.enable = pyneat.AutoFlag.On
    options.preprocess.normalize.mean = (0.0, 0.0, 0.0)
    options.preprocess.normalize.stddev = (1.0, 1.0, 1.0)
    options.preprocess.normalize.has_explicit_stats = True
    options.decode_type = pyneat.BoxDecodeType.SuperPoint
    options.superpoint.profile = pyneat.SuperPointProfile.LightGlueV1
    options.superpoint.output_format = pyneat.SuperPointOutputFormat.FeaturePointsV1
    options.superpoint.descriptor_output_dtype = pyneat.TensorDType.Float32
    options.top_k = KEYPOINTS
    return options


def matcher_options():
    """The matchers take keypoint features, not images: Neat only quantizes and tessellates them."""
    options = pyneat.ModelOptions()
    options.preprocess.kind = pyneat.InputKind.Tensor
    options.preprocess.resize.enable = pyneat.AutoFlag.Off
    options.preprocess.color_convert.enable = pyneat.AutoFlag.Off
    options.preprocess.normalize.enable = pyneat.AutoFlag.Off
    return options


def model_input_size(model) -> tuple[int, int]:
    """The SuperPoint input size, the frame its keypoints are in (sima-neat/core#1328)."""
    contract = model.resolved_preprocess_plan().mla_contract
    if contract.width <= 0 or contract.height <= 0:
        raise RuntimeError("the SuperPoint model input size is unresolved")
    return int(contract.width), int(contract.height)


def frame_tensor(sample):
    part = sample
    if sample.fields:
        part = next(field for field in sample.fields if field.stream_label == "frame")
    return part.tensor if part.kind == pyneat.SampleKind.Tensor else part.tensors[0]


def frame_luma_and_bgr(tensor):
    """A decoded frame's luma plane, for SuperPoint, and its BGR pixels, for the view.

    Decoders pad rows and planes, so each plane is read at its own offset and stride. MJPEG with
    4:2:2 chroma, as most webcams send, decodes to NV16 instead of NV12; each frame is converted by
    the layout it actually has.
    """
    width, height = tensor.width(), tensor.height()
    payload = np.frombuffer(tensor.copy_payload_bytes(), np.uint8)
    planes = {plane.role: plane for plane in tensor.planes}
    nv16 = tensor.image_format() == pyneat.PixelFormat.NV16
    if not nv16 and not tensor.is_nv12():
        raise RuntimeError(f"unsupported decoded frame format {tensor.image_format()}")

    def plane(role, rows, channels):
        layout = planes[role]
        return np.ndarray(
            (rows, width // channels, channels),
            np.uint8,
            buffer=payload,
            offset=layout.byte_offset,
            strides=(layout.strides_bytes[0], channels, 1),
        )

    luma = plane(pyneat.PlaneRole.Y, height, 1)
    chroma = plane(pyneat.PlaneRole.UV, height if nv16 else height // 2, 2)
    if nv16:
        packed = np.dstack([luma[..., 0], chroma.reshape(height, width)])
        bgr = cv2.cvtColor(packed, cv2.COLOR_YUV2BGR_YUY2)
    else:
        bgr = cv2.cvtColorTwoPlane(luma, chroma, cv2.COLOR_YUV2BGR_NV12)
    return luma.copy(), bgr


def luma_tensor(luma):
    return pyneat.Tensor.from_numpy(
        luma, copy=True, image_format=pyneat.PixelFormat.GRAY8, memory=pyneat.TensorMemory.EV74
    )


def superpoint_features(outputs) -> Features | None:
    """The frame's 512 best keypoints, or None when SuperPoint found fewer."""
    (decoded,) = pyneat.decode_superpoint(list(outputs))
    keypoints = decoded.keypoints.to_numpy(copy=True)
    if len(keypoints) < KEYPOINTS:
        return None
    return Features(
        keypoints.astype(np.float32),
        decoded.scores.to_numpy(copy=True).astype(np.float32),
        decoded.descriptors.to_numpy(copy=True).astype(np.float32),
    )


def model_tensor(array):
    """A [512, C] matcher input as the [1, 1, 512, C] EV74 tensor the model reads."""
    return pyneat.Tensor.from_numpy(
        np.ascontiguousarray(array, np.float32).reshape(1, 1, KEYPOINTS, -1),
        copy=True,
        memory=pyneat.TensorMemory.EV74,
    )


def lightglue_encoding(keypoints):
    """LightGlue's rotary encoding [512, 128] of SuperPoint's 640x480 keypoints: each keypoint
    normalized as ((x, y) - (320, 240)) / 320, projected to 32 phases in float64, each phase
    repeated twice, then the cosines and the sines."""
    projection = np.asarray(LIGHTGLUE_PROJECTION, np.float64).T
    phases = ((keypoints.astype(np.float64) - (320.0, 240.0)) / 320.0) @ projection
    encoding = np.empty((len(keypoints), 128), np.float32)
    encoding[:, 0:64:2] = encoding[:, 1:64:2] = np.cos(phases)
    encoding[:, 64::2] = encoding[:, 65::2] = np.sin(phases)
    return encoding


def lightglue_inputs(features: Features) -> list:
    return [
        model_tensor(lightglue_encoding(features.keypoints)),
        model_tensor(features.descriptors),
    ]


def mutual_matches(scores, scores_t, threshold: float) -> Matches:
    """The mutual best pairs of a [512, 512] score matrix whose score exceeds the threshold.
    `scores_t` is its transpose, so both argmaxes run along contiguous rows."""
    best_live = scores.argmax(1)
    best_reference = scores_t.argmax(1)
    rows = np.arange(len(best_live))
    best = scores[rows, best_live]
    keep = (best_reference[best_live] == rows) & (best > threshold)
    return Matches(rows[keep], best_live[keep], best[keep])


def lightglue_matches(scores, threshold: float) -> Matches:
    """LightGlue's assignment [512, 512] is the probability that two keypoints match."""
    assignment = scores.reshape(KEYPOINTS, KEYPOINTS)
    return mutual_matches(assignment, cv2.transpose(assignment), threshold)


def superglue_inputs(features: Features) -> list:
    """SuperGlue's inputs: keypoints centered on the 640x480 input and divided by 0.7 times its
    longer side, SuperPoint's scores, and the descriptors."""
    keypoints = (features.keypoints.astype(np.float64) - (320.0, 240.0)) / (0.7 * 640.0)
    return [
        model_tensor(keypoints),
        model_tensor(features.scores),
        model_tensor(features.descriptors),
    ]


def superglue_matches(scores, threshold: float, iterations: int) -> Matches:
    """SuperGlue's Sinkhorn iterations over its [513, 513] couplings (index 512 is the dustbin),
    in the linear domain as in SuperGlue's match_pairs.py evaluation. Each score is the transport
    mass of a pair times M + N."""
    couplings = scores.reshape(KEYPOINTS + 1, KEYPOINTS + 1)
    total = 2 * KEYPOINTS
    marginal = np.full((KEYPOINTS + 1, 1), 1.0 / total, np.float32)
    marginal[-1] = KEYPOINTS / total
    # Clamped above float32's smallest normal number so that no row or column sums to zero;
    # the clamped entries are negligible against every row's largest.
    kernel = cv2.exp(np.maximum(couplings - couplings.max(), -80.0))
    kernel_t = cv2.transpose(kernel)
    u = np.ones((KEYPOINTS + 1, 1), np.float32)
    v = np.ones((KEYPOINTS + 1, 1), np.float32)
    # cv2.gemm, as numpy's reference BLAS on the board is several times slower.
    for _ in range(iterations):
        u = marginal / cv2.gemm(kernel, v, 1.0, None, 0.0)
        v = marginal / cv2.gemm(kernel_t, u, 1.0, None, 0.0)
    transport = kernel[:-1, :-1] * (u[:-1] * v[:-1].T) * total
    return mutual_matches(transport, cv2.transpose(transport), threshold)


def select_matcher(cfg: AppConfig) -> Matcher:
    if cfg.matcher_type == "lightglue":
        return Matcher(
            "LightGlue",
            lightglue_inputs,
            lambda scores: lightglue_matches(scores, cfg.match_threshold),
        )
    return Matcher(
        "SuperGlue",
        superglue_inputs,
        lambda scores: superglue_matches(scores, cfg.match_threshold, cfg.sinkhorn_iterations),
    )


def pipelined_run_options():
    run_options = pyneat.RunOptions()
    run_options.queue_depth = RUNNER_QUEUE_DEPTH
    run_options.overflow_policy = pyneat.OverflowPolicy.Block
    return run_options


def encoder_level(width: int, height: int) -> str:
    macroblocks = ((width + 15) // 16) * ((height + 15) // 16)
    return "4.0" if macroblocks <= LEVEL_4_MAX_MACROBLOCKS else "5.1"


def build_insight_run(cfg: AppConfig, width: int, height: int, fps: int):
    """Composed BGR frames, converted to the encoder's NV12 on the CVU, to H.264 for Insight."""
    input_options = pyneat.InputOptions()
    input_options.payload_type = pyneat.PayloadType.Image
    input_options.format = pyneat.Format.BGR
    input_options.width = width
    input_options.height = height
    input_options.depth = 3
    input_options.fps_n = fps
    input_options.fps_d = 1
    input_options.memory_policy = pyneat.InputMemoryPolicy.Ev74

    nv12 = pyneat.ColorConvertOptions()
    nv12.output_format = pyneat.ImageFormat.NV12
    nv12.surface = pyneat.ImageSurfaceLayout.EncoderNv12

    encode = pyneat.SimaEncodeOptions()
    encode.fps = fps
    sender_options = pyneat.VideoSenderOptions.from_raw(encode)
    sender_options.host = cfg.insight_host
    sender_options.video_port_base = cfg.video_port
    sender_options.encoder.bitrate_kbps = cfg.bitrate_kbps
    sender_options.encoder.level = encoder_level(width, height)

    graph = pyneat.Graph("insight")
    graph.add(pyneat.nodes.input(input_options))
    graph.add(pyneat.nodes.color_convert(nv12))
    graph.add(pyneat.groups.video_sender(sender_options))
    seed = bgr_tensor(np.zeros((height, width, 3), np.uint8))
    return graph, graph.build([seed]), sender_options.video_port


def bgr_tensor(bgr):
    return pyneat.Tensor.from_numpy(
        bgr, copy=True, image_format=pyneat.PixelFormat.BGR, memory=pyneat.TensorMemory.EV74
    )


def match_colors(scores, threshold: float):
    """BGR line colors from red at the match threshold to green at score 1."""
    strength = np.clip((scores - threshold) / (1.0 - threshold), 0.0, 1.0)
    return np.stack([np.zeros_like(strength), 255 * strength, 255 * (1 - strength)], 1).astype(int)


def match_lines(matches: Matches, reference: Features, live: Features, scale, width: int):
    """Line endpoints in pixels of the composed frame and their scores, weakest match first, so
    the strongest lines are drawn on top. Keypoints are in model-input pixels until
    sima-neat/core#1328, so `scale` maps them to the frame."""
    order = np.argsort(matches.scores)
    start = reference.keypoints[matches.reference[order]] * scale
    end = live.keypoints[matches.live[order]] * scale + (width, 0)
    return np.rint(np.hstack([start, end])).astype(int), matches.scores[order]


def inlier_ratio(lines) -> float:
    """The share of matches consistent with one camera motion (a fundamental matrix), the match
    quality the summary reports."""
    if len(lines) < 8:
        return 0.0
    _, inliers = cv2.findFundamentalMat(
        lines[:, :2].astype(np.float32), lines[:, 2:].astype(np.float32), cv2.FM_RANSAC, 1.0, 0.999
    )
    return float(inliers.sum()) / len(lines) if inliers is not None else 0.0


def draw_label(canvas, text: str, origin: tuple[int, int], scale: float) -> None:
    """White text on a darkened box, readable on any frame."""
    thickness = max(1, round(scale * 1.5))
    (width, height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x, y = origin
    pad = max(4, height // 3)
    canvas[y - height - pad : y + baseline + pad, x - pad : x + width + pad] //= 3
    cv2.putText(
        canvas,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def compose_frame(reference_bgr, live_bgr, lines, colors, status: str):
    """The reference on the left and the live frame on the right, one line per match."""
    height, width = reference_bgr.shape[:2]
    canvas = cv2.hconcat([reference_bgr, live_bgr])
    for (x0, y0, x1, y1), color in zip(lines.tolist(), colors.tolist(), strict=True):
        cv2.line(canvas, (x0, y0), (x1, y1), color, 1, cv2.LINE_AA)
        cv2.circle(canvas, (x0, y0), 2, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, (x1, y1), 2, color, -1, cv2.LINE_AA)
    scale = height / 1200
    margin, line_height = round(30 * scale), round(56 * scale)
    draw_label(canvas, "Reference", (margin, margin + line_height // 2), scale)
    draw_label(canvas, "Live", (width + margin, margin + line_height // 2), scale)
    draw_label(canvas, status, (width + margin, margin + line_height // 2 + line_height), scale)
    draw_label(
        canvas, "Line color: match score, red low, green high", (margin, height - margin), scale
    )
    return canvas


def put_unless_failed(items: queue.Queue, item, failed: threading.Event) -> bool:
    while not failed.is_set():
        try:
            items.put(item, timeout=0.1)
            return True
        except queue.Full:
            continue
    return False


def get_unless_failed(items: queue.Queue, failed: threading.Event):
    while not failed.is_set():
        try:
            return items.get(timeout=0.1)
        except queue.Empty:
            continue
    return None


def pull_frame(source_run, source_key: str):
    """The newest decoded frame's luma plane and BGR pixels."""
    sample = source_run.pull("frame", SOURCE_TIMEOUT_MS)
    if sample is None:
        raise RuntimeError(f"no frame arrived from {source_key} for {SOURCE_TIMEOUT_MS} ms")
    return frame_luma_and_bgr(frame_tensor(sample))


def capture_reference(cfg, source_run, superpoint_runner, stop: threading.Event) -> Frame | None:
    """The first frame with 512 keypoints, or None when the run stops first."""
    while not stop.is_set():
        luma, bgr = pull_frame(source_run, cfg.source_key)
        features = superpoint_features(superpoint_runner.run([luma_tensor(luma)]))
        if features is not None:
            return Frame(bgr, features)
    return None


def capture_frames(
    cfg: AppConfig, source_run, superpoint_runner, stop: threading.Event, links: StageLinks
) -> None:
    """Stage 1: each newest decoded frame's luma into SuperPoint, its BGR pixels to stage 2."""
    count = 0
    try:
        while not stop.is_set() and (cfg.frames == 0 or count < cfg.frames):
            luma, bgr = pull_frame(source_run, cfg.source_key)
            if not superpoint_runner.push([luma_tensor(luma)]):
                break
            if not put_unless_failed(links.captured, bgr, links.failed):
                break
            count += 1
    finally:
        links.captured_all.set()
        superpoint_runner.close_input()


def match_frames(
    superpoint_runner, matcher: Matcher, matcher_runner, reference_inputs: list, links: StageLinks
) -> None:
    """Stage 2: each frame's SuperPoint features into the matcher, beside the reference's."""
    try:
        while True:
            outputs = superpoint_runner.pull(MODEL_TIMEOUT_MS).tensors
            if not outputs:
                # An empty pull is the end of input once capture has finished, and a stall before.
                if links.captured_all.is_set() or links.failed.is_set():
                    break
                raise RuntimeError(f"SuperPoint returned no output for {MODEL_TIMEOUT_MS} ms")
            features = superpoint_features(outputs)
            del outputs
            bgr = get_unless_failed(links.captured, links.failed)
            if bgr is None:
                break
            if features is not None and not matcher_runner.push(
                reference_inputs + matcher.image_inputs(features)
            ):
                break
            if not put_unless_failed(links.matched, Frame(bgr, features), links.failed):
                break
    finally:
        matcher_runner.close_input()
        put_unless_failed(links.matched, None, links.failed)


def publish_frames(
    cfg: AppConfig,
    matcher: Matcher,
    matcher_runner,
    insight_run,
    reference: Frame,
    point_scale,
    stats: MatchStats,
    links: StageLinks,
) -> None:
    """Stage 3: each frame's matches drawn beside the reference and sent to Insight, in order.
    Frames are matched and drawn on worker threads; this thread pulls the matcher and sends."""
    width = reference.bgr.shape[1]
    no_lines, no_colors = np.empty((0, 4), int), np.empty((0, 3), int)

    def render(frame: Frame, scores, fps: float, sample_inliers: bool) -> Rendered:
        if scores is None:
            status = f"{matcher.name}: fewer than {KEYPOINTS} keypoints, not matched"
            canvas = compose_frame(reference.bgr, frame.bgr, no_lines, no_colors, status)
            return Rendered(canvas, None, None)
        matches = matcher.match(scores)
        lines, line_scores = match_lines(
            matches, reference.features, frame.features, point_scale, width
        )
        colors = match_colors(line_scores, cfg.match_threshold)
        status = f"{matcher.name}: {len(lines)} matches, {fps:.1f} fps"
        canvas = compose_frame(reference.bgr, frame.bgr, lines, colors, status)
        return Rendered(canvas, len(lines), inlier_ratio(lines) if sample_inliers else None)

    def send(rendered: Rendered) -> None:
        if not insight_run.push([bgr_tensor(rendered.canvas)]):
            raise RuntimeError("Insight video push failed")
        stats.record(rendered)
        if cfg.save_dir and cfg.save_every and stats.frames % cfg.save_every == 0:
            path = Path(cfg.save_dir) / f"frame_{stats.frames:06d}.jpg"
            if not cv2.imwrite(str(path), rendered.canvas):
                raise RuntimeError(f"could not save {path}")

    rendering: collections.deque = collections.deque()
    received = 0
    with ThreadPoolExecutor(RENDER_WORKERS, thread_name_prefix="render") as pool:
        while (frame := get_unless_failed(links.matched, links.failed)) is not None:
            scores = None
            if frame.features is not None:
                outputs = matcher_runner.pull(MODEL_TIMEOUT_MS).tensors
                if not outputs:
                    raise RuntimeError(
                        f"{matcher.name} returned no output for {MODEL_TIMEOUT_MS} ms"
                    )
                # A copy (about 1 MB), so that the runner reuses its output while workers draw.
                scores = outputs[0].to_numpy(copy=True)
                del outputs
            fps = stats.recent_fps()
            sample_inliers = received % INLIER_SAMPLE_INTERVAL == 0
            rendering.append(pool.submit(render, frame, scores, fps, sample_inliers))
            received += 1
            if len(rendering) == RENDER_WORKERS:
                send(rendering.popleft().result())
        while rendering:
            send(rendering.popleft().result())


def run(cfg: AppConfig, stop: threading.Event) -> int:
    matcher = select_matcher(cfg)
    fps = stream_fps(cfg)
    _source_graph, source_run = build_source_run(cfg, fps)
    runners = [source_run]
    try:
        luma, _ = pull_frame(source_run, cfg.source_key)
        height, width = luma.shape[:2]
        superpoint = pyneat.Model(cfg.superpoint_path, superpoint_options(width, height))
        model_width, model_height = model_input_size(superpoint)
        if (model_width, model_height) != MODEL_INPUT_SIZE:
            raise RuntimeError(
                f"SuperPoint input is {model_width}x{model_height}; both matchers need "
                f"{MODEL_INPUT_SIZE[0]}x{MODEL_INPUT_SIZE[1]}"
            )
        point_scale = np.array([width / model_width, height / model_height], np.float32)
        superpoint_runner = superpoint.build(
            [luma_tensor(luma)], run_options=pipelined_run_options()
        )
        runners.append(superpoint_runner)

        reference = capture_reference(cfg, source_run, superpoint_runner, stop)
        if reference is None:
            return 0
        # The reference's inputs are made once and sent with every live frame's.
        reference_inputs = matcher.image_inputs(reference.features)
        matcher_model = pyneat.Model(cfg.matcher_path, matcher_options())
        matcher_runner = matcher_model.build(
            reference_inputs * 2, run_options=pipelined_run_options()
        )
        runners.append(matcher_runner)
        _insight_graph, insight_run, video_port = build_insight_run(cfg, 2 * width, height, fps)
        runners.append(insight_run)
        print(
            f"source={cfg.source_type} stream={width}x{height}@{fps} matcher={matcher.name} "
            f"insight={cfg.insight_host}:{video_port}",
            flush=True,
        )

        stats = MatchStats()
        errors: list[BaseException] = []
        links = StageLinks(
            queue.Queue(STAGE_QUEUE_DEPTH),
            queue.Queue(STAGE_QUEUE_DEPTH),
            threading.Event(),
            threading.Event(),
        )

        def run_stage(target, *args) -> None:
            """Runs one stage; its error stops the others and is raised after they end."""
            try:
                target(*args)
            except BaseException as error:  # noqa: BLE001
                errors.append(error)
                links.failed.set()
                superpoint_runner.close_input()
                matcher_runner.close_input()

        match_args = (superpoint_runner, matcher, matcher_runner, reference_inputs, links)
        publish_args = (
            cfg,
            matcher,
            matcher_runner,
            insight_run,
            reference,
            point_scale,
            stats,
            links,
        )
        threads = [
            threading.Thread(target=run_stage, args=(match_frames, *match_args)),
            threading.Thread(target=run_stage, args=(publish_frames, *publish_args)),
        ]
        for thread in threads:
            thread.start()
        run_stage(capture_frames, cfg, source_run, superpoint_runner, stop, links)
        for thread in threads:
            thread.join()
        if errors:
            raise errors[0]
        print(f"{stats.summary()} video_sender={cfg.insight_host}:{video_port}", flush=True)
        return 0
    finally:
        for runner in reversed(runners):
            runner.close()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        cfg = load_app_config(args.config)
    except (OSError, ValueError, yaml.YAMLError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    for path in (cfg.superpoint_path, cfg.matcher_path):
        if not Path(path).is_file():
            print(f"Error: model does not exist: {path}", file=sys.stderr)
            return 2
    if cfg.save_dir:
        Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)

    load_runtime_dependencies()
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda _signum, _frame: stop.set())
    try:
        return run(cfg, stop)
    except Exception as error:  # noqa: BLE001
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
