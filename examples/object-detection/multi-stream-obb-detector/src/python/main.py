"""Multi-stream RTSP YOLO26 oriented-box detector that publishes to Insight, using pyneat."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import glob
import json
import math
from pathlib import Path
import sys

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "common" / "config.yaml"
MAX_STREAMS = 4
PULL_TIMEOUT_MS = 50
# Matches the C++ config reader's error wording.
KIND_NAMES = {bool: "true or false", int: "an integer", float: "numeric", str: "a string"}

cv2 = None
np = None
pyneat = None


@dataclass(frozen=True)
class AppConfig:
    model_path: str
    labels_path: str
    rtsp_urls: list[str]
    codec: str
    tcp: bool
    latency_ms: int
    frames: int
    max_inflight_per_stream: int
    max_inflight_total: int
    min_score: float
    max_detections: int
    insight_host: str
    video_port_base: int
    metadata_port_base: int
    debug_dir: str
    save_every: int


@dataclass
class Stream:
    index: int
    url: str
    width: int
    height: int
    fps: int
    metadata_sender: object
    processed: int = 0
    latest_frame: object | None = None


@dataclass(frozen=True)
class Detections:
    """One frame's oriented boxes in source-frame pixels, one array row per box.

    Each `boxes` row is cx, cy, width, height, angle_radians; a positive angle turns the
    box clockwise. Arrays keep a frame with hundreds of boxes cheap to publish.
    """

    boxes: object
    scores: object
    class_ids: object


def load_runtime_dependencies() -> None:
    global cv2, np, pyneat
    for path in glob.glob("/usr/lib/python3*/dist-packages"):
        if path not in sys.path:
            sys.path.insert(0, path)

    import cv2 as cv2_module
    import numpy as np_module
    import pyneat as pyneat_module

    cv2 = cv2_module
    np = np_module
    pyneat = pyneat_module


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-stream YOLO26 oriented-box detector")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--validate-config-only", action="store_true")
    return parser.parse_args(argv)


def setting(raw: dict, key: str, default, kind: type):
    """Reads the dotted `key` from the parsed YAML, or `default` when it is absent."""
    value = raw
    for part in key.split("."):
        value = value.get(part) if isinstance(value, dict) else None
    if value is None:
        return default
    if kind is float and isinstance(value, int) and not isinstance(value, bool):
        value = float(value)
    if not isinstance(value, kind) or (kind is int and isinstance(value, bool)):
        raise ValueError(f"{key} must be {KIND_NAMES[kind]}")
    return value


def parse_codec(value: str) -> str:
    lowered = value.lower()
    if lowered in {"h264", "avc", "h.264"}:
        return "h264"
    if lowered in {"h265", "hevc", "h.265"}:
        return "h265"
    raise ValueError("input.codec must be h264/avc or h265/hevc")


def parse_streams(raw: dict) -> list[str]:
    streams = raw.get("streams")
    if not isinstance(streams, list) or not streams:
        raise ValueError("streams must be a non-empty list")
    for index, url in enumerate(streams):
        if not isinstance(url, str) or not url.strip():
            raise ValueError(f"streams[{index}] must be a non-empty string")
    return streams


def validate_config(cfg: AppConfig) -> None:
    checks = [
        (cfg.model_path, "model.path must be set"),
        (cfg.labels_path, "model.labels must be set"),
        (len(cfg.rtsp_urls) <= MAX_STREAMS, "streams supports up to four entries"),
        (cfg.insight_host, "output.insight.host must be set"),
        (cfg.latency_ms >= 0, "input.latency_ms must be >= 0"),
        (cfg.frames >= 0, "inference.frames must be >= 0"),
        (
            cfg.max_inflight_per_stream == -1 or cfg.max_inflight_per_stream > 0,
            "inference.max_inflight_per_stream must be -1 or > 0",
        ),
        (
            cfg.max_inflight_total == -1 or cfg.max_inflight_total > 0,
            "inference.max_inflight_total must be -1 or > 0",
        ),
        (0.0 <= cfg.min_score <= 1.0, "inference.min_score must be between 0 and 1"),
        (cfg.max_detections > 0, "inference.max_detections must be > 0"),
        (cfg.video_port_base > 0, "output.insight.video_port_base must be > 0"),
        (cfg.metadata_port_base > 0, "output.insight.metadata_port_base must be > 0"),
        (cfg.save_every >= 0, "output.save_every must be >= 0"),
    ]
    for passed, message in checks:
        if not passed:
            raise ValueError(message)


def load_app_config(config_path: Path) -> AppConfig:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("config root must be a mapping")

    default_labels = str(Path(__file__).resolve().parents[1] / "common" / "dota_label.txt")
    cfg = AppConfig(
        model_path=setting(raw, "model.path", "", str),
        labels_path=setting(raw, "model.labels", default_labels, str),
        rtsp_urls=parse_streams(raw),
        codec=parse_codec(setting(raw, "input.codec", "h264", str)),
        tcp=setting(raw, "input.tcp", True, bool),
        latency_ms=setting(raw, "input.latency_ms", 100, int),
        frames=setting(raw, "inference.frames", 0, int),
        max_inflight_per_stream=setting(raw, "inference.max_inflight_per_stream", 4, int),
        max_inflight_total=setting(raw, "inference.max_inflight_total", 16, int),
        min_score=setting(raw, "inference.min_score", 0.30, float),
        max_detections=setting(raw, "inference.max_detections", 300, int),
        insight_host=setting(raw, "output.insight.host", "", str),
        video_port_base=setting(raw, "output.insight.video_port_base", 9000, int),
        metadata_port_base=setting(raw, "output.insight.metadata_port_base", 9100, int),
        debug_dir=setting(raw, "output.debug_dir", "", str),
        save_every=setting(raw, "output.save_every", 0, int),
    )
    validate_config(cfg)
    return cfg


def load_labels(labels_path: str) -> list[str]:
    path = Path(labels_path)
    if not path.is_file():
        raise RuntimeError(f"labels file does not exist: {path}")
    labels = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    labels = [label for label in labels if label]
    if not labels:
        raise RuntimeError(f"labels file is empty: {path}")
    return labels


def save_frames_enabled(cfg: AppConfig) -> bool:
    return bool(cfg.debug_dir) and cfg.save_every > 0


def rtsp_codec(cfg: AppConfig):
    return pyneat.RtspCodec.H265 if cfg.codec == "h265" else pyneat.RtspCodec.H264


def probe_stream(url: str) -> tuple[int, int, int]:
    """Returns the RTSP stream's width, height, and frame rate."""
    capture = cv2.VideoCapture(url)
    if not capture.isOpened():
        raise RuntimeError(f"failed to open RTSP source for probing: {url}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = int(round(capture.get(cv2.CAP_PROP_FPS) or 0))
    capture.release()
    if width <= 0 or height <= 0 or fps <= 0:
        raise RuntimeError(f"failed to probe RTSP frame size and rate: {url}")
    return width, height, fps


def open_stream(cfg: AppConfig, index: int, url: str) -> Stream:
    width, height, fps = probe_stream(url)
    metadata_options = pyneat.MetadataSenderOptions()
    metadata_options.host = cfg.insight_host
    metadata_options.channel = index
    metadata_options.metadata_port_base = cfg.metadata_port_base
    stream = Stream(index, url, width, height, fps, pyneat.MetadataSender(metadata_options))
    print(
        f"[stream {index}] rtsp={url} stream={width}x{height}@{fps} insight={cfg.insight_host} "
        f"video={video_options(cfg, index).video_port} "
        f"metadata={stream.metadata_sender.metadata_port()}",
        flush=True,
    )
    return stream


def build_model(cfg: AppConfig):
    opt = pyneat.ModelOptions()
    opt.preprocess.kind = pyneat.InputKind.Image
    opt.preprocess.enable = pyneat.AutoFlag.On
    opt.preprocess.color_convert.input_format = pyneat.PreprocessColorFormat.NV12
    # COCO_YOLO letterboxes, which keeps one scale on both axes so rotated boxes map back.
    opt.preprocess.preset = pyneat.NormalizePreset.COCO_YOLO
    opt.decode_type = pyneat.BoxDecodeType.YoloV26Obb
    opt.score_threshold = cfg.min_score
    # YOLO26 needs no NMS, and the oriented-box decoder requires it disabled.
    opt.nms_iou_threshold = 0.0
    opt.top_k = cfg.max_detections
    return pyneat.Model(cfg.model_path, opt)


def source_options(cfg: AppConfig, stream: Stream):
    opt = pyneat.RtspEncodedInputOptions()
    opt.url = stream.url
    opt.codec = rtsp_codec(cfg)
    opt.latency_ms = cfg.latency_ms
    opt.tcp = cfg.tcp
    opt.source_fps = stream.fps
    if cfg.codec == "h264":
        opt.fallback_h264_width = stream.width
        opt.fallback_h264_height = stream.height
    return opt


def decoder_options(cfg: AppConfig, stream: Stream):
    opt = pyneat.SimaDecodeOptions()
    opt.type = pyneat.SimaDecodeType.H265 if cfg.codec == "h265" else pyneat.SimaDecodeType.H264
    opt.out_format = pyneat.Format.NV12
    opt.decoder_name = "decoder"
    opt.raw_output = True
    opt.dec_width = stream.width
    opt.dec_height = stream.height
    opt.dec_fps = stream.fps
    return opt


def video_options(cfg: AppConfig, index: int):
    opt = pyneat.VideoSenderOptions.passthrough(rtsp_codec(cfg))
    opt.host = cfg.insight_host
    opt.channel = index
    opt.video_port_base = cfg.video_port_base
    opt.async_ = False
    return opt


def stream_id(index: int) -> str:
    return f"stream{index}"


def detector_link(cfg: AppConfig, index: int):
    """Keeps each stream's newest frames and tags them so results find their stream."""
    link = pyneat.GraphLinkOptions()
    link.policy = pyneat.GraphLinkPolicy.RealtimeLatestByStream
    link.max_inflight_per_stream = cfg.max_inflight_per_stream
    link.max_inflight_total = cfg.max_inflight_total
    link.stream_id = stream_id(index)
    return link


def latest_frame_link():
    """Keeps only the newest frame, so a slow consumer drops frames instead of slowing the stream."""
    link = pyneat.GraphLinkOptions()
    link.policy = pyneat.GraphLinkPolicy.RealtimeLatestByStream
    return link


def debug_frame_output(index: int) -> str:
    return f"debug_frame_{index}"


def build_graph(cfg: AppConfig, streams: list[Stream], model):
    """Feeds every stream into one shared detector; Core renders a single pipeline."""
    detector_input = model.input_appsrc_options(False)
    detector_input.block = True
    detector = pyneat.Graph("detector")
    detector.connect(pyneat.nodes.input("detector_frame", detector_input), model)

    graph = pyneat.Graph()
    for stream in streams:
        source = pyneat.Graph("rtsp_encoded_source")
        source.add(pyneat.groups.rtsp_encoded_input(source_options(cfg, stream)))

        decoder = pyneat.Graph("decode")
        decoder.add(pyneat.nodes.sima_decode(decoder_options(cfg, stream)))
        decoder.add(
            pyneat.nodes.caps_raw(
                "NV12", stream.width, stream.height, stream.fps, pyneat.CapsMemory.Any
            )
        )
        # Named so connect() can match the shared model's "detector_frame" input.
        decoder.add(pyneat.nodes.output("detector_frame"))

        video = pyneat.Graph("video_sender")
        video.add(pyneat.groups.video_sender(video_options(cfg, stream.index)))

        # A plain source link lets Core tee the encoded stream to the decoder and the sender.
        graph.connect(source, decoder)
        graph.connect(decoder, detector, detector_link(cfg, stream.index))
        graph.connect(source, video, latest_frame_link())
        if save_frames_enabled(cfg):
            frames = pyneat.Graph("debug_frame")
            frames.connect(
                pyneat.nodes.input("debug_frame"),
                pyneat.nodes.output(
                    debug_frame_output(stream.index), pyneat.OutputOptions.every_frame(4)
                ),
            )
            graph.connect(decoder, frames, latest_frame_link())

    detections = pyneat.Graph("detections")
    detections.add(pyneat.nodes.output("detections", pyneat.OutputOptions.every_frame(4)))
    graph.connect(detector, detections)
    return graph


def run_options():
    opt = pyneat.RunOptions()
    opt.preset = pyneat.RunPreset.Realtime
    opt.queue_depth = 4
    opt.overflow_policy = pyneat.OverflowPolicy.KeepLatest
    opt.output_memory = pyneat.OutputMemory.ZeroCopy
    return opt


def sample_tensors(sample) -> list:
    if sample.kind == pyneat.SampleKind.Tensor and sample.tensor is not None:
        return [sample.tensor]
    if sample.kind == pyneat.SampleKind.TensorSet:
        return list(sample.tensors)
    tensors = []
    for field in sample.fields:
        tensors.extend(sample_tensors(field))
    return tensors


def decode_detections(sample) -> Detections:
    """Reads the decoder's oriented boxes, already mapped to source-frame pixels."""
    tensors = sample_tensors(sample)
    if not tensors:
        raise RuntimeError("detection sample carried no tensors")
    result = pyneat.decode_obb(tensors)[0]
    # float64, so that rounded values stay short in JSON (float32 0.9 prints 0.899999976).
    return Detections(
        boxes=result.geometry.to_numpy().astype(np.float64).reshape(-1, 5),
        scores=result.scores.to_numpy().astype(np.float64),
        class_ids=result.class_ids.to_numpy().astype(np.int64),
    )


def label_for(labels: list[str], class_id: int) -> str:
    return labels[class_id] if 0 <= class_id < len(labels) else "unknown"


def insight_objects(detections: Detections, labels: list[str]) -> list[dict]:
    """Builds one frame's Insight `object-detection` entries.

    `oriented_bbox` is what Insight draws. `bbox` is the box's upright extent, which
    places the label and is what viewers without oriented-box support draw instead.
    Values are rounded so that a frame with hundreds of boxes stays under the
    64 KB limit of one metadata message.
    """
    cx, cy, width, height, angle = detections.boxes.T
    cos, sin = np.abs(np.cos(angle)), np.abs(np.sin(angle))
    extent_w = width * cos + height * sin
    extent_h = width * sin + height * cos
    bboxes = np.column_stack([cx - extent_w / 2, cy - extent_h / 2, extent_w, extent_h])
    oriented_bboxes = np.column_stack([detections.boxes[:, :4].round(1), angle.round(4)])
    rows = zip(
        bboxes.round(1).tolist(),
        oriented_bboxes.tolist(),
        detections.scores.round(3).tolist(),
        detections.class_ids.tolist(),
    )
    return [
        {
            "id": f"obj_{index}",
            "label": label_for(labels, class_id),
            "confidence": confidence,
            "bbox": bbox,
            "oriented_bbox": oriented_bbox,
        }
        for index, (bbox, oriented_bbox, confidence, class_id) in enumerate(rows, start=1)
    ]


def send_metadata(stream: Stream, sample, detections: Detections, labels: list[str]) -> None:
    objects = insight_objects(detections, labels)
    timestamp_ms = int(sample.pts_ns // 1_000_000) if sample.pts_ns >= 0 else -1
    frame_id = str(sample.frame_id) if sample.frame_id >= 0 else ""
    payload = json.dumps({"objects": objects}, separators=(",", ":"))
    # A lost message only drops one frame's boxes, so warn and keep streaming, as C++ does.
    error = ""
    try:
        sent = stream.metadata_sender.send_metadata(
            "object-detection", payload, timestamp_ms, frame_id
        )
    except RuntimeError as exc:
        sent, error = False, str(exc)
    if not sent:
        print(f"[warn] stream {stream.index} metadata send failed: {error}", file=sys.stderr)


def nv12_to_bgr(tensor):
    width = int(tensor.width())
    height = int(tensor.height())
    payload = np.frombuffer(tensor.contiguous().copy_payload_bytes(), dtype=np.uint8)
    nv12 = payload[: width * height * 3 // 2].reshape((height * 3 // 2, width))
    return cv2.cvtColor(nv12, cv2.COLOR_YUV2BGR_NV12)


def draw_boxes(frame, detections: Detections, labels: list[str]) -> None:
    rows = zip(detections.boxes.tolist(), detections.scores.tolist(), detections.class_ids.tolist())
    for (cx, cy, width, height, angle), score, class_id in rows:
        corners = cv2.boxPoints(((cx, cy), (width, height), math.degrees(angle)))
        corners = np.round(corners).astype(np.int32)
        cv2.polylines(frame, [corners], True, (0, 255, 0), 2)
        text_x, text_y = int(corners[:, 0].min()), int(corners[:, 1].min())
        cv2.putText(
            frame,
            f"{label_for(labels, class_id)} {score:.2f}",
            (text_x, max(0, text_y - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 0),
            1,
        )


def save_debug_frame(cfg: AppConfig, stream: Stream, detections: Detections, labels) -> None:
    if stream.latest_frame is None or stream.processed % cfg.save_every != 0:
        return
    frame = stream.latest_frame.copy()
    draw_boxes(frame, detections, labels)
    path = Path(cfg.debug_dir) / f"stream_{stream.index}_frame_{stream.processed}.jpg"
    if not cv2.imwrite(str(path), frame):
        print(f"[warn] failed to write debug frame: {path}", file=sys.stderr)


def drain_debug_frames(run, streams: list[Stream]) -> None:
    """Keeps each stream's newest decoded frame for the next saved detection."""
    for stream in streams:
        while (sample := run.pull(debug_frame_output(stream.index), 0)) is not None:
            tensors = sample_tensors(sample)
            if tensors:
                stream.latest_frame = nv12_to_bgr(tensors[0])


def pull_detections(run):
    """Returns the next detection sample, or None when none arrived within the timeout."""
    sample = run.pull("detections", PULL_TIMEOUT_MS)
    if sample is not None:
        return sample
    # pyneat reports a closed output as None, like a timeout, so check the run itself.
    if not run.running():
        raise RuntimeError(f"detections output closed unexpectedly: {run.last_error()}")
    if run.last_error():
        raise RuntimeError(f"runtime error: {run.last_error()}")
    return None


def stream_for(sample, streams: list[Stream]) -> Stream:
    for stream in streams:
        if sample.stream_id == stream_id(stream.index):
            return stream
    raise RuntimeError(f"detection sample has an unknown stream id: {sample.stream_id!r}")


def all_streams_done(streams: list[Stream], frame_limit: int) -> bool:
    return frame_limit > 0 and all(stream.processed >= frame_limit for stream in streams)


def run_app(cfg: AppConfig) -> None:
    if save_frames_enabled(cfg):
        Path(cfg.debug_dir).mkdir(parents=True, exist_ok=True)

    labels = load_labels(cfg.labels_path)
    model = build_model(cfg)
    streams = [open_stream(cfg, index, url) for index, url in enumerate(cfg.rtsp_urls)]
    graph = build_graph(cfg, streams, model)

    run = graph.build(run_options())
    try:
        while not all_streams_done(streams, cfg.frames):
            if save_frames_enabled(cfg):
                drain_debug_frames(run, streams)
            sample = pull_detections(run)
            if sample is None:
                continue
            stream = stream_for(sample, streams)
            if cfg.frames > 0 and stream.processed >= cfg.frames:
                continue
            detections = decode_detections(sample)
            stream.processed += 1
            send_metadata(stream, sample, detections, labels)
            if save_frames_enabled(cfg):
                save_debug_frame(cfg, stream, detections, labels)
    finally:
        run.close()
        for stream in streams:
            print(f"[stream {stream.index}] processed={stream.processed}", flush=True)


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        if not args.config.exists():
            print(f"Error: config file not found: {args.config}", file=sys.stderr)
            return 2
        cfg = load_app_config(args.config)
        if args.validate_config_only:
            print(f"Config validated: {args.config} (streams={len(cfg.rtsp_urls)})")
            return 0

        load_runtime_dependencies()
        run_app(cfg)
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"[ERR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
