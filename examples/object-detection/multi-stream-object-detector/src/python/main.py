"""Multi-camera RTSP YOLO26 Insight example using pyneat."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import glob
import json
import os
import struct
import sys
import threading
import time

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "common" / "config.yaml"

cv2 = None
np = None
pyneat = None


@dataclass(frozen=True)
class AppConfig:
    model_path: str
    labels_path: Path
    rtsp_urls: list[str]
    latency_ms: int = 100
    tcp: bool = True
    frames: int = 0
    fps: int = 0
    min_score: float = 0.55
    nms_iou: float = 0.60
    max_detections: int = 50
    profile: bool = False
    warmup_frames: int = 30
    insight_host: str = "127.0.0.1"
    video_port_base: int = 9000
    metadata_port_base: int = 9100
    video_enabled: bool = True
    save_dir: str = ""
    save_every: int = 0


@dataclass
class StreamRuntime:
    index: int
    url: str
    model: object
    graph: object
    run: object
    metadata_sender: object
    labels: list[str]
    profile: "ProfileWindow"
    frame_w: int
    frame_h: int
    output_fps: int
    video_port: int
    processed: int = 0
    closed: bool = False


class ProfileWindow:
    def __init__(self, enabled: bool, stream_index: int, interval: int = 100) -> None:
        self.enabled = enabled
        self.stream_index = stream_index
        self.interval = interval
        self.frames = 0
        self.boxes = 0
        self.start_ms = 0.0
        self.detection_pull_ms = 0.0
        self.metadata_send_ms = 0.0

    def add(self, detection_pull_ms: float, metadata_send_ms: float, box_count: int) -> None:
        if not self.enabled:
            return
        if self.frames == 0:
            self.start_ms = time_ms()
        self.frames += 1
        self.boxes += box_count
        self.detection_pull_ms += detection_pull_ms
        self.metadata_send_ms += metadata_send_ms
        if self.frames >= self.interval:
            self.flush()

    def flush(self) -> None:
        if not self.enabled or self.frames == 0:
            return
        elapsed = time_ms() - self.start_ms
        output_fps = self.frames * 1000.0 / elapsed if elapsed > 0.0 else 0.0
        print(
            f"[profile stream={self.stream_index}] frames={self.frames} "
            f"output_fps={output_fps} "
            f"avg_detection_pull_ms={self.detection_pull_ms / self.frames} "
            f"avg_metadata_send_ms={self.metadata_send_ms / self.frames} "
            f"avg_boxes={self.boxes / self.frames}",
            flush=True,
        )
        self.frames = 0
        self.boxes = 0
        self.start_ms = 0.0
        self.detection_pull_ms = 0.0
        self.metadata_send_ms = 0.0


def load_runtime_dependencies() -> None:
    global cv2, np, pyneat
    if pyneat is not None:
        return

    for path in glob.glob("/usr/lib/python3*/dist-packages"):
        if path not in sys.path:
            sys.path.insert(0, path)

    import cv2 as cv2_module
    import numpy as np_module
    import pyneat as pyneat_module

    cv2 = cv2_module
    np = np_module
    pyneat = pyneat_module


def time_ms() -> float:
    return time.perf_counter() * 1000.0


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-camera RTSP YOLO26 Insight example")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--validate-config-only", action="store_true")
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
    if not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    return int(value)


def float_or(raw: dict, key: str, default: float) -> float:
    value = raw.get(key, default)
    if value is None:
        return default
    if not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be numeric")
    return float(value)


def bool_or(raw: dict, key: str, default: bool) -> bool:
    value = raw.get(key, default)
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be true or false")
    return value


def validate_config(cfg: AppConfig) -> None:
    if not cfg.model_path:
        raise ValueError("model.path must be set")
    if not str(cfg.labels_path):
        raise ValueError("model.labels must be set")
    if not cfg.rtsp_urls:
        raise ValueError("streams must be set")
    if len(cfg.rtsp_urls) > 4:
        raise ValueError("this phase supports up to four streams")
    if not cfg.insight_host:
        raise ValueError("output.insight.host must be set")
    if cfg.latency_ms < 0:
        raise ValueError("input.latency_ms must be >= 0")
    if cfg.frames < 0:
        raise ValueError("inference.frames must be >= 0")
    if cfg.fps < 0:
        raise ValueError("inference.fps must be >= 0")
    if not 0.0 <= cfg.min_score <= 1.0:
        raise ValueError("inference.min_score must be between 0 and 1")
    if not 0.0 <= cfg.nms_iou <= 1.0:
        raise ValueError("inference.nms_iou must be between 0 and 1")
    if cfg.max_detections <= 0:
        raise ValueError("inference.max_detections must be > 0")
    if cfg.warmup_frames < 0:
        raise ValueError("runtime.warmup_frames must be >= 0")
    if cfg.video_port_base <= 0:
        raise ValueError("output.insight.video_port_base must be > 0")
    if cfg.metadata_port_base <= 0:
        raise ValueError("output.insight.metadata_port_base must be > 0")
    if cfg.save_every < 0:
        raise ValueError("output.save_every must be >= 0")


def load_app_config(config_path: Path) -> AppConfig:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("config root must be a mapping")

    model = section(raw, "model")
    input_cfg = section(raw, "input")
    inference = section(raw, "inference")
    runtime = section(raw, "runtime")
    output = section(raw, "output")
    insight = section(output, "insight")
    default_labels = Path(__file__).resolve().parents[1] / "common" / "coco_label.txt"

    streams = raw.get("streams")
    if not isinstance(streams, list) or not streams:
        raise ValueError("streams must be a non-empty list")
    rtsp_urls = []
    for index, value in enumerate(streams):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"streams[{index}] must be a non-empty string")
        rtsp_urls.append(value)

    cfg = AppConfig(
        model_path=string_or(model, "path"),
        labels_path=Path(string_or(model, "labels", str(default_labels))),
        rtsp_urls=rtsp_urls,
        latency_ms=int_or(input_cfg, "latency_ms", 100),
        tcp=bool_or(input_cfg, "tcp", True),
        frames=int_or(inference, "frames", 0),
        fps=int_or(inference, "fps", 0),
        min_score=float_or(inference, "min_score", 0.55),
        nms_iou=float_or(inference, "nms_iou", 0.60),
        max_detections=int_or(inference, "max_detections", 50),
        profile=bool_or(runtime, "profile", False),
        warmup_frames=int_or(runtime, "warmup_frames", 30),
        insight_host=string_or(insight, "host"),
        video_port_base=int_or(insight, "video_port_base", 9000),
        metadata_port_base=int_or(insight, "metadata_port_base", 9100),
        video_enabled=bool_or(output, "video_enabled", True),
        save_dir=string_or(output, "debug_dir"),
        save_every=int_or(output, "save_every", 0),
    )
    validate_config(cfg)
    return cfg


def load_labels(labels_path: Path) -> list[str]:
    if not labels_path.is_file():
        raise RuntimeError(f"labels file does not exist: {labels_path}")
    labels = [line.strip() for line in labels_path.read_text(encoding="utf-8").splitlines()]
    labels = [label for label in labels if label]
    if not labels:
        raise RuntimeError(f"labels file is empty: {labels_path}")
    return labels


def extract_tensor_bbox_payload(sample, tensor=None) -> bytes:
    tensor = tensor if tensor is not None else getattr(sample, "tensor", None)
    if tensor is None:
        raise RuntimeError("capture_missing_tensor")
    fmt = getattr(sample, "payload_tag", "") or getattr(sample, "format", "")
    semantic = getattr(tensor, "semantic", None)
    tess = getattr(semantic, "tess", None)
    if not fmt and tess is not None:
        fmt = getattr(tess, "format", "")
    fmt = str(fmt).upper()
    if fmt and fmt != "BBOX":
        raise RuntimeError(f"capture_expected_bbox format={fmt}")
    payload = tensor.copy_payload_bytes()
    if not payload:
        raise RuntimeError("capture_empty_payload")
    return payload


def extract_bbox_payload(sample) -> bytes:
    if sample.kind == pyneat.SampleKind.Bundle:
        for field in sample.fields:
            try:
                return extract_bbox_payload(field)
            except RuntimeError:
                continue
        raise RuntimeError("bundle missing BBOX field")
    if sample.kind == pyneat.SampleKind.TensorSet and sample.tensors:
        return extract_tensor_bbox_payload(sample, sample.tensors[0])
    if sample.kind != pyneat.SampleKind.Tensor:
        raise RuntimeError("capture_expected_tensor")
    return extract_tensor_bbox_payload(sample)


def parse_boxes_strict(payload: bytes, img_w: int, img_h: int, expected_topk: int) -> list[dict]:
    if len(payload) < 4:
        raise RuntimeError("bbox buffer too small")

    header = struct.unpack_from("<I", payload, 0)[0]
    max_boxes = (len(payload) - 4) // 24
    if header > max_boxes:
        raise RuntimeError("bbox header exceeds payload count")
    if expected_topk > 0 and header > expected_topk:
        raise RuntimeError("bbox header exceeds expected topk")

    boxes = []
    offset = 4
    for _ in range(header):
        x, y, w, h, score, class_id = struct.unpack_from("<iiiifi", payload, offset)
        offset += 24
        boxes.append(
            {
                "x1": max(0.0, min(float(x), float(img_w))),
                "y1": max(0.0, min(float(y), float(img_h))),
                "x2": max(0.0, min(float(x + w), float(img_w))),
                "y2": max(0.0, min(float(y + h), float(img_h))),
                "score": float(score),
                "class_id": int(class_id),
            }
        )
    return boxes


def build_metadata_boxes(boxes: list[dict], labels: list[str], frame_w: int, frame_h: int) -> list[dict]:
    metadata_boxes = []
    for index, box in enumerate(boxes, start=1):
        x = max(0, int(box["x1"]))
        y = max(0, int(box["y1"]))
        w = max(0, int(box["x2"] - box["x1"]))
        h = max(0, int(box["y2"] - box["y1"]))
        if x + w > frame_w:
            w = frame_w - x
        if y + h > frame_h:
            h = frame_h - y
        class_id = int(box["class_id"])
        metadata_boxes.append(
            {
                "id": f"obj_{index}",
                "label": labels[class_id] if 0 <= class_id < len(labels) else "unknown",
                "confidence": float(box["score"]),
                "bbox": [float(x), float(y), float(max(0, w)), float(max(0, h))],
            }
        )
    return metadata_boxes


def probe_rtsp(url: str) -> tuple[int, int, int]:
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        raise RuntimeError(f"failed to open RTSP source for probing: {url}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = int(round(cap.get(cv2.CAP_PROP_FPS) or 0))
    cap.release()
    if width <= 0 or height <= 0:
        raise RuntimeError("failed to probe RTSP frame size")
    if fps <= 0:
        raise RuntimeError("failed to probe RTSP frame rate")
    return width, height, fps


def make_source_options(cfg: AppConfig, url: str, fps: int, width: int, height: int):
    opt = pyneat.RtspDecodedInputOptions()
    opt.url = url
    opt.latency_ms = cfg.latency_ms
    opt.tcp = cfg.tcp
    opt.payload_type = 96
    opt.insert_queue = True
    opt.decoder_name = "decoder"
    opt.decoder_raw_output = True
    opt.auto_caps_from_stream = True
    opt.fallback_h264_width = width
    opt.fallback_h264_height = height
    opt.fallback_h264_fps = fps
    opt.output_caps.enable = True
    opt.output_caps.format = pyneat.Format.NV12
    opt.output_caps.width = width
    opt.output_caps.height = height
    opt.output_caps.fps = fps
    opt.output_caps.memory = pyneat.CapsMemory.Any
    return opt


def make_model(cfg: AppConfig):
    opt = pyneat.ModelOptions()
    opt.preprocess.kind = pyneat.InputKind.Image
    opt.preprocess.enable = pyneat.AutoFlag.On
    opt.preprocess.color_convert.input_format = pyneat.PreprocessColorFormat.NV12
    opt.preprocess.preset = pyneat.NormalizePreset.COCO_YOLO
    opt.decode_type = pyneat.BoxDecodeType.YoloV26
    opt.score_threshold = cfg.min_score
    opt.nms_iou_threshold = cfg.nms_iou
    opt.top_k = cfg.max_detections
    return pyneat.Model(cfg.model_path, opt)


def build_stream_runtime(
    cfg: AppConfig, stream_index: int, url: str, labels: list[str]
) -> StreamRuntime:
    frame_w, frame_h, fps = probe_rtsp(url)
    output_fps = cfg.fps if cfg.fps > 0 else fps
    model = make_model(cfg)

    source = pyneat.groups.rtsp_decoded_input(make_source_options(cfg, url, fps, frame_w, frame_h))
    save_debug_frames = bool(cfg.save_dir and cfg.save_every > 0)
    outputs = ["model"]
    if cfg.video_enabled:
        outputs.append("video")
    if save_debug_frames:
        outputs.append("debug_frame")
    branch = pyneat.graphs.branch("source", outputs)

    graph = pyneat.Graph()
    graph.connect(source, branch)
    video_port = 0
    if cfg.video_enabled:
        video_options = pyneat.VideoSenderOptions.h264_rtp_udp_from_raw(frame_w, frame_h, output_fps)
        video_options.host = cfg.insight_host
        video_options.channel = stream_index
        video_options.video_port_base = cfg.video_port_base
        video_options.encoder.bitrate_kbps = 1000
        video_port = video_options.video_port

        video_graph = pyneat.Graph("video")
        video_graph.connect(pyneat.nodes.input("video"), pyneat.groups.video_sender(video_options))
        graph.connect(branch, video_graph)

    model_graph = pyneat.Graph("model")
    model_graph.connect(pyneat.nodes.input("model"), model)
    detections_graph = pyneat.Graph("detections")
    detections_graph.add(pyneat.nodes.output("detections", pyneat.OutputOptions.every_frame(4)))
    graph.connect(branch, model_graph)
    graph.connect(model_graph, detections_graph)

    if save_debug_frames:
        frames = pyneat.Graph("debug_frame")
        frames.add(pyneat.nodes.output("debug_frame", pyneat.OutputOptions.every_frame(4)))
        debug_join = pyneat.graphs.combine(
            ["debug_frame", "detections"], "debug_output", pyneat.CombinePolicy.ByFrame
        )
        graph.connect(branch, frames)
        graph.connect(frames, debug_join)
        graph.connect(detections_graph, debug_join)
    if cfg.profile:
        print(f"Backend stream={stream_index}:\n{graph.describe_backend()}")

    run_options = pyneat.RunOptions()
    run_options.preset = pyneat.RunPreset.Realtime
    run_options.queue_depth = 3
    run_options.overflow_policy = pyneat.OverflowPolicy.KeepLatest
    run_options.output_memory = pyneat.OutputMemory.ZeroCopy
    run = graph.build(run_options)

    metadata_options = pyneat.MetadataSenderOptions()
    metadata_options.host = cfg.insight_host
    metadata_options.channel = stream_index
    metadata_options.metadata_port_base = cfg.metadata_port_base
    metadata_sender = pyneat.MetadataSender(metadata_options)

    print(
        f"[stream {stream_index}] rtsp={url} stream={frame_w}x{frame_h}@{output_fps} "
        f"insight={cfg.insight_host} "
        f"video={video_port if cfg.video_enabled else 'disabled'} "
        f"metadata={metadata_sender.metadata_port()}"
    )
    return StreamRuntime(
        index=stream_index,
        url=url,
        model=model,
        graph=graph,
        run=run,
        metadata_sender=metadata_sender,
        labels=labels,
        profile=ProfileWindow(cfg.profile, stream_index),
        frame_w=frame_w,
        frame_h=frame_h,
        output_fps=output_fps,
        video_port=video_port,
    )


def send_metadata(stream: StreamRuntime, sample, boxes: list[dict]) -> None:
    metadata_boxes = build_metadata_boxes(boxes, stream.labels, stream.frame_w, stream.frame_h)
    pts_ns = getattr(sample, "pts_ns", -1)
    if pts_ns is not None and pts_ns >= 0:
        timestamp_ms = int(pts_ns // 1_000_000)
    else:
        timestamp_ms = int(time.time() * 1000)
    frame_id = getattr(sample, "frame_id", -1)
    frame_id = int(frame_id) if frame_id is not None and frame_id >= 0 else 0
    stream.metadata_sender.send_metadata(
        "object-detection",
        json.dumps({"objects": metadata_boxes}, separators=(",", ":")),
        timestamp_ms,
        str(frame_id),
    )


def tensor_dim(tensor, name: str) -> int:
    value = getattr(tensor, name)
    return int(value() if callable(value) else value)


def first_tensor_from_sample(sample):
    if sample is None:
        return None
    if sample.kind == pyneat.SampleKind.Tensor and sample.tensor is not None:
        return sample.tensor
    if sample.kind == pyneat.SampleKind.TensorSet and sample.tensors:
        return sample.tensors[0]
    for field in sample.fields:
        tensor = first_tensor_from_sample(field)
        if tensor is not None:
            return tensor
    return None


def tensor_bgr_from_decoded(tensor):
    if tensor.is_nv12():
        width = tensor_dim(tensor, "width")
        height = tensor_dim(tensor, "height")
        payload = np.frombuffer(tensor.copy_payload_bytes(), dtype=np.uint8)
        expected = width * height * 3 // 2
        if payload.size < expected:
            raise RuntimeError(f"NV12 payload too small: {payload.size} < {expected}")
        nv12 = payload[:expected].reshape((height * 3 // 2, width))
        return np.ascontiguousarray(cv2.cvtColor(nv12, cv2.COLOR_YUV2BGR_NV12))

    frame = np.asarray(tensor.to_numpy(copy=True))
    if frame.ndim == 4 and frame.shape[0] == 1:
        frame = frame[0]
    if frame.ndim != 3:
        raise RuntimeError(f"unexpected decoded tensor shape {frame.shape}")
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(frame)


def draw_boxes(frame, boxes: list[dict], min_score: float) -> None:
    for box in boxes:
        score = float(box["score"])
        if score < min_score:
            continue
        x1 = max(0, int(round(box["x1"])))
        y1 = max(0, int(round(box["y1"])))
        x2 = min(frame.shape[1] - 1, int(round(box["x2"])))
        y2 = min(frame.shape[0] - 1, int(round(box["y2"])))
        if x2 <= x1 or y2 <= y1:
            continue
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"id={int(box['class_id'])} score={score:.6f}"
        cv2.putText(
            frame,
            label,
            (x1, max(0, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 0),
            1,
        )


def maybe_save_debug_frame(cfg: AppConfig, stream: StreamRuntime, sample, boxes: list[dict]) -> None:
    if not cfg.save_dir or cfg.save_every <= 0 or stream.processed % cfg.save_every != 0:
        return

    frame_tensor = first_tensor_from_sample(sample)
    if frame_tensor is None:
        return

    frame = tensor_bgr_from_decoded(frame_tensor)
    draw_boxes(frame, boxes, cfg.min_score)
    out_path = Path(cfg.save_dir) / f"stream_{stream.index}_frame_{stream.processed}.jpg"
    if not cv2.imwrite(str(out_path), frame):
        print(f"[warn] failed to write output frame: {out_path}", file=sys.stderr)


def process_stream_once(stream: StreamRuntime, cfg: AppConfig, output_name: str) -> bool:
    pull_start = time_ms()
    sample = stream.run.pull(output_name, 50)
    pull_end = time_ms()
    if sample is None:
        if hasattr(stream.run, "can_pull") and not stream.run.can_pull():
            stream.closed = True
        return False

    payload = extract_bbox_payload(sample)
    boxes = parse_boxes_strict(payload, stream.frame_w, stream.frame_h, cfg.max_detections)

    stream.processed += 1
    warming_up = stream.processed <= cfg.warmup_frames
    if not warming_up:
        metadata_start = time_ms()
        send_metadata(stream, sample, boxes)
        metadata_end = time_ms()
        maybe_save_debug_frame(cfg, stream, sample, boxes)
        stream.profile.add(pull_end - pull_start, metadata_end - metadata_start, len(boxes))
    return True


def consume_stream(
    stream: StreamRuntime,
    cfg: AppConfig,
    output_name: str,
    stop_event: threading.Event,
    errors: list[Exception],
) -> None:
    try:
        while (
            not stop_event.is_set()
            and not stream.closed
            and (cfg.frames <= 0 or stream.processed < cfg.frames)
        ):
            process_stream_once(stream, cfg, output_name)
    except Exception as exc:
        errors.append(exc)
        stop_event.set()


def run_app(cfg: AppConfig) -> None:
    if cfg.profile:
        os.environ.setdefault("SIMA_GST_ELEMENT_TIMINGS", "1")
        os.environ.setdefault("SIMA_GST_FLOW_DEBUG", "1")
        os.environ.setdefault("SIMA_GST_BOUNDARY_PROBES", "1")
    if cfg.save_dir:
        Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)

    labels = load_labels(cfg.labels_path)
    streams = [
        build_stream_runtime(cfg, index, url, labels) for index, url in enumerate(cfg.rtsp_urls)
    ]
    output_name = "debug_output" if cfg.save_dir and cfg.save_every > 0 else "detections"
    stop_event = threading.Event()
    errors: list[Exception] = []
    consumers = [
        threading.Thread(
            target=consume_stream,
            args=(stream, cfg, output_name, stop_event, errors),
            name=f"stream-{stream.index}-consumer",
        )
        for stream in streams
    ]

    try:
        for consumer in consumers:
            consumer.start()
        for consumer in consumers:
            consumer.join()
    except KeyboardInterrupt:
        stop_event.set()
    finally:
        stop_event.set()
        for stream in streams:
            stream.run.close()
        for consumer in consumers:
            if consumer.is_alive():
                consumer.join(timeout=2.0)
        for stream in streams:
            stream.profile.flush()
            print(f"[stream {stream.index}] processed={stream.processed}")
    if errors:
        raise errors[0]


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
    except Exception as exc:
        print(f"[ERR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
