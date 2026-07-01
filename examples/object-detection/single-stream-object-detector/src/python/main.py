"""Single-camera RTSP YOLO26 Insight example using pyneat."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import glob
import json
import os
from pathlib import Path
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
    rtsp_url: str
    latency_ms: int = 200
    tcp: bool = True
    frames: int = 0
    min_score: float = 0.55
    nms_iou: float = 0.60
    max_detections: int = 50
    profile: bool = False
    profile_interval: int = 100
    insight_host: str = "127.0.0.1"
    video_port: int = 9000
    metadata_port: int = 9100
    save_dir: str = ""
    save_every: int = 0


@dataclass
class PipelineRuntime:
    model: object
    source_graph: object
    source_run: object
    decode_graph: object
    decode_run: object
    video_graph: object
    video_run: object
    pending_encoded_sample: object
    metadata_sender: object
    labels: list[str]
    frame_w: int
    frame_h: int
    video_port: int


class ProfileWindow:
    def __init__(self, enabled: bool, interval: int) -> None:
        self.enabled = enabled
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
            f"[profile] frames={self.frames} "
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
    parser = argparse.ArgumentParser(description="Single-camera RTSP YOLO26 Insight example")
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
    return value


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
    if not cfg.rtsp_url:
        raise ValueError("source.rtsp_url must be set")
    if not cfg.model_path:
        raise ValueError("model.path must be set")
    if not str(cfg.labels_path):
        raise ValueError("model.labels must be set")
    if not cfg.insight_host:
        raise ValueError("output.insight.host must be set")
    if cfg.latency_ms < 0:
        raise ValueError("source.latency_ms must be >= 0")
    if cfg.frames < 0:
        raise ValueError("inference.frames must be >= 0")
    if not 0.0 <= cfg.min_score <= 1.0:
        raise ValueError("inference.min_score must be between 0 and 1")
    if not 0.0 <= cfg.nms_iou <= 1.0:
        raise ValueError("inference.nms_iou must be between 0 and 1")
    if cfg.max_detections <= 0:
        raise ValueError("inference.max_detections must be > 0")
    if cfg.profile_interval <= 0:
        raise ValueError("runtime.profile_interval must be > 0")
    if cfg.video_port <= 0:
        raise ValueError("output.insight.video_port must be > 0")
    if cfg.metadata_port <= 0:
        raise ValueError("output.insight.metadata_port must be > 0")
    if cfg.save_every < 0:
        raise ValueError("output.save_every must be >= 0")


def load_app_config(config_path: Path) -> AppConfig:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("config root must be a mapping")

    model = section(raw, "model")
    source = section(raw, "source")
    inference = section(raw, "inference")
    runtime = section(raw, "runtime")
    output = section(raw, "output")
    insight = section(output, "insight")
    default_labels = Path(__file__).resolve().parents[1] / "common" / "coco_label.txt"

    cfg = AppConfig(
        model_path=string_or(model, "path"),
        labels_path=Path(string_or(model, "labels", str(default_labels))),
        rtsp_url=string_or(source, "rtsp_url"),
        latency_ms=int_or(source, "latency_ms", 200),
        tcp=bool_or(source, "tcp", True),
        frames=int_or(inference, "frames", 0),
        min_score=float_or(inference, "min_score", 0.55),
        nms_iou=float_or(inference, "nms_iou", 0.60),
        max_detections=int_or(inference, "max_detections", 50),
        profile=bool_or(runtime, "profile", False),
        profile_interval=int_or(runtime, "profile_interval", 100),
        insight_host=string_or(insight, "host"),
        video_port=int_or(insight, "video_port", 9000),
        metadata_port=int_or(insight, "metadata_port", 9100),
        save_dir=string_or(output, "save_dir"),
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
    try:
        payload = tensor.copy_payload_bytes()
    except Exception as exc:
        raise RuntimeError(f"capture_payload_failed err={exc}") from exc
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


def build_source_options(cfg: AppConfig, fps: int, width: int, height: int):
    opt = pyneat.RtspDecodedInputOptions()
    opt.url = cfg.rtsp_url
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


def output_caps_enabled(caps) -> bool:
    return caps.enable or caps.width > 0 or caps.height > 0 or caps.fps > 0


def build_encoded_source_graph(opt) -> pyneat.Graph:
    source = pyneat.Graph("rtsp_encoded_source")
    use_auto_caps = opt.auto_caps_from_stream and (
        opt.h264_fps <= 0 or opt.h264_width <= 0 or opt.h264_height <= 0
    )
    insert_queue = opt.insert_queue and not opt.sync_mode
    source.add(pyneat.nodes.rtsp_input(opt.url, opt.latency_ms, opt.tcp))
    if insert_queue:
        source.add(pyneat.nodes.queue())
    source.add(
        pyneat.nodes.h264_depacketize(
            payload_type=opt.payload_type,
            h264_parse_config_interval=opt.h264_parse_config_interval,
            h264_fps=opt.h264_fps,
            h264_width=opt.h264_width,
            h264_height=opt.h264_height,
            enforce_h264_caps=not use_auto_caps,
        )
    )
    if insert_queue:
        source.add(pyneat.nodes.queue())
    if use_auto_caps:
        source.add(
            pyneat.nodes.h264_caps_fixup(
                opt.fallback_h264_fps, opt.fallback_h264_width, opt.fallback_h264_height
            )
        )
    source.add(pyneat.nodes.output("encoded", pyneat.OutputOptions.every_frame(3)))
    return source


def h264_encoded_input_options():
    opt = pyneat.InputOptions()
    opt.payload_type = pyneat.PayloadType.Encoded
    opt.caps_override = (
        "video/x-h264,parsed=true,stream-format=(string)byte-stream,"
        "alignment=(string)au"
    )
    return opt


def build_decode_model_graph(input_name: str, output_name: str, opt, model) -> pyneat.Graph:
    decode = pyneat.Graph("decode_model")
    dec_w = opt.h264_width if opt.h264_width > 0 else opt.fallback_h264_width
    dec_h = opt.h264_height if opt.h264_height > 0 else opt.fallback_h264_height
    dec_fps = opt.h264_fps if opt.h264_fps > 0 else opt.fallback_h264_fps

    decode.add(pyneat.nodes.input(input_name, h264_encoded_input_options()))
    decode.add(
        pyneat.nodes.h264_decode(
            sima_allocator_type=opt.sima_allocator_type,
            out_format="NV12",
            decoder_name=opt.decoder_name,
            raw_output=opt.decoder_raw_output,
            next_element=opt.decoder_next_element,
            dec_width=dec_w,
            dec_height=dec_h,
            dec_fps=dec_fps,
        )
    )
    if opt.use_videoconvert:
        decode.add(pyneat.nodes.video_convert())
    if opt.use_videoscale:
        decode.add(pyneat.nodes.video_scale())
    if output_caps_enabled(opt.output_caps):
        decode.add(
            pyneat.nodes.caps_raw(
                "NV12",
                opt.output_caps.width,
                opt.output_caps.height,
                opt.output_caps.fps,
                opt.output_caps.memory,
            )
        )
    if opt.extra_fragment:
        decode.add(pyneat.nodes.custom(opt.extra_fragment))
    decode.add(model)
    decode.add(pyneat.nodes.output(output_name, pyneat.OutputOptions.every_frame(4)))
    return decode


def build_video_sender_graph(input_name: str, video_options) -> pyneat.Graph:
    video = pyneat.Graph("video_sender")
    video.add(pyneat.nodes.input(input_name, h264_encoded_input_options()))
    video.add(pyneat.groups.video_sender(video_options))
    return video


def build_model(cfg: AppConfig):
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


def build_run_options() -> pyneat.RunOptions:
    run_options = pyneat.RunOptions()
    run_options.preset = pyneat.RunPreset.Realtime
    run_options.queue_depth = 3
    run_options.overflow_policy = pyneat.OverflowPolicy.KeepLatest
    run_options.output_memory = pyneat.OutputMemory.ZeroCopy
    return run_options


def build_pipeline(cfg: AppConfig) -> PipelineRuntime:
    frame_w, frame_h, fps = probe_rtsp(cfg.rtsp_url)
    model = build_model(cfg)
    labels = load_labels(cfg.labels_path)

    source_options = build_source_options(cfg, fps, frame_w, frame_h)
    source_graph = build_encoded_source_graph(source_options)
    video_options = pyneat.VideoSenderOptions.h264_rtp_udp_from_encoded()
    video_options.host = cfg.insight_host
    video_options.channel = 0
    video_options.video_port_base = cfg.video_port

    decode_graph = build_decode_model_graph("encoded", "detections", source_options, model)
    video_graph = build_video_sender_graph("encoded", video_options)
    if cfg.profile:
        print(f"Source backend:\n{source_graph.describe_backend()}")
        print(f"Decode backend:\n{decode_graph.describe_backend()}")
        print(f"Video backend:\n{video_graph.describe_backend()}")

    source_run = source_graph.build(build_run_options())
    pending_encoded_sample = source_run.pull("encoded", 20000)
    if pending_encoded_sample is None:
        raise RuntimeError("timed out waiting for encoded RTSP frame")

    downstream_options = build_run_options()
    downstream_options.startup_preflight = False
    decode_run = decode_graph.build([pending_encoded_sample], options=downstream_options)
    video_run = video_graph.build([pending_encoded_sample], options=downstream_options)

    metadata_options = pyneat.MetadataSenderOptions()
    metadata_options.host = cfg.insight_host
    metadata_options.channel = 0
    metadata_options.metadata_port_base = cfg.metadata_port
    metadata_sender = pyneat.MetadataSender(metadata_options)

    print(
        f"rtsp={cfg.rtsp_url} stream={frame_w}x{frame_h}@{fps} "
        f"insight={cfg.insight_host} video={video_options.video_port} "
        f"metadata={metadata_sender.metadata_port()} channel=0"
    )
    return PipelineRuntime(
        model=model,
        source_graph=source_graph,
        source_run=source_run,
        decode_graph=decode_graph,
        decode_run=decode_run,
        video_graph=video_graph,
        video_run=video_run,
        pending_encoded_sample=pending_encoded_sample,
        metadata_sender=metadata_sender,
        labels=labels,
        frame_w=frame_w,
        frame_h=frame_h,
        video_port=video_options.video_port,
    )


def send_metadata(runtime: PipelineRuntime, sample, boxes: list[dict]) -> None:
    metadata_boxes = build_metadata_boxes(boxes, runtime.labels, runtime.frame_w, runtime.frame_h)
    frame_id = getattr(sample, "frame_id", -1)
    if frame_id is None or frame_id < 0:
        frame_id = 0
    runtime.metadata_sender.send_metadata(
        "object-detection",
        json.dumps({"objects": metadata_boxes}, separators=(",", ":")),
        int(time.time() * 1000),
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

    if tensor.is_i420():
        width = tensor_dim(tensor, "width")
        height = tensor_dim(tensor, "height")
        payload = np.frombuffer(tensor.copy_payload_bytes(), dtype=np.uint8)
        expected = width * height * 3 // 2
        if payload.size < expected:
            raise RuntimeError(f"I420 payload too small: {payload.size} < {expected}")
        i420 = payload[:expected].reshape((height * 3 // 2, width))
        return np.ascontiguousarray(cv2.cvtColor(i420, cv2.COLOR_YUV2BGR_I420))

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


def maybe_save_debug_frame(cfg: AppConfig, processed: int, sample, boxes: list[dict]) -> None:
    if not cfg.save_dir or cfg.save_every <= 0 or processed % cfg.save_every != 0:
        return

    frame_tensor = first_tensor_from_sample(sample)
    if frame_tensor is None:
        return

    frame = tensor_bgr_from_decoded(frame_tensor)
    draw_boxes(frame, boxes, cfg.min_score)
    out_path = Path(cfg.save_dir) / f"frame_{processed}.jpg"
    if not cv2.imwrite(str(out_path), frame):
        print(f"[warn] failed to write output frame: {out_path}", file=sys.stderr)


def pump_encoded_samples(
    runtime: PipelineRuntime,
    stop_event: threading.Event,
    errors: list[Exception],
) -> None:
    try:
        encoded_sample = runtime.pending_encoded_sample
        runtime.pending_encoded_sample = None
        while not stop_event.is_set():
            if encoded_sample is None:
                encoded_sample = runtime.source_run.pull("encoded", 200)
            if encoded_sample is None:
                if stop_event.is_set():
                    break
                last_error_fn = getattr(runtime.source_run, "last_error", None)
                last_error = last_error_fn() if callable(last_error_fn) else ""
                if last_error:
                    raise RuntimeError(f"source runtime error: {last_error}")
                continue
            if not runtime.decode_run.push("encoded", [encoded_sample]):
                if stop_event.is_set():
                    break
                raise RuntimeError("failed to push encoded sample to decode graph")
            if not runtime.video_run.push("encoded", [encoded_sample]):
                if stop_event.is_set():
                    break
                raise RuntimeError("failed to push encoded sample to video sender graph")
            encoded_sample = None
    except Exception as exc:
        errors.append(exc)
        stop_event.set()


def run_pipeline(runtime: PipelineRuntime, cfg: AppConfig) -> int:
    profile = ProfileWindow(cfg.profile, cfg.profile_interval)
    stop_event = threading.Event()
    pump_errors: list[Exception] = []
    pump = threading.Thread(
        target=pump_encoded_samples,
        args=(runtime, stop_event, pump_errors),
        name="encoded-pump",
        daemon=True,
    )
    processed = 0
    pump.start()
    try:
        while cfg.frames <= 0 or processed < cfg.frames:
            if pump_errors:
                raise pump_errors[0]
            pull_start = time_ms()
            detection_sample = runtime.decode_run.pull("detections", 20000)
            pull_end = time_ms()
            if detection_sample is None:
                if pump_errors:
                    raise pump_errors[0]
                print("[warn] timed out waiting for detections", file=sys.stderr)
                continue

            payload = extract_bbox_payload(detection_sample)
            boxes = parse_boxes_strict(payload, runtime.frame_w, runtime.frame_h, cfg.max_detections)

            metadata_start = time_ms()
            send_metadata(runtime, detection_sample, boxes)
            metadata_end = time_ms()

            processed += 1
            maybe_save_debug_frame(cfg, processed, detection_sample, boxes)
            profile.add(pull_end - pull_start, metadata_end - metadata_start, len(boxes))
    finally:
        stop_event.set()
        runtime.video_run.close()
        runtime.decode_run.close()
        runtime.source_run.close()
        pump.join(timeout=2.0)

    if pump_errors:
        raise pump_errors[0]
    profile.flush()
    print(f"processed={processed} video_sender={cfg.insight_host}:{runtime.video_port}")
    return processed


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        cfg = load_app_config(args.config)
        if args.validate_config_only:
            print(f"Config validated: {args.config}")
            return 0

        load_runtime_dependencies()
        if cfg.profile:
            os.environ.setdefault("SIMA_GST_ELEMENT_TIMINGS", "1")
            os.environ.setdefault("SIMA_GST_FLOW_DEBUG", "1")
            os.environ.setdefault("SIMA_GST_BOUNDARY_PROBES", "1")
        if cfg.save_dir:
            Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)

        runtime = build_pipeline(cfg)
        run_pipeline(runtime, cfg)
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"[ERR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
