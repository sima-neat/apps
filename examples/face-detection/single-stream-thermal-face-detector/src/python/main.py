"""Single-camera RTSP yolov5s-face Insight example using pyneat.

Streams an RTSP source into a NEAT graph, runs yolov5s-face inference on the MLA,
and publishes results to a neat-insight viewer:

  RTSP decode (NV12) --> branch --> video_sender (H264 RTP/UDP -> Insight)
                               \\--> model (raw split heads) --> detections

The model archive emits six raw FP32 split heads (paired 18-channel box and
30-channel landmark heads at three pyramid levels). `decode_type` is left
Unspecified, so no fused BoxDecode runs -- the box + 5-landmark decode runs on
the host (APU) because the NEAT BBOX wire format carries no landmark slots.

Each frame's detections are published to Insight as a `pose-estimation` overlay
carrying the 5 named facial landmarks (eyes, nose, mouth corners). The Insight
viewer renders one metadata type per channel at a time, so the pipeline sends a
single type rather than competing overlays.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "common" / "config.yaml"

cv2 = None
np = None
pyneat = None

# The model was compiled for an 800x800 canvas (pyramid levels 100/50/25).
INFER_SIZE = 800
NUM_ANCHORS = 3
NUM_LANDMARKS = 5
BOX_CHANNELS = 18
LM_CHANNELS = 30

# Landmark names, in yolov5s-face output order. The Insight pose overlay draws a
# named dot per keypoint, and joins any keypoints whose names match a COCO body
# skeleton pair (nose/left_eye/right_eye/...). These names deliberately avoid the
# COCO joint names so no skeleton lines are drawn across the face.
LM_NAMES = ("eye_l", "eye_r", "nose_tip", "mouth_l", "mouth_r")

# yolov5s-face anchors / strides (fixed by the model architecture).
_STRIDES = (8, 16, 32)


def _anchors():
    return (
        np.array([[4.0, 5.0], [8.0, 10.0], [13.0, 16.0]], dtype=np.float32),
        np.array([[23.0, 29.0], [43.0, 55.0], [73.0, 105.0]], dtype=np.float32),
        np.array([[146.0, 217.0], [231.0, 300.0], [335.0, 433.0]], dtype=np.float32),
    )


@dataclass(frozen=True)
class AppConfig:
    model_path: str
    labels_path: Path
    rtsp_url: str
    latency_ms: int = 200
    tcp: bool = True
    frames: int = 0
    min_score: float = 0.25
    nms_iou: float = 0.45
    max_detections: int = 50
    profile: bool = False
    profile_interval: int = 100
    insight_host: str = "127.0.0.1"
    video_port: int = 9000
    metadata_port: int = 9100


@dataclass
class PipelineRuntime:
    model: object
    graph: object
    run: object
    metadata_sender: object
    labels: list
    frame_w: int
    frame_h: int
    video_port: int
    scale: float
    pad_l: int
    pad_t: int


class ProfileWindow:
    def __init__(self, enabled: bool, interval: int) -> None:
        self.enabled = enabled
        self.interval = interval
        self.frames = 0
        self.faces = 0
        self.start_ms = 0.0
        self.pull_ms = 0.0
        self.decode_ms = 0.0
        self.metadata_ms = 0.0

    def add(self, pull_ms: float, decode_ms: float, metadata_ms: float, face_count: int) -> None:
        if not self.enabled:
            return
        if self.frames == 0:
            self.start_ms = time_ms()
        self.frames += 1
        self.faces += face_count
        self.pull_ms += pull_ms
        self.decode_ms += decode_ms
        self.metadata_ms += metadata_ms
        if self.frames >= self.interval:
            self.flush()

    def flush(self) -> None:
        if not self.enabled or self.frames == 0:
            return
        elapsed = time_ms() - self.start_ms
        output_fps = self.frames * 1000.0 / elapsed if elapsed > 0.0 else 0.0
        print(
            f"[profile] frames={self.frames} output_fps={output_fps:.1f} "
            f"avg_pull_ms={self.pull_ms / self.frames:.2f} "
            f"avg_decode_ms={self.decode_ms / self.frames:.2f} "
            f"avg_metadata_ms={self.metadata_ms / self.frames:.2f} "
            f"avg_faces={self.faces / self.frames:.2f}",
            flush=True,
        )
        self.frames = 0
        self.faces = 0
        self.start_ms = 0.0
        self.pull_ms = 0.0
        self.decode_ms = 0.0
        self.metadata_ms = 0.0


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


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Single-camera RTSP yolov5s-face Insight example")
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
    default_labels = Path(__file__).resolve().parents[1] / "common" / "face_label.txt"

    cfg = AppConfig(
        model_path=string_or(model, "path"),
        labels_path=Path(string_or(model, "labels", str(default_labels))),
        rtsp_url=string_or(source, "rtsp_url"),
        latency_ms=int_or(source, "latency_ms", 200),
        tcp=bool_or(source, "tcp", True),
        frames=int_or(inference, "frames", 0),
        min_score=float_or(inference, "min_score", 0.25),
        nms_iou=float_or(inference, "nms_iou", 0.45),
        max_detections=int_or(inference, "max_detections", 50),
        profile=bool_or(runtime, "profile", False),
        profile_interval=int_or(runtime, "profile_interval", 100),
        insight_host=string_or(insight, "host"),
        video_port=int_or(insight, "video_port", 9000),
        metadata_port=int_or(insight, "metadata_port", 9100),
    )
    validate_config(cfg)
    return cfg


def load_labels(labels_path: Path) -> list:
    if not labels_path.is_file():
        raise RuntimeError(f"labels file does not exist: {labels_path}")
    labels = [line.strip() for line in labels_path.read_text(encoding="utf-8").splitlines()]
    labels = [label for label in labels if label]
    if not labels:
        raise RuntimeError(f"labels file is empty: {labels_path}")
    return labels


def letterbox_params(orig_w: int, orig_h: int, target_w: int, target_h: int):
    scale = min(target_w / orig_w, target_h / orig_h)
    nw = int(round(orig_w * scale))
    nh = int(round(orig_h * scale))
    pad_l = (target_w - nw) // 2
    pad_t = (target_h - nh) // 2
    return scale, pad_l, pad_t


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _nms_xyxy(boxes_xyxy, scores, iou_threshold: float, max_detections: int):
    x1, y1 = boxes_xyxy[:, 0], boxes_xyxy[:, 1]
    x2, y2 = boxes_xyxy[:, 2], boxes_xyxy[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        # Candidates are score-sorted, so the first max_detections survivors are
        # the top-scoring ones. Stopping here caps the published count and bounds
        # NMS cost on crowded / low-threshold inputs.
        if max_detections > 0 and len(keep) >= max_detections:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[1:][iou <= iou_threshold]
    return np.array(keep, dtype=np.intp)


def _decode_level(box_nhwc, lm_nhwc, stride: int, anchors, conf_threshold: float):
    _, ny, nx, _ = box_nhwc.shape
    box = box_nhwc.reshape(ny, nx, NUM_ANCHORS, 6)
    lm = lm_nhwc.reshape(ny, nx, NUM_ANCHORS, 10)

    obj = _sigmoid(box[..., 4])
    cls = _sigmoid(box[..., 5])
    scores_all = obj * cls
    mask = scores_all > conf_threshold
    if not mask.any():
        return None

    yy, xx, aa = np.nonzero(mask)
    n = yy.size

    sig_xywh = _sigmoid(box[yy, xx, aa, :4])
    aw = anchors[aa, 0]
    ah = anchors[aa, 1]
    xx_f = xx.astype(np.float32)
    yy_f = yy.astype(np.float32)
    cx = (sig_xywh[:, 0] * 2.0 - 0.5 + xx_f) * stride
    cy = (sig_xywh[:, 1] * 2.0 - 0.5 + yy_f) * stride
    bw = (sig_xywh[:, 2] * 2.0) ** 2 * aw
    bh = (sig_xywh[:, 3] * 2.0) ** 2 * ah

    boxes_xyxy = np.empty((n, 4), dtype=np.float32)
    boxes_xyxy[:, 0] = cx - bw * 0.5
    boxes_xyxy[:, 1] = cy - bh * 0.5
    boxes_xyxy[:, 2] = cx + bw * 0.5
    boxes_xyxy[:, 3] = cy + bh * 0.5

    lm_raw = lm[yy, xx, aa]
    lms = np.empty((n, NUM_LANDMARKS, 2), dtype=np.float32)
    lms[:, :, 0] = lm_raw[:, 0::2] * aw[:, None] + xx_f[:, None] * stride
    lms[:, :, 1] = lm_raw[:, 1::2] * ah[:, None] + yy_f[:, None] * stride

    return boxes_xyxy, scores_all[yy, xx, aa].astype(np.float32, copy=False), lms


def decode_yolov5face_split(outputs, conf_threshold: float, iou_threshold: float,
                            max_detections: int):
    pairs = {}
    for o in outputs:
        a = np.asarray(o)
        if a.ndim != 4 or a.shape[0] != 1:
            raise ValueError(f"Expected 4D [1,...] tensor, got shape {a.shape}")
        if a.shape[-1] in (BOX_CHANNELS, LM_CHANNELS):
            ch, ny, nx = a.shape[-1], a.shape[1], a.shape[2]
            arr = a.astype(np.float32, copy=False)
        elif a.shape[1] in (BOX_CHANNELS, LM_CHANNELS):
            ch, ny, nx = a.shape[1], a.shape[2], a.shape[3]
            arr = a.astype(np.float32, copy=False).transpose(0, 2, 3, 1)
        else:
            raise ValueError(
                f"Unrecognized split output shape {a.shape}; expected channel dim "
                f"{BOX_CHANNELS} or {LM_CHANNELS}")
        pairs.setdefault(max(ny, nx), {})[ch] = arr

    sizes = sorted(pairs.keys(), reverse=True)
    if len(sizes) != 3:
        raise ValueError(f"expected 3 pyramid levels, got {len(sizes)}")

    anchors = _anchors()
    out_boxes, out_scores, out_lms = [], [], []
    for lvl, size in enumerate(sizes):
        heads = pairs[size]
        if BOX_CHANNELS not in heads or LM_CHANNELS not in heads:
            raise ValueError(f"Level (size={size}) missing box or landmark head")
        result = _decode_level(
            heads[BOX_CHANNELS], heads[LM_CHANNELS], _STRIDES[lvl], anchors[lvl], conf_threshold)
        if result is None:
            continue
        b, s, l = result
        out_boxes.append(b)
        out_scores.append(s)
        out_lms.append(l)

    if not out_boxes:
        return (np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0, NUM_LANDMARKS, 2), dtype=np.float32))

    boxes = np.concatenate(out_boxes, axis=0)
    scores = np.concatenate(out_scores, axis=0)
    lms = np.concatenate(out_lms, axis=0)
    keep = _nms_xyxy(boxes, scores, iou_threshold, max_detections)
    return boxes[keep], scores[keep], lms[keep]


def unletterbox(boxes_xyxy, landmarks, scale, pad_l, pad_t, orig_w, orig_h):
    if len(boxes_xyxy) == 0:
        return boxes_xyxy, landmarks
    boxes = boxes_xyxy.copy()
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_l) / scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_t) / scale
    np.clip(boxes[:, [0, 2]], 0, orig_w, out=boxes[:, [0, 2]])
    np.clip(boxes[:, [1, 3]], 0, orig_h, out=boxes[:, [1, 3]])
    lms = landmarks.copy()
    lms[..., 0] = (lms[..., 0] - pad_l) / scale
    lms[..., 1] = (lms[..., 1] - pad_t) / scale
    return boxes, lms


def iter_tensors(sample):
    if sample.kind == pyneat.SampleKind.Tensor and sample.tensor is not None:
        yield sample.tensor
    elif sample.kind == pyneat.SampleKind.TensorSet:
        yield from sample.tensors
    for field in sample.fields:
        yield from iter_tensors(field)


def sample_to_outputs(sample):
    return [np.asarray(t.to_numpy(copy=False)) for t in iter_tensors(sample)]


def probe_rtsp(url: str):
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


def make_source_options(cfg: AppConfig, fps: int, width: int, height: int):
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


def make_model(cfg: AppConfig):
    opt = pyneat.ModelOptions()
    opt.preprocess.kind = pyneat.InputKind.Image
    opt.preprocess.enable = pyneat.AutoFlag.On
    # The RTSP decoder emits NV12; the model's on-device preproc converts NV12->RGB,
    # letterboxes to the 800x800 canvas, and normalizes /255 (COCO_YOLO preset).
    opt.preprocess.color_convert.input_format = pyneat.PreprocessColorFormat.NV12
    opt.preprocess.preset = pyneat.NormalizePreset.COCO_YOLO
    # decode_type stays Unspecified: the model emits raw split heads; the box +
    # landmark decode runs on the host below.
    return pyneat.Model(cfg.model_path, opt)


def build_pipeline(cfg: AppConfig) -> PipelineRuntime:
    frame_w, frame_h, fps = probe_rtsp(cfg.rtsp_url)
    model = make_model(cfg)
    labels = load_labels(cfg.labels_path)
    scale, pad_l, pad_t = letterbox_params(frame_w, frame_h, INFER_SIZE, INFER_SIZE)

    source = pyneat.groups.rtsp_decoded_input(make_source_options(cfg, fps, frame_w, frame_h))
    branch = pyneat.graphs.branch("source", ["video", "model"])

    video_options = pyneat.VideoSenderOptions.h264_rtp_udp_from_raw(frame_w, frame_h, fps)
    video_options.host = cfg.insight_host
    video_options.channel = 0
    video_options.video_port_base = cfg.video_port
    video_options.encoder.bitrate_kbps = 4000

    video_graph = pyneat.Graph("video")
    video_graph.connect(pyneat.nodes.input("video"), pyneat.groups.video_sender(video_options))

    model_graph = pyneat.Graph("model")
    model_graph.connect(pyneat.nodes.input("model"), model)

    detections_graph = pyneat.Graph("detections")
    detections_graph.add(pyneat.nodes.output("detections", pyneat.OutputOptions.every_frame(4)))

    graph = pyneat.Graph()
    live_link_options = pyneat.GraphLinkOptions()
    live_link_options.policy = pyneat.GraphLinkPolicy.RealtimeLatestByStream
    graph.connect(source, branch)
    graph.connect(branch, video_graph, live_link_options)
    graph.connect(branch, model_graph, live_link_options)
    graph.connect(model_graph, detections_graph)
    if cfg.profile:
        print(f"Backend:\n{graph.describe_backend()}")

    run_options = pyneat.RunOptions()
    run_options.preset = pyneat.RunPreset.Realtime
    run_options.queue_depth = 3
    run_options.overflow_policy = pyneat.OverflowPolicy.KeepLatest
    run_options.output_memory = pyneat.OutputMemory.ZeroCopy
    run = graph.build(run_options)

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
        graph=graph,
        run=run,
        metadata_sender=metadata_sender,
        labels=labels,
        frame_w=frame_w,
        frame_h=frame_h,
        video_port=video_options.video_port,
        scale=scale,
        pad_l=pad_l,
        pad_t=pad_t,
    )


def build_poses_json(landmarks, scores, label: str) -> str:
    poses = []
    for index, (lms, score) in enumerate(zip(landmarks, scores), start=1):
        keypoints = [
            {
                "name": LM_NAMES[k],
                "x": float(lms[k][0]),
                "y": float(lms[k][1]),
                "confidence": float(score),
            }
            for k in range(NUM_LANDMARKS)
        ]
        poses.append({"id": f"face_{index}", "label": label, "keypoints": keypoints})
    return json.dumps({"poses": poses}, separators=(",", ":"))


def send_metadata(runtime: PipelineRuntime, sample, scores, landmarks) -> None:
    label = runtime.labels[0]
    frame_id = getattr(sample, "frame_id", -1)
    if frame_id is None or frame_id < 0:
        frame_id = 0
    # Stamp metadata with the detection sample's source PTS (not wall-clock) so
    # Insight can align the landmarks with the matching video frame, which the
    # branch carries through the video_sender on the same PTS timeline.
    ts_ms = int(sample.pts_ns // 1_000_000) if sample.pts_ns >= 0 else -1
    runtime.metadata_sender.send_metadata(
        "pose-estimation",
        build_poses_json(landmarks, scores, label),
        ts_ms,
        str(frame_id),
    )


def run_pipeline(runtime: PipelineRuntime, cfg: AppConfig) -> int:
    profile = ProfileWindow(cfg.profile, cfg.profile_interval)
    processed = 0
    while cfg.frames <= 0 or processed < cfg.frames:
        pull_start = time_ms()
        sample = runtime.run.pull("detections", 20000)
        pull_end = time_ms()
        if sample is None:
            print("[warn] timed out waiting for detections", file=sys.stderr)
            continue

        outputs = sample_to_outputs(sample)
        boxes_xyxy, scores, landmarks = decode_yolov5face_split(
            outputs, cfg.min_score, cfg.nms_iou, cfg.max_detections)
        boxes_xyxy, landmarks = unletterbox(
            boxes_xyxy, landmarks, runtime.scale, runtime.pad_l, runtime.pad_t,
            runtime.frame_w, runtime.frame_h)
        decode_end = time_ms()

        send_metadata(runtime, sample, scores, landmarks)
        metadata_end = time_ms()

        processed += 1
        profile.add(pull_end - pull_start, decode_end - pull_end,
                    metadata_end - decode_end, len(boxes_xyxy))

    profile.flush()
    print(f"processed={processed} video_sender={cfg.insight_host}:{runtime.video_port}")
    return processed


def main(argv=None) -> int:
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

        runtime = build_pipeline(cfg)
        try:
            run_pipeline(runtime, cfg)
        finally:
            runtime.run.close()
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"[ERR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
