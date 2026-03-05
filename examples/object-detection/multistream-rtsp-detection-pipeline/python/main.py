"""Multi-camera RTSP YOLOv8 detection demo using pyneat (manual box decode)."""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
import queue
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pyneat
import struct


DFL_BINS_16 = np.arange(16, dtype=np.float32)
DEFAULT_LABELS_FILE = Path(__file__).resolve().parents[1] / "common" / "coco_label.txt"


@dataclass(frozen=True)
class RtspProbe:
    width: int
    height: int
    fps: int

def class_color(cls_id: int) -> tuple[int, int, int]:
    # Stable deterministic color per class id.
    return (
        int((37 * cls_id + 17) % 256),
        int((97 * cls_id + 73) % 256),
        int((53 * cls_id + 191) % 256),
    )


def load_labels(path: str | None) -> dict[int, str]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    labels: dict[int, str] = {}
    with p.open("r", encoding="utf-8") as f:
        for idx, raw in enumerate(f):
            name = raw.strip()
            if name:
                labels[idx] = name
    return labels


def draw_boxes(frame: np.ndarray, boxes: list[dict], labels: dict[int, str]) -> np.ndarray:
    """Draw bounding boxes with class labels on a BGR frame."""
    for b in boxes:
        x1, y1 = int(b["x1"]), int(b["y1"])
        x2, y2 = int(b["x2"]), int(b["y2"])
        cls_id = b["class_id"]
        score = b["score"]
        color = class_color(cls_id)
        label = labels.get(cls_id, str(cls_id))
        text = f"{label} {score:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
        cv2.putText(frame, text, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return frame




def tensor_to_numpy(t: pyneat.Tensor) -> np.ndarray:
    return np.asarray(t.to_numpy(copy=True))


def iter_tensors(sample: pyneat.Sample):
    if sample.kind == pyneat.SampleKind.Tensor and sample.tensor is not None:
        yield sample.tensor
    for field in sample.fields:
        yield from iter_tensors(field)


def extract_bbox_payload(sample: pyneat.Sample) -> bytes | None:
    stack = [sample]
    while stack:
        s = stack.pop()
        stack.extend(reversed(list(s.fields)))
        if s.kind != pyneat.SampleKind.Tensor or s.tensor is None:
            continue
        fmt = (s.payload_tag or s.format or "").upper()
        if fmt and fmt != "BBOX":
            continue
        try:
            payload = s.tensor.copy_payload_bytes()
        except Exception:
            continue
        if payload:
            return payload
    return None


def parse_bbox_payload(payload: bytes, img_w: int, img_h: int) -> list[dict]:
    if len(payload) < 4:
        return []
    count = min(struct.unpack_from("<I", payload, 0)[0], (len(payload) - 4) // 24)
    out = []
    off = 4
    for _ in range(count):
        x, y, w, h, score, cls_id = struct.unpack_from("<iiiifi", payload, off)
        off += 24
        x1 = max(0.0, min(float(img_w), float(x)))
        y1 = max(0.0, min(float(img_h), float(y)))
        x2 = max(0.0, min(float(img_w), float(x + w)))
        y2 = max(0.0, min(float(img_h), float(y + h)))
        if x2 <= x1 or y2 <= y1:
            continue
        out.append(dict(x1=x1, y1=y1, x2=x2, y2=y2, score=float(score), class_id=int(cls_id)))
    return out


def probe_rtsp(url: str) -> RtspProbe:
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        raise RuntimeError(f"failed to open RTSP source for probing: {url}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = int(round(cap.get(cv2.CAP_PROP_FPS) or 0))
    cap.release()
    if width <= 0 or height <= 0:
        raise RuntimeError("failed to probe RTSP frame size")
    return RtspProbe(width=width, height=height, fps=max(0, fps))


def tensor_to_hwc_f32(t: pyneat.Tensor) -> np.ndarray:
    arr = tensor_to_numpy(t).astype(np.float32)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"unexpected tensor rank {arr.ndim}")
    return arr


def iou_xyxy(a, b) -> float:
    xx1 = max(a[0], b[0])
    yy1 = max(a[1], b[1])
    xx2 = min(a[2], b[2])
    yy2 = min(a[3], b[3])
    w = max(0.0, xx2 - xx1)
    h = max(0.0, yy2 - yy1)
    inter = w * h
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    den = area_a + area_b - inter
    return inter / den if den > 0 else 0.0


def decode_yolov8_boxes_from_sample(
    sample: pyneat.Sample,
    infer_size: int,
    img_w: int,
    img_h: int,
    min_score_logit: float,
    nms_iou: float,
    max_det: int,
) -> list[dict]:
    tensors = list(iter_tensors(sample))
    if len(tensors) < 6:
        raise ValueError(f"expected at least 6 tensors, got {len(tensors)}")
    regs = [tensor_to_hwc_f32(tensors[i]) for i in range(3)]
    clss = [tensor_to_hwc_f32(tensors[i]) for i in range(3, 6)]

    # YOLOv8 letterboxes the input: uniform scale to fit within infer_size x infer_size,
    # then pads the shorter dimension symmetrically.
    scale = min(infer_size / img_w, infer_size / img_h)
    pad_x = (infer_size - img_w * scale) / 2.0
    pad_y = (infer_size - img_h * scale) / 2.0

    boxes_all: list[np.ndarray] = []
    scores_all: list[np.ndarray] = []
    class_all: list[np.ndarray] = []
    for reg, cls in zip(regs, clss):
        h, w, c = reg.shape
        if c < 64:
            continue
        stride = infer_size / float(h)

        cls_flat = cls.reshape(-1, cls.shape[2])
        best_class = np.argmax(cls_flat, axis=1).astype(np.int32)
        flat_idx = np.arange(cls_flat.shape[0], dtype=np.int32)
        best_logit = cls_flat[flat_idx, best_class]

        # Sigmoid is monotonic, so threshold in logit domain first.
        keep = best_logit >= min_score_logit
        if not np.any(keep):
            continue

        kept_idx = flat_idx[keep]
        kept_class = best_class[keep]
        kept_logit = best_logit[keep]
        kept_score = 1.0 / (1.0 + np.exp(-np.clip(kept_logit, -60.0, 60.0)))

        reg_flat = reg.reshape(-1, 4, 16)[kept_idx]
        reg_flat = reg_flat - np.max(reg_flat, axis=2, keepdims=True)
        reg_exp = np.exp(reg_flat)
        reg_prob = reg_exp / np.sum(reg_exp, axis=2, keepdims=True)
        dists = np.sum(reg_prob * DFL_BINS_16[None, None, :], axis=2) * stride

        ys = kept_idx // w
        xs = kept_idx % w
        cx = (xs.astype(np.float32) + 0.5) * stride
        cy = (ys.astype(np.float32) + 0.5) * stride

        x1 = np.clip((cx - dists[:, 0] - pad_x) / scale, 0.0, float(img_w))
        y1 = np.clip((cy - dists[:, 1] - pad_y) / scale, 0.0, float(img_h))
        x2 = np.clip((cx + dists[:, 2] - pad_x) / scale, 0.0, float(img_w))
        y2 = np.clip((cy + dists[:, 3] - pad_y) / scale, 0.0, float(img_h))

        valid = (x2 > x1) & (y2 > y1)
        if not np.any(valid):
            continue

        boxes_all.append(np.stack((x1[valid], y1[valid], x2[valid], y2[valid]), axis=1))
        scores_all.append(kept_score[valid].astype(np.float32))
        class_all.append(kept_class[valid].astype(np.int32))

    if not boxes_all:
        return []

    boxes = np.concatenate(boxes_all, axis=0)
    scores = np.concatenate(scores_all, axis=0)
    class_ids = np.concatenate(class_all, axis=0)

    order = np.argsort(-scores, kind="stable")
    keep_idx: list[int] = []
    keep_by_class: dict[int, list[int]] = defaultdict(list)
    for i in order:
        ci = int(class_ids[i])
        suppressed = False
        for k in keep_by_class.get(ci, []):
            if iou_xyxy(boxes[k], boxes[i]) > nms_iou:
                suppressed = True
                break
        if suppressed:
            continue
        ii = int(i)
        keep_idx.append(ii)
        keep_by_class.setdefault(ci, []).append(ii)
        if len(keep_idx) >= max_det:
            break

    out: list[dict] = []
    for i in keep_idx:
        out.append(
            dict(
                x1=float(boxes[i, 0]),
                y1=float(boxes[i, 1]),
                x2=float(boxes[i, 2]),
                y2=float(boxes[i, 3]),
                score=float(scores[i]),
                class_id=int(class_ids[i]),
            )
        )
    return out


def build_rtsp_run(
    url: str,
    latency_ms: int,
    tcp: bool,
    out_w: int,
    out_h: int,
    stream_fps: int,
    sample_every: int,
    run_queue_depth: int,
    overflow_policy: pyneat.OverflowPolicy,
    output_memory: pyneat.OutputMemory,
    fallback_width: int = 0,
    fallback_height: int = 0,
):
    ro = pyneat.RtspDecodedInputOptions()
    ro.url = url
    ro.latency_ms = latency_ms
    ro.tcp = tcp
    ro.payload_type = 96
    ro.insert_queue = True
    ro.out_format = "BGR"
    ro.decoder_raw_output = False
    ro.auto_caps_from_stream = True
    ro.use_videoconvert = False
    ro.use_videoscale = True
    if fallback_width > 0:
        ro.fallback_h264_width = fallback_width
    if fallback_height > 0:
        ro.fallback_h264_height = fallback_height
    if stream_fps > 0:
        ro.fallback_h264_fps = stream_fps
    ro.output_caps.enable = True
    ro.output_caps.format = "BGR"
    ro.output_caps.width = out_w
    ro.output_caps.height = out_h
    if stream_fps > 0:
        ro.output_caps.fps = stream_fps
    ro.output_caps.memory = pyneat.CapsMemory.SystemMemory

    sess = pyneat.Session()
    sess.add(pyneat.groups.rtsp_decoded_input(ro))
    sess.add(pyneat.nodes.output(pyneat.OutputOptions.every_frame(max(1, sample_every))))
    run_opt = pyneat.RunOptions()
    run_opt.queue_depth = max(1, run_queue_depth)
    run_opt.overflow_policy = overflow_policy
    run_opt.output_memory = output_memory
    run = sess.build(run_opt)
    return sess, run


def parse_overflow_policy(value: str) -> pyneat.OverflowPolicy:
    v = value.strip().lower()
    if v == "block":
        return pyneat.OverflowPolicy.Block
    if v == "keep-latest":
        return pyneat.OverflowPolicy.KeepLatest
    if v == "drop-incoming":
        return pyneat.OverflowPolicy.DropIncoming
    raise ValueError(f"unsupported overflow policy: {value}")


def parse_output_memory(value: str) -> pyneat.OutputMemory:
    v = value.strip().lower()
    if v == "auto":
        return pyneat.OutputMemory.Auto
    if v == "zero-copy":
        return pyneat.OutputMemory.ZeroCopy
    if v == "owned":
        return pyneat.OutputMemory.Owned
    raise ValueError(f"unsupported output memory: {value}")


@dataclass
class StreamState:
    idx: int
    url: str
    probe: RtspProbe
    session: pyneat.Session
    run: pyneat.Run
    processed: int = 0
    pulled: int = 0
    producer_done: bool = False
    infer_done: bool = False


@dataclass
class FramePacket:
    stream_idx: int
    frame: np.ndarray
    pulled_ts: float


@dataclass
class ResultPacket:
    stream_idx: int
    frame: np.ndarray
    boxes: list[dict]
    pulled_ts: float


def put_keep_latest(q: "queue.Queue[object]", item: object) -> tuple[int, int]:
    dropped = 0
    while True:
        try:
            q.put_nowait(item)
            return dropped, q.qsize()
        except queue.Full:
            try:
                q.get_nowait()
                dropped += 1
            except queue.Empty:
                pass


@dataclass
class TimingAgg:
    count: int = 0
    total_s: float = 0.0
    max_s: float = 0.0

    def add(self, dt_s: float) -> None:
        self.count += 1
        self.total_s += dt_s
        if dt_s > self.max_s:
            self.max_s = dt_s

    def avg_ms(self) -> float:
        if self.count <= 0:
            return 0.0
        return (self.total_s * 1000.0) / float(self.count)

    def max_ms(self) -> float:
        if self.count <= 0:
            return 0.0
        return self.max_s * 1000.0


class ProfileTracker:
    def __init__(self, enabled: bool, every: int):
        self.enabled = enabled
        self.every = max(1, every)
        self.mu = threading.Lock()
        self.timing_window: dict[str, TimingAgg] = defaultdict(TimingAgg)
        self.window_drop: dict[str, int] = defaultdict(int)
        self.total_drop: dict[str, int] = defaultdict(int)
        self.window_hwm: dict[str, int] = defaultdict(int)
        self.total_hwm: dict[str, int] = defaultdict(int)
        self.started = False
        self.processing_start_ts = 0.0
        self.processing_start_done = 0
        self.window_start_ts = 0.0
        self.window_start_frame = 0

    def add_time(self, key: str, dt_s: float) -> None:
        if not self.enabled:
            return
        with self.mu:
            self.timing_window[key].add(dt_s)

    def note_queue(self, key: str, qsize: int, dropped: int = 0) -> None:
        if not self.enabled:
            return
        with self.mu:
            if qsize > self.window_hwm[key]:
                self.window_hwm[key] = qsize
            if qsize > self.total_hwm[key]:
                self.total_hwm[key] = qsize
            if dropped > 0:
                self.window_drop[key] += dropped
                self.total_drop[key] += dropped

    @staticmethod
    def stage_key(name: str, stream_idx: int) -> str:
        return f"{name}:s{stream_idx}"

    def mark_processing_started(self, done_frames: int) -> None:
        with self.mu:
            if self.started:
                return
            self.started = True
            self.processing_start_ts = time.perf_counter()
            self.processing_start_done = max(0, done_frames)
            self.window_start_ts = self.processing_start_ts
            self.window_start_frame = self.processing_start_done

    def processed_since_start(self, done_frames: int) -> int:
        with self.mu:
            if not self.started:
                return max(0, done_frames)
            return max(0, done_frames - self.processing_start_done)

    def elapsed_since_start_s(self, now_ts: float) -> float:
        with self.mu:
            if not self.started:
                return 0.0
            return max(0.0, now_ts - self.processing_start_ts)

    def maybe_report(
        self,
        done_frames: int,
        stream_snapshot: list[tuple[int, int, int]],
        frame_q_sizes: list[int],
        result_q_sizes: list[int],
    ) -> None:
        if not self.enabled:
            return
        if done_frames <= 0 or (done_frames % self.every) != 0:
            return

        with self.mu:
            if not self.started:
                return
            now = time.perf_counter()
            window_frames = done_frames - self.window_start_frame
            if window_frames <= 0:
                return
            window_elapsed = max(1e-6, now - self.window_start_ts)
            window_fps = window_frames / window_elapsed

            print("--------------------------")
            print(
                f"[PROGRESS] frames={done_frames} fps={window_fps:.2f} "
                f"(last {window_frames} in {window_elapsed:.2f}s)"
            )

            for idx, processed, pulled in stream_snapshot:
                print(f"[PROFILE][stream {idx}] processed={processed} pulled={pulled}")
                print("[PROFILE]   stage         avg(ms)   max(ms)   n")
                combined_stages = (
                    ("pull_call", "to_numpy", "rtsp_decode"),
                    ("model_run", "decode_boxes", "model"),
                    ("draw_boxes", "imwrite", "write"),
                )
                for key_a, key_b, pretty in combined_stages:
                    agg_a = self.timing_window.get(self.stage_key(key_a, idx))
                    agg_b = self.timing_window.get(self.stage_key(key_b, idx))
                    has_a = agg_a is not None and agg_a.count > 0
                    has_b = agg_b is not None and agg_b.count > 0
                    if not has_a and not has_b:
                        continue
                    total_s = 0.0
                    max_s = 0.0
                    count = 0
                    if has_a:
                        total_s += agg_a.total_s
                        max_s = max(max_s, agg_a.max_s)
                        count = agg_a.count
                    if has_b:
                        total_s += agg_b.total_s
                        max_s = max(max_s, agg_b.max_s)
                        count = max(count, agg_b.count)
                    avg_ms = (total_s * 1000.0) / float(count) if count > 0 else 0.0
                    max_ms = max_s * 1000.0
                    print(f"[PROFILE]   {pretty:<12} {avg_ms:8.1f} {max_ms:8.1f} {count:4d}")

            self.timing_window = defaultdict(TimingAgg)
            self.window_drop = defaultdict(int)
            self.window_hwm = defaultdict(int)
            self.window_start_ts = now
            self.window_start_frame = done_frames


@dataclass(frozen=True)
class AppConfig:
    rtsp_urls: list[str]
    model_path: str
    labels_file: str | None
    output_dir: Path
    frames: int
    debug: bool
    tcp: bool
    latency_ms: int
    width: int
    height: int
    fps: int
    sample_every: int
    save_every: int
    run_queue_depth: int
    overflow_policy: pyneat.OverflowPolicy
    output_memory: pyneat.OutputMemory
    pull_timeout_ms: int
    max_idle_ms: int
    reconnect_miss: int
    infer_size: int
    min_score: float
    min_score_logit: float
    nms_iou: float
    max_det: int
    model_timeout_ms: int
    model_queue_depth: int
    frame_queue: int
    result_queue: int
    profile: bool
    profile_every: int


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="YOLOv8 multi-RTSP demo (pyneat)")
    parser.add_argument("--rtsp", type=str, action="append", required=True,
                        help="RTSP URL (repeat for multiple streams)")
    parser.add_argument("--model", type=str, required=True, help="Path to yolo_v8s MPK tarball")
    parser.add_argument(
        "--labels-file",
        type=str,
        default=str(DEFAULT_LABELS_FILE),
        help="Labels file (one label per line). Defaults to common/coco_label.txt for this example",
    )
    parser.add_argument("--output", type=str, required=True, help="Output folder for annotated frames")
    parser.add_argument("--frames", type=int, default=100, help="Frames per stream to process")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--tcp", action="store_true")
    parser.add_argument("--latency-ms", type=int, default=200)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--sample-every", type=int, default=1, help="Use every Nth decoded frame")
    parser.add_argument("--save-every", type=int, default=10, help="Overlay/write every Nth processed frame")
    parser.add_argument("--run-queue-depth", type=int, default=4, help="Source run output queue depth")
    parser.add_argument(
        "--overflow-policy",
        type=str,
        default="keep-latest",
        choices=["block", "keep-latest", "drop-incoming"],
        help="Run output queue overflow policy",
    )
    parser.add_argument(
        "--output-memory",
        type=str,
        default="owned",
        choices=["auto", "zero-copy", "owned"],
        help="Run output memory mode",
    )
    parser.add_argument("--pull-timeout-ms", type=int, default=50, help="Per-pull timeout")
    parser.add_argument("--max-idle-ms", type=int, default=15000, help="Close stream after no frames for this duration")
    parser.add_argument(
        "--reconnect-miss",
        type=int,
        default=3,
        help="Consecutive pull misses while run is not running before reconnect",
    )
    parser.add_argument("--infer-size", type=int, default=640, help="Model input size for decode math")
    parser.add_argument("--min-score", type=float, default=0.52, help="Detection score threshold")
    parser.add_argument("--nms-iou", type=float, default=0.50, help="Class-wise NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=100, help="Max detections per frame")
    parser.add_argument("--model-timeout-ms", type=int, default=3000, help="Model run timeout")
    parser.add_argument("--model-queue-depth", type=int, default=4, help="Async model inference pipeline depth per stream")
    parser.add_argument("--frame-queue", type=int, default=64, help="Decode->infer queue size")
    parser.add_argument("--result-queue", type=int, default=64, help="Infer->overlay queue size")
    parser.add_argument("--profile", action="store_true", help="Enable timing/queue profiling logs")
    parser.add_argument("--profile-every", type=int, default=50, help="Emit profile logs every N processed frames")
    return parser


def parse_config(argv: list[str] | None = None) -> AppConfig:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if not (0.0 < args.min_score < 1.0):
        parser.error("--min-score must be in (0, 1)")
    if not (0.0 < args.nms_iou < 1.0):
        parser.error("--nms-iou must be in (0, 1)")
    if args.max_det <= 0:
        parser.error("--max-det must be > 0")
    if args.save_every <= 0:
        parser.error("--save-every must be > 0")
    return AppConfig(
        rtsp_urls=list(args.rtsp),
        model_path=args.model,
        labels_file=args.labels_file or None,
        output_dir=Path(args.output),
        frames=args.frames,
        debug=bool(args.debug),
        tcp=bool(args.tcp),
        latency_ms=args.latency_ms,
        width=args.width,
        height=args.height,
        fps=args.fps,
        sample_every=args.sample_every,
        save_every=args.save_every,
        run_queue_depth=args.run_queue_depth,
        overflow_policy=parse_overflow_policy(args.overflow_policy),
        output_memory=parse_output_memory(args.output_memory),
        pull_timeout_ms=args.pull_timeout_ms,
        max_idle_ms=args.max_idle_ms,
        reconnect_miss=args.reconnect_miss,
        infer_size=args.infer_size,
        min_score=args.min_score,
        min_score_logit=float(np.log(args.min_score / (1.0 - args.min_score))),
        nms_iou=args.nms_iou,
        max_det=args.max_det,
        model_timeout_ms=args.model_timeout_ms,
        model_queue_depth=args.model_queue_depth,
        frame_queue=args.frame_queue,
        result_queue=args.result_queue,
        profile=bool(args.profile),
        profile_every=args.profile_every,
    )


class PipelineApp:
    """Pipeline orchestrator.

    NEAT usage (Python):
    - RTSP decode runs are created per stream via pyneat.Session in build_rtsp_run(...)
    - Model objects are created per stream in _make_model(...)
    - Model runner sessions are lazily built in _infer_worker(...) on first frame
    """

    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.class_labels = load_labels(cfg.labels_file)
        self.out_root = cfg.output_dir
        self.streams: list[StreamState] = []
        self.out_dirs: list[Path] = []
        self.models: list[pyneat.Model] = []
        self.frame_queues: list[queue.Queue[FramePacket]] = []
        self.result_queues: list[queue.Queue[ResultPacket]] = []
        self.stats_mu = threading.Lock()
        self.model_build_mu = threading.Lock()
        self.stop_event = threading.Event()
        self.total_target = 0
        self.total_done = 0
        self.t0 = 0.0
        self.profiler = ProfileTracker(cfg.profile, cfg.profile_every)
        self.producer_threads: list[threading.Thread] = []
        self.infer_threads: list[threading.Thread] = []
        self.overlay_threads: list[threading.Thread] = []

    def _make_model(self) -> pyneat.Model:
        mopt = pyneat.ModelOptions()
        mopt.media_type = "video/x-raw"
        mopt.format = "BGR"
        mopt.input_max_width = self.cfg.width
        mopt.input_max_height = self.cfg.height
        mopt.input_max_depth = 3
        return pyneat.Model(self.cfg.model_path, mopt)

    def _setup(self) -> None:
        self.out_root.mkdir(parents=True, exist_ok=True)
        for i, url in enumerate(self.cfg.rtsp_urls):
            probe = probe_rtsp(url)
            sess, run = build_rtsp_run(
                url,
                self.cfg.latency_ms,
                self.cfg.tcp,
                self.cfg.width,
                self.cfg.height,
                probe.fps,
                self.cfg.sample_every,
                self.cfg.run_queue_depth,
                self.cfg.overflow_policy,
                self.cfg.output_memory,
                fallback_width=probe.width,
                fallback_height=probe.height,
            )
            self.streams.append(StreamState(i, url, probe, sess, run))
            stream_dir = self.out_root / f"stream_{i}"
            stream_dir.mkdir(parents=True, exist_ok=True)
            self.out_dirs.append(stream_dir)
            fps_part = f" @{probe.fps} fps" if probe.fps > 0 else ""
            print(
                f"[stream {i}] probed {probe.width}x{probe.height}{fps_part}; "
                f"started: {url} -> {stream_dir}/"
            )

        self.models = [self._make_model() for _ in self.streams]
        self.frame_queues = [
            queue.Queue(maxsize=max(1, self.cfg.frame_queue)) for _ in self.streams
        ]
        self.result_queues = [
            queue.Queue(maxsize=max(1, self.cfg.result_queue)) for _ in self.streams
        ]
        self.total_target = self.cfg.frames * len(self.streams)
        self.t0 = time.perf_counter()

    def _producer_worker(self, st: StreamState) -> None:
        try:
            miss_limit = max(1, self.cfg.max_idle_ms // max(1, self.cfg.pull_timeout_ms))
            miss_count = 0
            while not self.stop_event.is_set():
                with self.stats_mu:
                    if st.processed >= self.cfg.frames:
                        break
                t_pull0 = time.perf_counter()
                frame_t = st.run.pull_tensor(timeout_ms=self.cfg.pull_timeout_ms)
                self.profiler.add_time(
                    ProfileTracker.stage_key("pull_call", st.idx), time.perf_counter() - t_pull0
                )
                if frame_t is None:
                    miss_count += 1
                    running = True
                    try:
                        running = st.run.running()
                    except Exception:
                        running = False
                    if not running and miss_count >= max(1, self.cfg.reconnect_miss):
                        try:
                            st.run.close()
                        except Exception:
                            pass
                        try:
                            sess_new, run_new = build_rtsp_run(
                                st.url,
                                self.cfg.latency_ms,
                                self.cfg.tcp,
                                self.cfg.width,
                                self.cfg.height,
                                st.probe.fps,
                                self.cfg.sample_every,
                                self.cfg.run_queue_depth,
                                self.cfg.overflow_policy,
                                self.cfg.output_memory,
                                fallback_width=st.probe.width,
                                fallback_height=st.probe.height,
                            )
                            st.session = sess_new
                            st.run = run_new
                            miss_count = 0
                            print(f"[stream {st.idx}] reconnecting RTSP run")
                            continue
                        except Exception as e:
                            print(f"[stream {st.idx}] reconnect failed: {e}", file=sys.stderr)
                    if miss_count >= miss_limit:
                        print(
                            f"[stream {st.idx}] no frames for ~{self.cfg.max_idle_ms}ms; closing stream",
                            file=sys.stderr,
                        )
                        break
                    continue
                miss_count = 0
                pulled_ts = time.perf_counter()
                t_np0 = time.perf_counter()
                bgr = tensor_to_numpy(frame_t)
                if bgr.ndim == 4 and bgr.shape[0] == 1:
                    bgr = bgr[0]
                if bgr.dtype != np.uint8:
                    bgr = np.clip(bgr, 0, 255).astype(np.uint8)
                bgr = np.ascontiguousarray(bgr)
                self.profiler.add_time(
                    ProfileTracker.stage_key("to_numpy", st.idx), time.perf_counter() - t_np0
                )
                with self.stats_mu:
                    st.pulled += 1
                dropped, qsize = put_keep_latest(
                    self.frame_queues[st.idx], FramePacket(st.idx, bgr, pulled_ts)
                )
                self.profiler.note_queue(f"frame_q_{st.idx}", qsize, dropped)
                self.profiler.note_queue("frame_q", sum(q.qsize() for q in self.frame_queues), dropped)
        except Exception as e:
            print(f"[stream {st.idx}] pull error: {e} — closing stream", file=sys.stderr)
        finally:
            with self.stats_mu:
                st.producer_done = True

    def _infer_worker(self, stream_idx: int, model_local: pyneat.Model) -> None:
        frame_q = self.frame_queues[stream_idx]
        result_q = self.result_queues[stream_idx]
        runner = None
        pending: deque[FramePacket] = deque()
        model_qdepth = max(1, self.cfg.model_queue_depth)
        push_failures = 0
        max_push_failures = 3

        def get_frame(block_timeout: float = 0.05) -> FramePacket | None:
            try:
                t_wait0 = time.perf_counter()
                pkt = frame_q.get(timeout=block_timeout)
                self.profiler.add_time(
                    ProfileTracker.stage_key("infer_q_wait", stream_idx),
                    time.perf_counter() - t_wait0,
                )
                self.profiler.note_queue(f"frame_q_{stream_idx}", frame_q.qsize(), 0)
                self.profiler.note_queue("frame_q", sum(q.qsize() for q in self.frame_queues), 0)
                return pkt
            except queue.Empty:
                return None

        def process_result(sample: pyneat.Sample, pkt: FramePacket) -> None:
            t_dec0 = time.perf_counter()
            payload = extract_bbox_payload(sample)
            if payload:
                boxes = parse_bbox_payload(payload, pkt.frame.shape[1], pkt.frame.shape[0])
            else:
                boxes = decode_yolov8_boxes_from_sample(
                    sample,
                    self.cfg.infer_size,
                    pkt.frame.shape[1],
                    pkt.frame.shape[0],
                    self.cfg.min_score_logit,
                    self.cfg.nms_iou,
                    self.cfg.max_det,
                )
            self.profiler.add_time(
                ProfileTracker.stage_key("decode_boxes", stream_idx),
                time.perf_counter() - t_dec0,
            )
            dropped, qsize = put_keep_latest(
                result_q, ResultPacket(pkt.stream_idx, pkt.frame, boxes, pkt.pulled_ts)
            )
            self.profiler.note_queue(f"result_q_{stream_idx}", qsize, dropped)
            self.profiler.note_queue("result_q", sum(q.qsize() for q in self.result_queues), dropped)

        try:
            input_done = False
            while True:
                while len(pending) < model_qdepth and not input_done:
                    if self.stop_event.is_set() and frame_q.empty():
                        input_done = True
                        break
                    pkt = get_frame(block_timeout=0.01 if pending else 0.05)
                    if pkt is None:
                        with self.stats_mu:
                            producer_done = self.streams[stream_idx].producer_done
                        if producer_done and frame_q.empty():
                            input_done = True
                        break
                    with self.stats_mu:
                        if self.streams[pkt.stream_idx].processed >= self.cfg.frames:
                            input_done = True
                            break
                    try:
                        if runner is None:
                            sopt = pyneat.ModelSessionOptions()
                            ropt = pyneat.RunOptions()
                            ropt.queue_depth = model_qdepth
                            ropt.overflow_policy = pyneat.OverflowPolicy.Block
                            with self.model_build_mu:
                                runner = model_local.build(pkt.frame, sopt, ropt)
                        t_push0 = time.perf_counter()
                        ok = runner.push(pkt.frame)
                        self.profiler.add_time(
                            ProfileTracker.stage_key("model_push", stream_idx),
                            time.perf_counter() - t_push0,
                        )
                        if ok:
                            pending.append(pkt)
                            push_failures = 0
                        else:
                            print(f"[stream {stream_idx}] runner.push rejected frame", file=sys.stderr)
                    except Exception as e:
                        push_failures += 1
                        print(f"[stream {stream_idx}] push error ({push_failures}/{max_push_failures}): {e}", file=sys.stderr)
                        if push_failures >= max_push_failures:
                            print(f"[stream {stream_idx}] too many push failures; stopping", file=sys.stderr)
                            input_done = True
                            break

                if not pending:
                    if input_done:
                        break
                    continue

                try:
                    t_pull0 = time.perf_counter()
                    sample = runner.pull(timeout_ms=self.cfg.model_timeout_ms)
                    self.profiler.add_time(
                        ProfileTracker.stage_key("model_run", stream_idx),
                        time.perf_counter() - t_pull0,
                    )
                    if sample is not None:
                        pkt = pending.popleft()
                        process_result(sample, pkt)
                    elif input_done and not pending:
                        break
                except Exception as e:
                    print(f"[stream {stream_idx}] pull error: {e}", file=sys.stderr)
                    if pending:
                        pending.popleft()

            if runner is not None and pending:
                while pending:
                    try:
                        sample = runner.pull(timeout_ms=self.cfg.model_timeout_ms)
                        if sample is not None:
                            pkt = pending.popleft()
                            process_result(sample, pkt)
                        else:
                            break
                    except Exception as e:
                        print(f"[stream {stream_idx}] drain error: {e}", file=sys.stderr)
                        pending.popleft()
        finally:
            if runner is not None:
                try:
                    runner.close()
                except Exception:
                    pass
            with self.stats_mu:
                self.streams[stream_idx].infer_done = True

    def _overlay_worker(self, stream_idx: int) -> None:
        result_q = self.result_queues[stream_idx]
        while True:
            if self.stop_event.is_set() and result_q.empty():
                return
            try:
                t_wait0 = time.perf_counter()
                pkt = result_q.get(timeout=0.05)
                self.profiler.add_time(
                    ProfileTracker.stage_key("overlay_q_wait", stream_idx),
                    time.perf_counter() - t_wait0,
                )
                self.profiler.note_queue(f"result_q_{stream_idx}", result_q.qsize(), 0)
                self.profiler.note_queue("result_q", sum(q.qsize() for q in self.result_queues), 0)
            except queue.Empty:
                with self.stats_mu:
                    infer_done = self.streams[stream_idx].infer_done
                if infer_done and result_q.empty():
                    return
                continue

            st = self.streams[pkt.stream_idx]
            with self.stats_mu:
                if st.processed >= self.cfg.frames:
                    continue
                frame_idx = st.processed
                st.processed += 1
                done_before = self.total_done

            self.profiler.mark_processing_started(done_before)
            should_save = (frame_idx % self.cfg.save_every) == 0
            if should_save:
                t_draw0 = time.perf_counter()
                draw_boxes(pkt.frame, pkt.boxes, self.class_labels)
                self.profiler.add_time(
                    ProfileTracker.stage_key("draw_boxes", stream_idx),
                    time.perf_counter() - t_draw0,
                )
                out_path = self.out_dirs[pkt.stream_idx] / f"frame_{frame_idx:06d}.jpg"
                t_wr0 = time.perf_counter()
                cv2.imwrite(str(out_path), pkt.frame)
                self.profiler.add_time(
                    ProfileTracker.stage_key("imwrite", stream_idx),
                    time.perf_counter() - t_wr0,
                )
                self.profiler.add_time(
                    ProfileTracker.stage_key("end_to_end", stream_idx),
                    time.perf_counter() - pkt.pulled_ts,
                )

            if self.cfg.debug or ((frame_idx + 1) % 10 == 0):
                with self.stats_mu:
                    pulled_now = st.pulled
                print(
                    f"[stream {pkt.stream_idx}] frames={frame_idx + 1} "
                    f"pulled={pulled_now} det={len(pkt.boxes)}"
                )
            with self.stats_mu:
                self.total_done += 1
                done_now = self.total_done
            if self.cfg.profile and (done_now % max(1, self.cfg.profile_every) == 0):
                with self.stats_mu:
                    stream_snapshot = [(s.idx, s.processed, s.pulled) for s in self.streams]
                self.profiler.maybe_report(
                    done_now,
                    stream_snapshot,
                    [q.qsize() for q in self.frame_queues],
                    [q.qsize() for q in self.result_queues],
                )
            if done_now >= self.total_target:
                self.stop_event.set()
                return

    def _start_threads(self) -> None:
        for st in self.streams:
            t = threading.Thread(target=self._producer_worker, args=(st,), name=f"pull-{st.idx}", daemon=True)
            t.start()
            self.producer_threads.append(t)
        for idx, model_local in enumerate(self.models):
            t = threading.Thread(
                target=self._infer_worker,
                args=(idx, model_local),
                name=f"infer-{idx}",
                daemon=True,
            )
            t.start()
            self.infer_threads.append(t)
        for idx in range(len(self.streams)):
            t = threading.Thread(target=self._overlay_worker, args=(idx,), name=f"overlay-{idx}", daemon=True)
            t.start()
            self.overlay_threads.append(t)

    def _monitor_completion(self) -> None:
        while not self.stop_event.is_set():
            with self.stats_mu:
                done = self.total_done
                infers_done = all(s.infer_done for s in self.streams)
            if done >= self.total_target:
                self.stop_event.set()
                break
            if infers_done and all(q.empty() for q in self.result_queues):
                self.stop_event.set()
                break
            time.sleep(0.02)

    def _join_threads(self) -> None:
        for t in self.producer_threads:
            t.join()
        for t in self.infer_threads:
            t.join()
        for t in self.overlay_threads:
            t.join()

    def _print_summary_and_close(self) -> None:
        t_end = time.perf_counter()
        done_since_start = self.profiler.processed_since_start(self.total_done)
        elapsed_from_start = self.profiler.elapsed_since_start_s(t_end)
        elapsed = (
            max(1e-6, elapsed_from_start)
            if done_since_start > 0 and elapsed_from_start > 0.0
            else max(1e-6, t_end - self.t0)
        )
        avg_fps = done_since_start / elapsed
        print(f"Processed {self.total_done} outputs in {elapsed:.2f}s avg_fps={avg_fps:.2f}")
        for st in self.streams:
            print(f"stream[{st.idx}] processed={st.processed} pulled={st.pulled} saved to {self.out_dirs[st.idx]}/")
            st.run.close()

    def run(self) -> int:
        try:
            self._setup()
            self._start_threads()
            self._monitor_completion()
            self._join_threads()
            self._print_summary_and_close()
            return 0
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            for st in self.streams:
                try:
                    st.run.close()
                except Exception:
                    pass
            return 4


def main() -> int:
    cfg = parse_config()
    return PipelineApp(cfg).run()


if __name__ == "__main__":
    raise SystemExit(main())
