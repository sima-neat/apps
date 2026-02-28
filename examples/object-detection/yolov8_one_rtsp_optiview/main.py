"""Single-camera RTSP YOLOv8 OptiView example using pyneat.

This mirrors the intent of the C++ reference sample in the same folder:

- pull one decoded RTSP stream
- run one YOLOv8 detector
- publish H.264 video plus detection JSON to OptiView

The implementation keeps those responsibilities loosely separated so the main
runtime path is easy to reason about:

1. RTSP probe/build
2. YOLO inference and box extraction
3. OptiView video/JSON publishing
"""

from __future__ import annotations

import argparse
import json
import math
import os
import socket
import struct
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pyneat as pn


INFER_SIZE = 640
MIN_SCORE = 0.52
NMS_IOU = 0.50
MAX_DET = 100
DEFAULT_FPS = 30

# Standard COCO class order for the bundled YOLOv8 model used by this sample.
COCO80_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed",
    "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]

DFL_BINS_16 = np.arange(16, dtype=np.float32)


def tensor_to_numpy(tensor: pn.Tensor) -> np.ndarray:
    """Copy a pyneat tensor into a NumPy array owned by Python."""
    return np.asarray(tensor.to_numpy(copy=True))


def tensor_bgr_from_decoded(tensor: pn.Tensor) -> np.ndarray:
    """Normalize decoded output into a writable HWC uint8 BGR frame."""
    arr = tensor_to_numpy(tensor)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"unexpected decoded tensor shape {arr.shape}")
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def iter_tensors(sample: pn.Sample):
    """Depth-first walk over all tensor leaves in a pyneat sample tree."""
    if sample.kind == pn.SampleKind.Tensor and sample.tensor is not None:
        yield sample.tensor
    for field in sample.fields:
        yield from iter_tensors(field)


def extract_bbox_payload(sample: pn.Sample) -> bytes | None:
    """Prefer the runtime's pre-decoded BBOX payload when the model emits it.

    Some YOLO pipelines already attach a compact BBOX payload. When that exists
    it is more reliable and cheaper to parse than re-decoding the raw YOLO head
    tensors in Python.
    """
    stack = [sample]
    while stack:
        current = stack.pop()
        stack.extend(reversed(list(current.fields)))
        if current.kind != pn.SampleKind.Tensor or current.tensor is None:
            continue
        fmt = (current.payload_tag or current.format or "").upper()
        if fmt and fmt != "BBOX":
            continue
        try:
            payload = current.tensor.copy_payload_bytes()
        except Exception:
            continue
        if payload:
            return payload
    return None


def parse_bbox_payload(payload: bytes, img_w: int, img_h: int) -> list[dict]:
    """Decode the packed BBOX payload format used by NEAT samples."""
    if len(payload) < 4:
        return []
    # Payload layout:
    #   uint32 count
    #   repeated { int32 x, int32 y, int32 w, int32 h, float score, int32 class_id }
    #
    # Guard the parsed count against truncated payloads so we never read past
    # the actual buffer contents.
    count = min(struct.unpack_from("<I", payload, 0)[0], (len(payload) - 4) // 24)
    boxes = []
    off = 4
    for _ in range(count):
        x, y, w, h, score, cls_id = struct.unpack_from("<iiiifi", payload, off)
        off += 24
        # Clamp every box back into the decoded frame. This keeps OptiView JSON
        # consistent even if the payload contains slightly out-of-bounds values.
        x1 = max(0.0, min(float(img_w), float(x)))
        y1 = max(0.0, min(float(img_h), float(y)))
        x2 = max(0.0, min(float(img_w), float(x + w)))
        y2 = max(0.0, min(float(img_h), float(y + h)))
        # Skip degenerate boxes after clamping so downstream consumers only see
        # valid xyxy coordinates.
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append(
            {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "score": float(score),
                "class_id": int(cls_id),
            }
        )
    return boxes


def tensor_to_hwc_f32(tensor: pn.Tensor) -> np.ndarray:
    """Convert a model output tensor into HWC float32 for Python-side decoding."""
    arr = tensor_to_numpy(tensor).astype(np.float32)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"unexpected tensor rank {arr.ndim}")
    return arr


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection-over-union helper for per-class NMS."""
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
    sample: pn.Sample,
    infer_size: int,
    img_w: int,
    img_h: int,
    min_score: float,
    nms_iou: float,
    max_det: int,
) -> list[dict]:
    """Fallback YOLOv8 decode path used when no packed BBOX payload exists.

    The sample accepts the first six output tensors as the three regression
    heads plus the three classification heads, then applies:

    - DFL distance decoding
    - undo of the model letterbox transform
    - per-class NMS
    """
    tensors = list(iter_tensors(sample))
    if len(tensors) < 6:
        raise ValueError(f"expected at least 6 tensors, got {len(tensors)}")
    regs = [tensor_to_hwc_f32(tensors[i]) for i in range(3)]
    clss = [tensor_to_hwc_f32(tensors[i]) for i in range(3, 6)]

    scale = min(infer_size / img_w, infer_size / img_h)
    pad_x = (infer_size - img_w * scale) / 2.0
    pad_y = (infer_size - img_h * scale) / 2.0
    min_score_logit = math.log(min_score / (1.0 - min_score))

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
    keep_by_class: dict[int, list[int]] = {}
    for i in order:
        cls_id = int(class_ids[i])
        suppressed = False
        for kept in keep_by_class.get(cls_id, []):
            if iou_xyxy(boxes[kept], boxes[i]) > nms_iou:
                suppressed = True
                break
        if suppressed:
            continue
        idx = int(i)
        keep_idx.append(idx)
        keep_by_class.setdefault(cls_id, []).append(idx)
        if len(keep_idx) >= max_det:
            break

    return [
        {
            "x1": float(boxes[i, 0]),
            "y1": float(boxes[i, 1]),
            "x2": float(boxes[i, 2]),
            "y2": float(boxes[i, 3]),
            "score": float(scores[i]),
            "class_id": int(class_ids[i]),
        }
        for i in keep_idx
    ]


def probe_rtsp(url: str) -> tuple[int, int, int]:
    """Probe the stream once so the rest of the pipeline uses real dimensions.

    The Python sample keeps this step explicit instead of hardcoding 640x480,
    which makes the output path behave correctly for streams such as 720p.
    """
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
        fps = DEFAULT_FPS
    return width, height, fps


def build_rtsp_run(
    url: str,
    width: int,
    height: int,
    fps: int,
    latency_ms: int,
    tcp: bool,
) -> tuple[pn.Session, pn.Run]:
    """Build a decoded RTSP input session that yields BGR frames to Python."""
    ro = pn.RtspDecodedInputOptions()
    ro.url = url
    ro.latency_ms = latency_ms
    ro.tcp = tcp
    ro.payload_type = 96
    ro.insert_queue = True
    ro.out_format = "BGR"
    ro.decoder_raw_output = False
    ro.use_videoconvert = False
    ro.use_videoscale = True
    ro.output_caps.enable = True
    ro.output_caps.format = "BGR"
    ro.output_caps.width = width
    ro.output_caps.height = height
    ro.output_caps.fps = fps
    ro.output_caps.memory = pn.CapsMemory.SystemMemory

    sess = pn.Session()
    sess.add(pn.groups.rtsp_decoded_input(ro))
    sess.add(pn.nodes.output(pn.OutputOptions.every_frame(1)))

    run_opt = pn.RunOptions()
    run_opt.queue_depth = 4
    run_opt.overflow_policy = pn.OverflowPolicy.Block
    run_opt.output_memory = pn.OutputMemory.Owned
    run = sess.build(run_opt)
    return sess, run


def build_model(model_path: str, width: int, height: int) -> pn.Model:
    """Create the YOLO model with input bounds derived from the live stream."""
    opt = pn.ModelOptions()
    opt.media_type = "video/x-raw"
    opt.format = "BGR"
    opt.input_max_width = width
    opt.input_max_height = height
    opt.input_max_depth = 3
    return pn.Model(model_path, opt)


def build_udp_video_writer(host: str, port: int, width: int, height: int, fps: int) -> cv2.VideoWriter:
    """Open a GStreamer-backed RTP/H.264 UDP writer for the OptiView video port.

    OptiView expects video on its UDP video channel, so the Python version uses
    OpenCV's GStreamer backend to push frames into a lightweight H.264 RTP path.
    """
    pipeline = (
        "appsrc ! "
        "videoconvert ! "
        "video/x-raw,format=I420 ! "
        f"x264enc tune=zerolatency speed-preset=ultrafast bitrate=4000 key-int-max={max(1, fps)} ! "
        "h264parse ! "
        "rtph264pay pt=96 config-interval=1 ! "
        f'udpsink host={host} port={port} sync=false async=false'
    )
    writer = cv2.VideoWriter(pipeline, cv2.CAP_GSTREAMER, 0, float(max(1, fps)), (width, height), True)
    if not writer.isOpened():
        raise RuntimeError("failed to open UDP H.264 writer; verify OpenCV GStreamer and x264enc")
    return writer


def make_optiview_json(timestamp_ms: int, frame_id: str, boxes: list[dict]) -> str:
    """Build the OptiView object-detection JSON side-channel payload."""
    objects = []
    for idx, box in enumerate(boxes, start=1):
        cls_id = int(box["class_id"])
        label = COCO80_NAMES[cls_id] if 0 <= cls_id < len(COCO80_NAMES) else "Unknown"
        objects.append(
            {
                "id": f"obj_{idx}",
                "label": label,
                "confidence": float(box["score"]),
                "bbox": [
                    float(box["x1"]),
                    float(box["y1"]),
                    float(box["x2"] - box["x1"]),
                    float(box["y2"] - box["y1"]),
                ],
            }
        )
    return json.dumps(
        {
            "type": "object-detection",
            "timestamp": int(timestamp_ms),
            "frame_id": frame_id,
            "data": {"objects": objects},
        }
    )


def parse_args() -> argparse.Namespace:
    """Expose only the small set of controls needed for this reference flow."""
    parser = argparse.ArgumentParser(description="Single-camera RTSP YOLOv8 OptiView example")
    parser.add_argument("--rtsp", required=True, help="RTSP URL")
    parser.add_argument("--model", "--mpk", dest="model", default="", help="Path to YOLOv8 MPK tarball")
    parser.add_argument("--frames", type=int, default=0, help="Number of frames to process (0 = run forever)")
    parser.add_argument("--optiview-host", default="127.0.0.1", help="OptiView host")
    parser.add_argument("--optiview-video-port", type=int, default=9000, help="OptiView UDP video port")
    parser.add_argument("--optiview-json-port", type=int, default=9100, help="OptiView UDP JSON port")
    parser.add_argument("--latency-ms", type=int, default=200, help="RTSP latency in milliseconds")
    parser.add_argument("--udp", action="store_true", help="Use UDP RTSP transport instead of TCP")
    parser.add_argument("--debug", action="store_true", help="Enable timing prints")
    return parser.parse_args()


def resolve_yolov8s_model(root: Path) -> str:
    """Mirror the C++ sample's local-first model lookup strategy.

    Resolution order:

    1. explicit environment override
    2. local/common modelzoo directories
    3. `sima-cli modelzoo get yolo_v8s`
    """
    env_path = os.environ.get("SIMA_YOLO_TAR", "")
    if env_path and Path(env_path).exists():
        return env_path

    tmp_dir = root / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_tar = tmp_dir / "yolo_v8s_mpk.tar.gz"
    direct_tar = root / "yolo_v8s_mpk.tar.gz"
    if direct_tar.exists():
        return str(direct_tar)
    if tmp_tar.exists():
        return str(tmp_tar)

    home = Path.home()
    search_dirs = [
        root / "models",
        root,
        Path.cwd(),
        root / "tmp",
        home / ".simaai",
        home / ".simaai" / "modelzoo",
        home / ".sima" / "modelzoo",
        Path("/data/simaai/modelzoo"),
    ]
    names = [
        "yolo_v8s_mpk.tar.gz",
        "yolo-v8s_mpk.tar.gz",
        "yolov8s_mpk.tar.gz",
        "yolov8_s_mpk.tar.gz",
    ]
    for directory in search_dirs:
        for name in names:
            candidate = directory / name
            if candidate.exists():
                return str(candidate)

    try:
        subprocess.run(["sima-cli", "modelzoo", "get", "yolo_v8s"], check=True)
    except Exception:
        return ""

    if tmp_tar.exists():
        return str(tmp_tar)
    for directory in search_dirs:
        for name in names:
            candidate = directory / name
            if candidate.exists():
                return str(candidate)
    return ""


def main() -> int:
    args = parse_args()
    model_path = args.model or resolve_yolov8s_model(Path.cwd())
    if not model_path or not Path(model_path).is_file():
        print("Failed to locate yolo_v8s MPK tarball.", file=sys.stderr)
        print("Set --model/--mpk or run 'sima-cli modelzoo get yolo_v8s'.", file=sys.stderr)
        return 2

    rtsp_session = None
    rtsp_run = None
    writer = None
    json_sock = None
    try:
        # Probe first so decode, inference, and UDP output all agree on the
        # same live frame dimensions.
        frame_w, frame_h, fps = probe_rtsp(args.rtsp)
        print(f"[init] probed RTSP decode dims {frame_w}x{frame_h}")

        # The runtime is intentionally simple:
        # RTSP decode -> Python frame -> model.run() -> OptiView outputs.
        model = build_model(model_path, frame_w, frame_h)
        rtsp_session, rtsp_run = build_rtsp_run(
            args.rtsp,
            frame_w,
            frame_h,
            fps,
            args.latency_ms,
            tcp=not args.udp,
        )
        writer = build_udp_video_writer(
            args.optiview_host, args.optiview_video_port, frame_w, frame_h, fps
        )
        json_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        print(f"optiview host={args.optiview_host} video_port={args.optiview_video_port} "
              f"json_port={args.optiview_json_port} channel=0")

        processed = 0
        started = time.perf_counter()
        while args.frames <= 0 or processed < args.frames:
            # Pull one decoded frame from the RTSP session.
            t_pull0 = time.perf_counter()
            tensor = rtsp_run.pull_tensor(timeout_ms=5000)
            t_pull1 = time.perf_counter()
            if tensor is None:
                print("RTSP pull timed out / stream closed", file=sys.stderr)
                break

            frame = tensor_bgr_from_decoded(tensor)
            infer_input = frame

            # Run synchronous YOLO inference on the current frame.
            t_inf0 = time.perf_counter()
            result = model.run(infer_input, timeout_ms=5000)
            t_inf1 = time.perf_counter()

            # Prefer NEAT's packed bbox payload when available; otherwise decode
            # the YOLO heads in Python.
            payload = extract_bbox_payload(result)
            if payload:
                boxes = parse_bbox_payload(payload, frame.shape[1], frame.shape[0])
            else:
                boxes = decode_yolov8_boxes_from_sample(
                    result,
                    INFER_SIZE,
                    frame.shape[1],
                    frame.shape[0],
                    MIN_SCORE,
                    NMS_IOU,
                    MAX_DET,
                )

            # Publish the original video frame and the detection side-channel
            # independently, matching the OptiView contract used by the C++ sample.
            writer.write(frame)
            fid = str(processed)
            payload_json = make_optiview_json(int(time.time() * 1000), fid, boxes)
            json_sock.sendto(payload_json.encode("utf-8"), (args.optiview_host, args.optiview_json_port))

            processed += 1
            if args.debug and (processed <= 5 or processed % 30 == 0):
                print(
                    f"[debug] frame={processed} pull_ms={(t_pull1 - t_pull0) * 1000.0:.2f} "
                    f"infer_ms={(t_inf1 - t_inf0) * 1000.0:.2f} boxes={len(boxes)}"
                )

        elapsed = max(time.perf_counter() - started, 1e-6)
        print(f"processed={processed} fps={processed / elapsed:.2f} "
              f"udp={args.optiview_host}:{args.optiview_video_port}")
        return 0 if processed > 0 else 3
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    finally:
        # Clean shutdown matters on the DevKit because OpenCV/GStreamer and the
        # RTSP runtime otherwise tend to hold resources after exceptions.
        try:
            if writer is not None:
                writer.release()
        except Exception:
            pass
        try:
            if rtsp_run is not None:
                rtsp_run.close()
        except Exception:
            pass
        try:
            if json_sock is not None:
                json_sock.close()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
