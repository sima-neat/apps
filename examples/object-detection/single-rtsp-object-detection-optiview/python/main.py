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
from dataclasses import dataclass
import glob
import json
import os
import socket
import struct
import subprocess
import sys
import time
from pathlib import Path

# Prefer system OpenCV (built with GStreamer) when running inside a venv.
for p in glob.glob("/usr/lib/python3*/dist-packages"):
    if p not in sys.path:
        sys.path.insert(0, p)

import cv2
import numpy as np
import pyneat


MIN_SCORE = 0.52
NMS_IOU = 0.50
MAX_DET = 100
DEFAULT_FPS = 30
SOURCE_RUN_QUEUE_DEPTH = 4

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

@dataclass(frozen=True)
class AppConfig:
    rtsp: str
    model: str
    frames: int
    optiview_host: str
    optiview_video_port: int
    optiview_json_port: int
    latency_ms: int
    udp: bool
    debug: bool


def tensor_to_numpy(tensor: pyneat.Tensor) -> np.ndarray:
    """Copy a pyneat tensor into a NumPy array owned by Python."""
    return np.asarray(tensor.to_numpy(copy=True))


def tensor_dim(tensor: pyneat.Tensor, name: str) -> int:
    value = getattr(tensor, name)
    return int(value() if callable(value) else value)


def tensor_bgr_from_decoded(tensor: pyneat.Tensor) -> np.ndarray:
    """Normalize decoded output into a writable HWC uint8 BGR frame."""
    if tensor.is_nv12():
        width = tensor_dim(tensor, "width")
        height = tensor_dim(tensor, "height")
        payload = np.frombuffer(tensor.copy_payload_bytes(), dtype=np.uint8)
        expected = width * height * 3 // 2
        if payload.size < expected:
            raise ValueError(f"NV12 payload too small: {payload.size} < {expected}")
        nv12 = payload[:expected].reshape((height * 3 // 2, width))
        return np.ascontiguousarray(cv2.cvtColor(nv12, cv2.COLOR_YUV2BGR_NV12))

    if tensor.is_i420():
        width = tensor_dim(tensor, "width")
        height = tensor_dim(tensor, "height")
        payload = np.frombuffer(tensor.copy_payload_bytes(), dtype=np.uint8)
        expected = width * height * 3 // 2
        if payload.size < expected:
            raise ValueError(f"I420 payload too small: {payload.size} < {expected}")
        i420 = payload[:expected].reshape((height * 3 // 2, width))
        return np.ascontiguousarray(cv2.cvtColor(i420, cv2.COLOR_YUV2BGR_I420))

    arr = tensor_to_numpy(tensor)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"unexpected decoded tensor shape {arr.shape}")
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def tensor_from_bgr_frame(frame: np.ndarray) -> pyneat.Tensor:
    """Create an EV74 BGR image tensor for model-managed YOLO inference."""
    return pyneat.Tensor.from_numpy(
        np.ascontiguousarray(frame),
        copy=True,
        image_format=pyneat.PixelFormat.BGR,
        memory=pyneat.TensorMemory.EV74,
    )


def tensor_from_rgb_frame(frame: np.ndarray) -> pyneat.Tensor:
    """Create an EV74 RGB image tensor for the OptiView video encoder path."""
    return pyneat.Tensor.from_numpy(
        np.ascontiguousarray(frame),
        copy=True,
        image_format=pyneat.PixelFormat.RGB,
        memory=pyneat.TensorMemory.EV74,
    )


def is_tensor_like(value) -> bool:
    return hasattr(value, "copy_payload_bytes") and hasattr(value, "to_numpy")


def is_sample_like(value) -> bool:
    return hasattr(value, "kind") and hasattr(value, "fields")


def extract_bbox_payload_from_tensors(tensors) -> bytes | None:
    for tensor in tensors:
        try:
            payload = tensor.copy_payload_bytes()
        except Exception:
            continue
        if payload:
            return payload
    return None


def extract_bbox_payload(result) -> bytes | None:
    """Prefer the runtime's pre-decoded BBOX payload when the model emits it.

    Some YOLO pipelines already attach a compact BBOX payload. When that exists
    it is more reliable and cheaper to parse than re-decoding the raw YOLO head
    tensors in Python.
    """
    if isinstance(result, (list, tuple)) and all(is_tensor_like(item) for item in result):
        return extract_bbox_payload_from_tensors(result)

    if not is_sample_like(result):
        return None

    stack = [result]
    while stack:
        current = stack.pop()
        stack.extend(reversed(list(current.fields)))
        if current.kind == pyneat.SampleKind.TensorSet:
            payload = extract_bbox_payload_from_tensors(current.tensors)
            if payload:
                return payload
            continue
        if current.kind != pyneat.SampleKind.Tensor or current.tensor is None:
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
) -> tuple[pyneat.Session, pyneat.Run]:
    """Build a decoded RTSP input session that yields decoded frame tensors."""
    ro = pyneat.RtspDecodedInputOptions()
    ro.url = url
    ro.latency_ms = latency_ms
    ro.tcp = tcp
    ro.payload_type = 96
    ro.insert_queue = True
    ro.auto_caps_from_stream = True
    ro.fallback_h264_width = width
    ro.fallback_h264_height = height
    ro.fallback_h264_fps = fps
    ro.sima_allocator_type = 2
    ro.decoder_raw_output = False
    ro.use_videoconvert = False
    ro.use_videoscale = True
    ro.output_caps.enable = True
    ro.output_caps.width = width
    ro.output_caps.height = height
    ro.output_caps.fps = fps
    ro.output_caps.memory = pyneat.CapsMemory.SystemMemory

    sess = pyneat.Session()
    sess.add(pyneat.groups.rtsp_decoded_input(ro))
    sess.add(pyneat.nodes.output(pyneat.OutputOptions.every_frame(1)))

    run_opt = pyneat.RunOptions()
    run_opt.queue_depth = SOURCE_RUN_QUEUE_DEPTH
    run_opt.overflow_policy = pyneat.OverflowPolicy.KeepLatest
    run_opt.output_memory = pyneat.OutputMemory.Owned
    run = sess.build(run_opt)
    return sess, run


def build_model(model_path: str, width: int, height: int) -> pyneat.Model:
    """Create the YOLO model with input bounds derived from the live stream."""
    opt = pyneat.ModelOptions()
    opt.preprocess.kind = pyneat.InputKind.Image
    opt.preprocess.enable = pyneat.AutoFlag.On
    opt.preprocess.color_convert.input_format = pyneat.PreprocessColorFormat.BGR
    opt.preprocess.preset = pyneat.NormalizePreset.COCO_YOLO
    opt.decode_type = pyneat.BoxDecodeType.YoloV8
    opt.score_threshold = MIN_SCORE
    opt.nms_iou_threshold = NMS_IOU
    opt.top_k = MAX_DET
    opt.boxdecode_original_width = width
    opt.boxdecode_original_height = height
    return pyneat.Model(model_path, opt)


def set_optional_input_limits(input_opt: pyneat.InputOptions, width: int, height: int, depth: int) -> None:
    for attr, value in (
        ("max_width", width),
        ("max_height", height),
        ("max_depth", depth),
    ):
        if hasattr(input_opt, attr):
            setattr(input_opt, attr, value)


class NeatOptiViewVideoWriter:
    """Encode RGB frames with NEAT hardware H.264 and publish them over UDP."""

    def __init__(self, host: str, port: int, width: int, height: int, fps: int):
        self._width = width
        self._height = height

        input_opt = pyneat.InputOptions()
        input_opt.media_type = "video/x-raw"
        input_opt.format = "RGB"
        input_opt.use_simaai_pool = False
        set_optional_input_limits(input_opt, width, height, 3)

        session = pyneat.Session()
        session.add(pyneat.nodes.input(input_opt))
        session.add(pyneat.nodes.video_convert())
        session.add(
            pyneat.nodes.h264_encode_sima(
                width,
                height,
                max(1, fps),
                bitrate_kbps=4000,
                profile="baseline",
                level="4.1",
            )
        )

        udp_opt = pyneat.UdpH264OutputGroupOptions()
        udp_opt.payload_type = 96
        udp_opt.config_interval = 1
        udp_opt.udp_host = host
        udp_opt.udp_port = port
        udp_opt.udp_sync = False
        udp_opt.udp_async = False
        session.add(pyneat.groups.udp_h264_output_group(udp_opt))

        seed = tensor_from_rgb_frame(np.zeros((height, width, 3), dtype=np.uint8))
        run_opt = pyneat.RunOptions()
        run_opt.queue_depth = 1
        run_opt.overflow_policy = pyneat.OverflowPolicy.KeepLatest
        self._session = session
        self._run = session.build(seed, pyneat.RunMode.Async, run_opt)

    def write(self, frame: np.ndarray) -> None:
        if frame.shape[1] != self._width or frame.shape[0] != self._height:
            raise RuntimeError(f"unexpected frame size: got {frame.shape[1]}x{frame.shape[0]}")
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        rgb = cv2.cvtColor(np.ascontiguousarray(frame), cv2.COLOR_BGR2RGB)
        if not self._run.push_tensor(tensor_from_rgb_frame(rgb)):
            raise RuntimeError("OptiView video encoder rejected frame")

    def release(self) -> None:
        try:
            self._run.close()
        except Exception:
            pass


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


def build_arg_parser() -> argparse.ArgumentParser:
    """Expose only the small set of controls needed for this reference flow."""
    parser = argparse.ArgumentParser(description="Single-camera RTSP YOLOv8 OptiView example")
    parser.add_argument("--rtsp", required=True, help="RTSP URL")
    parser.add_argument("--model", dest="model", default="", help="Path to YOLOv8 compiled model package")
    parser.add_argument("--frames", type=int, default=0, help="Number of frames to process (0 = run forever)")
    parser.add_argument("--optiview-host", default="127.0.0.1", help="OptiView host")
    parser.add_argument("--optiview-video-port", type=int, default=9000, help="OptiView UDP video port")
    parser.add_argument("--optiview-json-port", type=int, default=9100, help="OptiView UDP JSON port")
    parser.add_argument("--latency-ms", type=int, default=200, help="RTSP latency in milliseconds")
    parser.add_argument("--udp", action="store_true", help="Use UDP RTSP transport instead of TCP")
    parser.add_argument("--debug", action="store_true", help="Enable timing prints")
    return parser


def parse_config(argv: list[str] | None = None) -> AppConfig:
    args = build_arg_parser().parse_args(argv)
    return AppConfig(
        rtsp=args.rtsp,
        model=args.model,
        frames=args.frames,
        optiview_host=args.optiview_host,
        optiview_video_port=args.optiview_video_port,
        optiview_json_port=args.optiview_json_port,
        latency_ms=args.latency_ms,
        udp=args.udp,
        debug=args.debug,
    )


def resolve_yolov8s_model(root: Path) -> str:
    """Mirror the C++ sample's local-first model lookup strategy.

    Resolution order:

    1. explicit environment override
    2. local/common modelzoo directories
    3. `sima-cli modelzoo -v 2.0.0 get yolo_v8s`
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
    cfg = parse_config()
    model_path = cfg.model or resolve_yolov8s_model(Path.cwd())
    if not model_path or not Path(model_path).is_file():
        print("Failed to locate yolo_v8s compiled model package.", file=sys.stderr)
        print("Set --model or run 'sima-cli modelzoo -v 2.0.0 get yolo_v8s'.", file=sys.stderr)
        return 2

    rtsp_session = None
    rtsp_run = None
    writer = None
    json_sock = None
    try:
        # Probe first so decode, inference, and UDP output all agree on the
        # same live frame dimensions.
        frame_w, frame_h, fps = probe_rtsp(cfg.rtsp)
        print(f"[init] probed RTSP decode dims {frame_w}x{frame_h}")

        # NEAT boundary: build YOLO model runtime from the resolved compiled model package.
        model = build_model(model_path, frame_w, frame_h)
        # NEAT boundary: build RTSP decode runtime used by pull_tensors().
        rtsp_session, rtsp_run = build_rtsp_run(
            cfg.rtsp,
            frame_w,
            frame_h,
            fps,
            cfg.latency_ms,
            tcp=not cfg.udp,
        )
        # NEAT boundary: build OptiView video/json transports.
        writer = NeatOptiViewVideoWriter(
            cfg.optiview_host, cfg.optiview_video_port, frame_w, frame_h, fps
        )
        json_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        print(f"optiview host={cfg.optiview_host} video_port={cfg.optiview_video_port} "
              f"json_port={cfg.optiview_json_port} channel=0")

        processed = 0
        started = time.perf_counter()
        # Contract: single-threaded frame order is pull -> infer -> publish video -> publish JSON.
        while cfg.frames <= 0 or processed < cfg.frames:
            # Push/pull integration point: pull one decoded frame from RTSP run.
            t_pull0 = time.perf_counter()
            tensors = rtsp_run.pull_tensors(timeout_ms=5000)
            t_pull1 = time.perf_counter()
            if not tensors:
                print("RTSP pull timed out / stream closed", file=sys.stderr)
                break

            frame = tensor_bgr_from_decoded(tensors[0])
            infer_input = tensor_from_bgr_frame(frame)

            # Push/pull integration point: run model and parse the BBOX payload.
            t_inf0 = time.perf_counter()
            result = model.run(infer_input, timeout_ms=5000)
            t_inf1 = time.perf_counter()

            payload = extract_bbox_payload(result)
            if not payload:
                raise RuntimeError("model returned no BBOX payload")
            boxes = parse_bbox_payload(payload, frame.shape[1], frame.shape[0])

            # Contract: publish video first, then publish matching JSON metadata.
            writer.write(frame)
            fid = str(processed)
            payload_json = make_optiview_json(int(time.time() * 1000), fid, boxes)
            json_sock.sendto(payload_json.encode("utf-8"), (cfg.optiview_host, cfg.optiview_json_port))

            processed += 1
            if cfg.debug and (processed <= 5 or processed % 30 == 0):
                print(
                    f"[debug] frame={processed} pull_ms={(t_pull1 - t_pull0) * 1000.0:.2f} "
                    f"infer_ms={(t_inf1 - t_inf0) * 1000.0:.2f} boxes={len(boxes)}"
                )

        elapsed = max(time.perf_counter() - started, 1e-6)
        print(f"processed={processed} fps={processed / elapsed:.2f} "
              f"udp={cfg.optiview_host}:{cfg.optiview_video_port}")
        return 0 if processed > 0 else 3
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    finally:
        # Contract: release video writer, then close RTSP run, then close JSON socket.
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
