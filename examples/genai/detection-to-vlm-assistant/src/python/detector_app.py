#!/usr/bin/env python3
"""Run the RTSP YOLO26 detector, Insight output, and optional GenAI captions."""

from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass
import json
from pathlib import Path
from queue import Empty, Full, Queue
import struct
import sys
import threading
import time
from urllib import error, request

import cv2
import numpy as np
import pyneat
import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "common" / "config.yaml"
DEFAULT_SYSTEM_PROMPT = (
    "Describe the visible action in the detected person crop. Be factual, "
    "concise, and avoid guessing identity or protected attributes."
)
DEFAULT_USER_PROMPT = "What is the person doing in this crop?"


@dataclass(frozen=True)
class Config:
    rtsp_url: str
    model_path: str
    labels_path: str
    frames: int
    min_score: float
    nms_iou: float
    max_detections: int
    classes: tuple[str, ...]
    timeout_ms: int
    debug: bool
    insight_host: str
    video_port: int
    metadata_port: int
    channel: int
    genai_enabled: bool
    genai_host: str
    genai_port: int
    genai_model: str
    genai_max_tokens: int
    genai_interval_seconds: float
    genai_timeout_seconds: float
    genai_max_pending_requests: int
    genai_system_prompt: str
    genai_user_prompt: str


def load_config(path: Path) -> Config:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    source = raw.get("source", {})
    model = raw.get("model", {})
    insight = raw.get("insight", {})
    inference = raw.get("inference", {})
    runtime = raw.get("runtime", {})
    genai_server = raw.get("genai_server", {})
    server_model = genai_server.get("model", {})
    genai = raw.get("genai", {})
    server_model_name = str(server_model.get("name", "") or "")
    genai_server_port = int(genai_server.get("port", 9998))
    return Config(
        rtsp_url=source.get("rtsp_url", ""),
        model_path=model.get("path", ""),
        labels_path=model.get("labels", ""),
        frames=int(inference.get("frames", 0)),
        min_score=float(inference.get("min_score", 0.55)),
        nms_iou=float(inference.get("nms_iou", 0.50)),
        max_detections=int(inference.get("max_detections", 24)),
        classes=class_filter(inference.get("classes")),
        timeout_ms=int(runtime.get("timeout_ms", 20000)),
        debug=bool(runtime.get("debug", False)),
        insight_host=insight.get("host", "127.0.0.1") or "127.0.0.1",
        video_port=int(insight.get("video_port", 9000)),
        metadata_port=int(insight.get("metadata_port", 9100)),
        channel=int(insight.get("channel", 0)),
        genai_enabled=bool(genai.get("enabled", False)),
        genai_host=genai.get("host", "127.0.0.1") or "127.0.0.1",
        genai_port=int(genai.get("port", genai_server_port)),
        genai_model=str(genai.get("model") or server_model_name),
        genai_max_tokens=int(genai.get("max_tokens", 128)),
        genai_interval_seconds=float(genai.get("interval_seconds", 5.0)),
        genai_timeout_seconds=float(genai.get("timeout_seconds", 30.0)),
        genai_max_pending_requests=max(1, int(genai.get("max_pending_requests", 1))),
        genai_system_prompt=genai.get("system_prompt", DEFAULT_SYSTEM_PROMPT),
        genai_user_prompt=genai.get("user_prompt", DEFAULT_USER_PROMPT),
    )


def class_filter(value) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        values = value.split(",")
    else:
        values = value
    return tuple(str(item).strip().lower() for item in values if str(item).strip())


def load_labels(path: str) -> list[str]:
    label_path = Path(path)
    if not path or not label_path.is_file():
        return []
    return [
        line.strip()
        for line in label_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def probe_rtsp(url: str) -> tuple[int, int, int]:
    cap = cv2.VideoCapture(url)
    try:
        if not cap.isOpened():
            raise RuntimeError(f"failed to open RTSP source: {url}")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = int(round(cap.get(cv2.CAP_PROP_FPS) or 0)) or 30
    finally:
        cap.release()

    if width <= 0 or height <= 0:
        raise RuntimeError("failed to probe RTSP dimensions")
    return width, height, fps


def tensor_dim(tensor, name: str) -> int:
    value = getattr(tensor, name)
    return int(value() if callable(value) else value)


def decoded_tensor_to_rgb(tensor) -> np.ndarray:
    if tensor.is_nv12():
        width = tensor_dim(tensor, "width")
        height = tensor_dim(tensor, "height")
        payload = np.frombuffer(tensor.copy_payload_bytes(), dtype=np.uint8)
        expected = width * height * 3 // 2
        if payload.size < expected:
            raise RuntimeError(f"NV12 payload too small: {payload.size} < {expected}")
        nv12 = payload[:expected].reshape((height * 3 // 2, width))
        bgr = cv2.cvtColor(nv12, cv2.COLOR_YUV2BGR_NV12)
        return np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    if tensor.is_i420():
        width = tensor_dim(tensor, "width")
        height = tensor_dim(tensor, "height")
        payload = np.frombuffer(tensor.copy_payload_bytes(), dtype=np.uint8)
        expected = width * height * 3 // 2
        if payload.size < expected:
            raise RuntimeError(f"I420 payload too small: {payload.size} < {expected}")
        i420 = payload[:expected].reshape((height * 3 // 2, width))
        bgr = cv2.cvtColor(i420, cv2.COLOR_YUV2BGR_I420)
        return np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    frame = np.asarray(tensor.to_numpy(copy=True))
    if frame.ndim == 4 and frame.shape[0] == 1:
        frame = frame[0]
    if frame.ndim != 3:
        raise RuntimeError(f"unexpected decoded tensor shape: {frame.shape}")
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(frame)


def is_tensor_like(value) -> bool:
    return hasattr(value, "copy_payload_bytes") and hasattr(value, "to_numpy")


def is_sample_like(value) -> bool:
    return hasattr(value, "kind") and hasattr(value, "fields")


def bbox_payload_from_tensors(tensors) -> bytes:
    for tensor in tensors:
        try:
            payload = tensor.copy_payload_bytes()
        except Exception:
            continue
        if payload:
            return payload
    return b""


def bbox_payload(result) -> bytes:
    if isinstance(result, (list, tuple)) and all(is_tensor_like(item) for item in result):
        return bbox_payload_from_tensors(result)

    if not is_sample_like(result):
        return b""

    stack = [result]
    while stack:
        current = stack.pop()
        stack.extend(reversed(list(current.fields)))
        if current.kind == pyneat.SampleKind.TensorSet:
            payload = bbox_payload_from_tensors(current.tensors)
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
    return b""


def parse_boxes(result) -> list[dict]:
    payload = bbox_payload(result)
    if len(payload) < 4:
        return []
    count = min(struct.unpack_from("<I", payload, 0)[0], (len(payload) - 4) // 24)
    boxes = []
    for idx in range(count):
        x, y, w, h, score, class_id = struct.unpack_from("<iiiifi", payload, 4 + idx * 24)
        if w > 0 and h > 0:
            boxes.append(
                {
                    "bbox": [x, y, w, h],
                    "score": score,
                    "class_id": class_id,
                }
            )
    return boxes


def find_field(sample, label: str):
    if getattr(sample, "stream_label", "") == label:
        return sample
    for field in getattr(sample, "fields", []):
        found = find_field(field, label)
        if found is not None:
            return found
    return None


def joined_field(sample, label: str, bundle_index: int):
    """Return one branch of the combined output, by label or by combine order."""
    field = find_field(sample, label)
    if field is not None:
        return field
    fields = list(getattr(sample, "fields", []))
    if sample.kind == pyneat.SampleKind.Bundle and len(fields) > bundle_index:
        return fields[bundle_index]
    raise RuntimeError(f"detector output missing {label} field")


def decoded_frame_tensor(sample):
    field = joined_field(sample, "frame", 0)
    if field.kind == pyneat.SampleKind.Tensor and field.tensor is not None:
        return field.tensor
    if field.kind == pyneat.SampleKind.TensorSet and field.tensors:
        return field.tensors[0]
    raise RuntimeError("detector output did not contain a decoded frame tensor")


def metadata_json(boxes: list[dict], labels: list[str], classes: tuple[str, ...] = ()) -> str:
    objects = []
    allowed = set(classes)
    for box in boxes:
        class_id = int(box["class_id"])
        label = labels[class_id] if 0 <= class_id < len(labels) else f"class_{class_id}"
        if allowed and label.lower() not in allowed:
            continue
        objects.append(
            {
                "id": f"obj_{len(objects) + 1}",
                "label": label,
                "confidence": box["score"],
                "bbox": box["bbox"],
            }
        )
    return json.dumps({"objects": objects}, separators=(",", ":"))


def best_box_for_label(boxes: list[dict], labels: list[str], wanted: str):
    wanted = wanted.lower()
    matches = []
    for box in boxes:
        class_id = int(box["class_id"])
        label = labels[class_id] if 0 <= class_id < len(labels) else f"class_{class_id}"
        if label.lower() == wanted:
            matches.append(box)
    return max(matches, key=lambda box: box["score"], default=None)


def crop_box(frame: np.ndarray, box: dict) -> np.ndarray:
    x, y, w, h = [int(value) for value in box["bbox"]]
    height, width = frame.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(width, x + w), min(height, y + h)
    if x1 <= x0 or y1 <= y0:
        raise RuntimeError(f"invalid person crop: {box['bbox']}")
    return np.ascontiguousarray(frame[y0:y1, x0:x1])


def rgb_to_bgr(frame: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


def request_person_crop_response(crop_rgb: np.ndarray, cfg: Config) -> str:
    crop = rgb_to_bgr(crop_rgb)
    ok, encoded = cv2.imencode(".jpg", crop)
    if not ok:
        raise RuntimeError("failed to encode person crop")

    image = base64.b64encode(encoded.tobytes()).decode("ascii")
    payload = {
        "model": cfg.genai_model,
        "stream": True,
        "max_tokens": cfg.genai_max_tokens,
        "messages": [
            {"role": "system", "content": cfg.genai_system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image}"},
                    },
                    {
                        "type": "text",
                        "text": cfg.genai_user_prompt,
                    },
                ],
            },
        ],
    }

    url = f"http://{cfg.genai_host}:{cfg.genai_port}/v1/chat/completions"
    req = request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    response_text = ""
    with request.urlopen(req, timeout=cfg.genai_timeout_seconds) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data: "):
                continue
            data = line.removeprefix("data: ")
            if data == "[DONE]":
                break
            delta = json.loads(data).get("choices", [{}])[0].get("delta", {})
            response_text += delta.get("content") or ""
    return response_text.strip()


class GenAICommenter:
    def __init__(self, cfg: Config, labels: list[str]):
        self.cfg = cfg
        self.labels = labels
        self.queue: Queue[np.ndarray] = Queue(maxsize=cfg.genai_max_pending_requests)
        self.stop_event = threading.Event()
        self.worker = threading.Thread(target=self._run, daemon=True)
        self.lock = threading.Lock()
        self.last_enqueue_at = 0.0
        self.in_flight = False
        self.server_available: bool | None = None
        self.response_count = 0
        self.started = False

    def start(self) -> None:
        if self.cfg.genai_enabled and not self.started:
            self.worker.start()
            self.started = True

    def try_enqueue(self, sample, boxes: list[dict]) -> None:
        if not self.cfg.genai_enabled:
            return
        now = time.monotonic()
        if now - self.last_enqueue_at < self.cfg.genai_interval_seconds:
            return
        if self._pending_count() >= self.cfg.genai_max_pending_requests:
            print("[genai-server] queue busy, dropping request", flush=True)
            self.last_enqueue_at = now
            return

        box = best_box_for_label(boxes, self.labels, "person")
        if box is None:
            return

        frame = decoded_tensor_to_rgb(decoded_frame_tensor(sample))
        try:
            self.queue.put_nowait(crop_box(frame, box).copy())
            self.last_enqueue_at = now
        except Full:
            print("[genai-server] queue full, dropping request", flush=True)
            self.last_enqueue_at = now

    def close(self) -> None:
        self.stop_event.set()
        if self.started:
            self.worker.join(timeout=1.0)

    def _pending_count(self) -> int:
        with self.lock:
            return self.queue.qsize() + int(self.in_flight)

    def _set_in_flight(self, value: bool) -> None:
        with self.lock:
            self.in_flight = value

    def _run(self) -> None:
        while not self.stop_event.is_set():
            try:
                crop = self.queue.get(timeout=0.2)
            except Empty:
                continue

            self._set_in_flight(True)
            try:
                if self._server_ready():
                    response = request_person_crop_response(crop, self.cfg)
                    if response:
                        self.response_count += 1
                        print(
                            f"\n[response #{self.response_count:03d}] {response}",
                            flush=True,
                        )
            except (TimeoutError, OSError, error.URLError) as exc:
                print(f"[genai-server] request failed: {exc}", flush=True)
            except Exception as exc:
                print(f"[genai-server] request failed: {exc}", flush=True)
            finally:
                self._set_in_flight(False)
                self.queue.task_done()

    def _server_ready(self) -> bool:
        url = f"http://{self.cfg.genai_host}:{self.cfg.genai_port}/v1/models"
        try:
            timeout = min(self.cfg.genai_timeout_seconds, 5.0)
            with request.urlopen(url, timeout=timeout) as res:
                ready = 200 <= res.status < 300
                if ready and self.server_available is False:
                    print(
                        f"\n[genai-server] connected "
                        f"http://{self.cfg.genai_host}:{self.cfg.genai_port}",
                        flush=True,
                    )
                self.server_available = ready
                return ready
        except (TimeoutError, OSError, error.URLError) as exc:
            if self.server_available is not False:
                print(
                    f"[genai-server] waiting for "
                    f"http://{self.cfg.genai_host}:{self.cfg.genai_port}: {exc}",
                    flush=True,
                )
            self.server_available = False
            return False


def build_source_graph(cfg: Config, width: int, height: int, fps: int):
    opt = pyneat.RtspDecodedInputOptions()
    opt.url = cfg.rtsp_url
    opt.payload_type = 96
    opt.insert_queue = True
    opt.auto_caps_from_stream = True
    opt.fallback_h264_width = width
    opt.fallback_h264_height = height
    opt.fallback_h264_fps = fps
    opt.sima_allocator_type = 2
    opt.decoder_raw_output = True
    opt.output_caps.enable = True
    opt.output_caps.format = pyneat.Format.NV12
    opt.output_caps.width = width
    opt.output_caps.height = height
    opt.output_caps.fps = fps
    opt.output_caps.memory = pyneat.CapsMemory.Any
    return pyneat.groups.rtsp_decoded_input(opt)


def build_video_graph(cfg: Config, width: int, height: int, fps: int):
    sender_opt = pyneat.VideoSenderOptions.h264_rtp_udp_from_raw(width, height, max(1, fps))
    sender_opt.host = cfg.insight_host
    sender_opt.channel = cfg.channel
    sender_opt.video_port_base = cfg.video_port

    graph = pyneat.Graph("video")
    graph.connect(pyneat.nodes.input("video"), pyneat.groups.video_sender(sender_opt))
    return graph


def build_model(cfg: Config, width: int, height: int):
    opt = pyneat.ModelOptions()
    opt.preprocess.kind = pyneat.InputKind.Image
    opt.preprocess.enable = pyneat.AutoFlag.On
    opt.preprocess.color_convert.input_format = pyneat.PreprocessColorFormat.NV12
    opt.preprocess.input_max_width = width
    opt.preprocess.input_max_height = height
    opt.preprocess.preset = pyneat.NormalizePreset.COCO_YOLO
    opt.decode_type = pyneat.BoxDecodeType.YoloV26
    opt.score_threshold = cfg.min_score
    opt.nms_iou_threshold = cfg.nms_iou
    opt.top_k = cfg.max_detections
    return pyneat.Model(cfg.model_path, opt)


def build_pipeline(cfg: Config, width: int, height: int, fps: int):
    model = build_model(cfg, width, height)
    source = build_source_graph(cfg, width, height, fps)
    video_graph = build_video_graph(cfg, width, height, fps)

    # Insight correlates the RTP timestamp with the metadata timestamp, so the encoder and the
    # detections must stay in one Run and therefore on one GStreamer timeline. The frame branch
    # returns the decoded frame the GenAI commenter crops.
    branch = pyneat.graphs.branch("source", ["video", "model", "frame"])

    model_graph = pyneat.Graph("model")
    model_graph.connect(pyneat.nodes.input("model"), model)

    detections_graph = pyneat.Graph("detections")
    detections_graph.add(pyneat.nodes.output("detections", pyneat.OutputOptions.every_frame(4)))

    frame_graph = pyneat.Graph("frame")
    frame_graph.add(pyneat.nodes.output("frame", pyneat.OutputOptions.every_frame(4)))

    joined = pyneat.graphs.combine(
        ["frame", "detections"], "detector_output", pyneat.CombinePolicy.ByFrame
    )

    graph = pyneat.Graph()
    graph.connect(source, branch)
    graph.connect(branch, video_graph)
    graph.connect(branch, model_graph)
    graph.connect(model_graph, detections_graph)
    graph.connect(branch, frame_graph)
    graph.connect(frame_graph, joined)
    graph.connect(detections_graph, joined)

    run_opt = pyneat.RunOptions()
    run_opt.queue_depth = 4
    run_opt.overflow_policy = pyneat.OverflowPolicy.KeepLatest
    run_opt.output_memory = pyneat.OutputMemory.Owned
    return model, graph, graph.build(run_opt)


def build_metadata_sender(cfg: Config):
    opt = pyneat.MetadataSenderOptions()
    opt.host = cfg.insight_host
    opt.channel = cfg.channel
    opt.metadata_port_base = cfg.metadata_port
    return pyneat.MetadataSender(opt)


def main() -> int:
    parser = argparse.ArgumentParser(description="Detection-to-VLM assistant")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    if not args.config.is_file():
        print(f"config does not exist: {args.config}", file=sys.stderr)
        return 2

    cfg = load_config(args.config)
    if not cfg.rtsp_url or not cfg.model_path:
        print("config requires source.rtsp_url and model.path", file=sys.stderr)
        return 2
    if not Path(cfg.model_path).is_file():
        print(f"model package does not exist: {cfg.model_path}", file=sys.stderr)
        return 2
    if cfg.genai_enabled and not cfg.genai_model:
        print("config requires genai.model or genai_server.model.name", file=sys.stderr)
        return 2

    detector_run = commenter = None
    try:
        labels = load_labels(cfg.labels_path)
        width, height, fps = probe_rtsp(cfg.rtsp_url)
        _model, _detector_graph, detector_run = build_pipeline(cfg, width, height, fps)
        metadata = build_metadata_sender(cfg)
        commenter = GenAICommenter(cfg, labels)
        commenter.start()
        print(
            f"[detector] stream {cfg.rtsp_url}\n"
            f"[detector] input {width}x{height}@{fps}\n"
            f"[insight] video={cfg.insight_host}:{cfg.video_port} "
            f"metadata={cfg.insight_host}:{cfg.metadata_port} channel={cfg.channel}\n"
            f"[genai-server] "
            f"{'enabled' if cfg.genai_enabled else 'disabled'} "
            f"model={cfg.genai_model or '-'} "
            f"url=http://{cfg.genai_host}:{cfg.genai_port} "
            f"interval={cfg.genai_interval_seconds:g}s\n"
        )

        processed = 0
        while cfg.frames <= 0 or processed < cfg.frames:
            sample = detector_run.pull("detector_output", cfg.timeout_ms)
            if sample is None:
                print("RTSP stream ended or pull timed out", file=sys.stderr)
                break
            boxes = parse_boxes(joined_field(sample, "detections", 1))
            commenter.try_enqueue(sample, boxes)
            ok = metadata.send_metadata(
                "object-detection",
                metadata_json(boxes, labels, cfg.classes),
                int(sample.pts_ns // 1_000_000) if sample.pts_ns >= 0 else -1,
                str(sample.frame_id) if sample.frame_id >= 0 else "",
            )
            if not ok:
                raise RuntimeError("failed to send metadata to Insight")
            processed += 1
            if cfg.debug:
                print(f"[detector] frame={processed} detections={len(boxes)}")
        return 0 if processed > 0 else 3
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    finally:
        if commenter is not None:
            commenter.close()
        if detector_run is not None:
            detector_run.close()


if __name__ == "__main__":
    raise SystemExit(main())
