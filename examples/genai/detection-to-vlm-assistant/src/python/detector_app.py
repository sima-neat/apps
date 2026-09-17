#!/usr/bin/env python3
"""Run browser-webcam YOLO26 detection with semantic people tracking."""

from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass
import json
from pathlib import Path
import socket
import struct
import sys
import time
from urllib import error, request

import cv2
import numpy as np
import pyneat
import yaml

from async_pipeline import AsyncModelPipeline
from dashboard import DashboardServer, DashboardState
from semantic_describer import AppearanceDescriber, SemanticState
from tracker import ObjectTracker, TrackedDetection, TrackerConfig
from visibility_gate import FullPersonVisibilityGate, overlapping_bboxes

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "common" / "config.yaml"
DEFAULT_SYSTEM_PROMPT = (
    "Output one concise comma-separated line of visible attributes only. "
    "Never explain, identify the person, describe actions, or repeat a field name."
)
DEFAULT_USER_PROMPT = (
    "Describe the visible person richly using 6 to 10 comma-separated attribute "
    "phrases when observable. Include apparent age group and male or female "
    "presentation when clear, hair color and style, upper clothing color, type "
    "and pattern, lower clothing, footwear, carried items or accessories, and "
    "one distinctive visible detail. Omit anything not visible. Never describe "
    "identity, ethnicity, emotion, or action. Never output field names, no, or "
    "none. Example: adult male, short dark hair, white striped polo, dark "
    "trousers, black shoes, glasses, blue lanyard."
)


@dataclass(frozen=True)
class Config:
    model_path: str
    labels_path: str
    frames: int
    min_score: float
    nms_iou: float
    max_detections: int
    inference_queue_depth: int
    inference_internal_queue_depth: int
    timeout_ms: int
    debug: bool
    tracker_high_score: float
    tracker_new_track_score: float
    tracker_iou_threshold: float
    tracker_max_center_distance: float
    tracker_velocity_momentum: float
    tracker_max_missing: int
    tracker_min_confirmed_hits: int
    tracker_max_active: int
    tracker_center_distance_enabled: bool
    description_edge_margin_ratio: float
    description_min_clear_frames: int
    webcam_target_fps: int
    webcam_upload_max_width: int
    webcam_max_frame_bytes: int
    genai_enabled: bool
    genai_host: str
    genai_port: int
    genai_model: str
    genai_max_tokens: int
    genai_timeout_seconds: float
    genai_max_pending_requests: int
    web_enabled: bool
    web_host: str
    web_port: int
    web_title: str
    web_tls_enabled: bool
    web_tls_cert: str
    web_tls_key: str
    web_tls_ca_cert: str

@dataclass(frozen=True)
class DetectionFrame:
    frame_bgr: np.ndarray
    width: int
    height: int


def load_config(path: Path) -> Config:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    model = raw.get("model", {})
    webcam = raw.get("webcam", {})
    inference = raw.get("inference", {})
    runtime = raw.get("runtime", {})
    tracking = raw.get("tracking", {})
    genai_server = raw.get("genai_server", {})
    server_model = genai_server.get("model", {})
    genai = raw.get("genai", {})
    web = raw.get("web", {})
    web_tls = web.get("tls", {})
    server_model_name = str(server_model.get("name", "") or "")
    genai_server_port = int(genai_server.get("port", 9998))
    return Config(
        model_path=str(model.get("path", "")),
        labels_path=str(model.get("labels", "")),
        frames=int(inference.get("frames", 0)),
        min_score=float(inference.get("min_score", 0.55)),
        nms_iou=float(inference.get("nms_iou", 0.50)),
        max_detections=int(inference.get("max_detections", 24)),
        inference_queue_depth=max(2, int(inference.get("queue_depth", 4))),
        inference_internal_queue_depth=max(
            1, int(inference.get("internal_queue_depth", 2))
        ),
        timeout_ms=int(runtime.get("timeout_ms", 20000)),
        debug=bool(runtime.get("debug", False)),
        tracker_high_score=float(tracking.get("high_score_threshold", 0.55)),
        tracker_new_track_score=float(tracking.get("new_track_threshold", 0.65)),
        tracker_iou_threshold=float(tracking.get("match_iou_threshold", 0.20)),
        tracker_max_center_distance=float(tracking.get("max_center_distance", 1.5)),
        tracker_velocity_momentum=float(tracking.get("velocity_momentum", 0.80)),
        tracker_max_missing=int(tracking.get("max_missing_frames", 30)),
        tracker_min_confirmed_hits=int(tracking.get("min_confirmed_hits", 3)),
        tracker_max_active=int(tracking.get("max_active_tracks", 32)),
        tracker_center_distance_enabled=bool(
            tracking.get("center_distance_enabled", True)
        ),
        description_edge_margin_ratio=float(
            tracking.get("description_edge_margin_ratio", 0.03)
        ),
        description_min_clear_frames=int(
            tracking.get("description_min_clear_frames", 5)
        ),
        webcam_target_fps=max(1, int(webcam.get("target_fps", 30))),
        webcam_upload_max_width=max(
            160, int(webcam.get("upload_max_width", 640))
        ),
        webcam_max_frame_bytes=max(
            65536, int(webcam.get("max_frame_bytes", 4 * 1024 * 1024))
        ),
        genai_enabled=bool(genai.get("enabled", False)),
        genai_host=str(genai.get("host", "127.0.0.1") or "127.0.0.1"),
        genai_port=int(genai.get("port", genai_server_port)),
        genai_model=str(genai.get("model") or server_model_name),
        genai_max_tokens=int(genai.get("max_tokens", 64)),
        genai_timeout_seconds=float(genai.get("timeout_seconds", 30.0)),
        genai_max_pending_requests=max(1, int(genai.get("max_pending_requests", 8))),
        web_enabled=bool(web.get("enabled", True)),
        web_host=str(web.get("host", "0.0.0.0") or "0.0.0.0"),
        web_port=int(web.get("port", 5000)),
        web_title=str(web.get("title", "Real-Time Semantic People Tracker")),
        web_tls_enabled=bool(web_tls.get("enabled", False)),
        web_tls_cert=str(web_tls.get("cert", "")),
        web_tls_key=str(web_tls.get("key", "")),
        web_tls_ca_cert=str(web_tls.get("ca_cert", "")),
    )


def load_labels(path: str) -> list[str]:
    label_path = Path(path)
    if not path or not label_path.is_file():
        return []
    return [
        line.strip()
        for line in label_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def bbox_payload(result) -> bytes:
    if not isinstance(result, (list, tuple)):
        return b""
    for tensor in result:
        try:
            payload = tensor.copy_payload_bytes()
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
    for index in range(count):
        x, y, width, height, score, class_id = struct.unpack_from(
            "<iiiifi", payload, 4 + index * 24
        )
        if width > 0 and height > 0:
            boxes.append(
                {
                    "bbox": [x, y, width, height],
                    "score": float(score),
                    "class_id": int(class_id),
                }
            )
    return boxes


def label_for_box(box: dict, labels: list[str]) -> str:
    class_id = int(box["class_id"])
    if 0 <= class_id < len(labels):
        return labels[class_id]
    return f"class_{class_id}"


def person_tracker_detections(boxes: list[dict], labels: list[str]) -> list[dict]:
    result: list[dict] = []
    for box in boxes:
        label = label_for_box(box, labels)
        if label.lower() != "person":
            continue
        x, y, width, height = [float(value) for value in box["bbox"]]
        result.append(
            {
                "x1": x,
                "y1": y,
                "x2": x + width,
                "y2": y + height,
                "score": float(box["score"]),
                "class_id": int(box["class_id"]),
            }
        )
    return result


def crop_track(
    frame: np.ndarray, track: TrackedDetection, padding: float = 0.10
) -> np.ndarray:
    frame_height, frame_width = frame.shape[:2]
    width = max(1.0, track.x2 - track.x1)
    height = max(1.0, track.y2 - track.y1)
    x0 = max(0, int(track.x1 - width * padding))
    y0 = max(0, int(track.y1 - height * padding))
    x1 = min(frame_width, int(track.x2 + width * padding + 0.5))
    y1 = min(frame_height, int(track.y2 + height * padding + 0.5))
    if x1 <= x0 or y1 <= y0:
        raise RuntimeError(f"invalid person crop for track {track.track_id}")
    crop_bgr = np.ascontiguousarray(frame[y0:y1, x0:x1]).copy()
    return np.ascontiguousarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))


def rgb_to_bgr(frame: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


def request_person_crop_response(
    crop_rgb: np.ndarray,
    cfg: Config,
) -> str:
    ok, encoded = cv2.imencode(".jpg", rgb_to_bgr(crop_rgb))
    if not ok:
        raise RuntimeError("failed to encode person crop")

    image = base64.b64encode(encoded.tobytes()).decode("ascii")
    payload = {
        "model": cfg.genai_model,
        "stream": True,
        "max_tokens": cfg.genai_max_tokens,
        "messages": [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image}"},
                    },
                    {
                        "type": "text",
                        "text": DEFAULT_USER_PROMPT,
                    },
                ],
            },
        ],
    }
    req = request.Request(
        f"http://{cfg.genai_host}:{cfg.genai_port}/v1/chat/completions",
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


class TrackAppearanceCommenter:
    def __init__(self, cfg: Config, dashboard: DashboardState):
        self.cfg = cfg
        self.dashboard = dashboard
        self.server_available: bool | None = None
        self.response_count = 0
        self.started = False
        self.describer = AppearanceDescriber(
            self._request,
            self._on_update,
            queue_size=cfg.genai_max_pending_requests,
            max_attempts=2,
        )

    def start(self) -> None:
        if self.cfg.genai_enabled and not self.started:
            self.describer.start()
            self.started = True
            self.dashboard.set_component("vlm", "online", "Ready for a new track")
        else:
            self.dashboard.set_component("vlm", "stopped", "GenAI disabled")

    def retain_tracks(self, active_track_ids: set[int]) -> None:
        if self.started:
            self.describer.retain_tracks(active_track_ids)

    def try_enqueue(self, track_id: int, crop_rgb: np.ndarray) -> bool:
        scheduled = self.started and self.describer.schedule(track_id, crop_rgb)
        if scheduled:
            self.dashboard.set_component(
                "vlm", "busy", f"Describing person #{track_id}"
            )
        return scheduled

    def needs_description(self, track_id: int) -> bool:
        return self.started and self.describer.needs_description(track_id)

    def state_for(self, track_id: int) -> SemanticState:
        if not self.started:
            return SemanticState("unavailable", None, 0)
        return self.describer.state_for(track_id) or SemanticState("waiting", None, 0)

    def close(self) -> None:
        self.describer.close()

    def _request(self, crop_rgb: np.ndarray) -> str:
        try:
            if not self._server_ready():
                raise RuntimeError("vision-language server is unavailable")
            return request_person_crop_response(crop_rgb, self.cfg)
        except Exception as exc:
            print(f"[genai-server] appearance request failed: {exc}", flush=True)
            raise

    def _on_update(
        self, track_id: int, state: SemanticState, latency_ms: float | None
    ) -> None:
        if state.status == "ready":
            self.response_count += 1
            print(
                f"[appearance #{self.response_count:03d}] "
                f"track={track_id} {state.description}",
                flush=True,
            )
            self.dashboard.set_component("vlm", "online", "Track description ready")
        else:
            print(f"[appearance] track={track_id} unavailable", flush=True)
            self.dashboard.set_component(
                "vlm", "warning", "A track description was unavailable"
            )
        self.dashboard.update_track_semantic(
            track_id,
            status=state.status,
            description=state.description,
            latency_ms=latency_ms,
        )

    def _server_ready(self) -> bool:
        url = f"http://{self.cfg.genai_host}:{self.cfg.genai_port}/v1/models"
        try:
            timeout = min(self.cfg.genai_timeout_seconds, 5.0)
            with request.urlopen(url, timeout=timeout) as response:
                ready = 200 <= response.status < 300
            self.server_available = ready
            self.dashboard.set_component(
                "vlm",
                "online" if ready else "warning",
                "Ready" if ready else "Server returned an error",
            )
            return ready
        except (TimeoutError, OSError, error.URLError) as exc:
            if self.server_available is not False:
                print(f"[genai-server] waiting for {url}: {exc}", flush=True)
            self.server_available = False
            self.dashboard.set_component("vlm", "warning", "Server unavailable")
            return False


def decode_uploaded_frame(payload: bytes) -> np.ndarray:
    encoded = np.frombuffer(payload, dtype=np.uint8)
    frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("uploaded frame is not a decodable color image")
    return np.ascontiguousarray(frame, dtype=np.uint8)


def image_tensor(frame_bgr: np.ndarray):
    return pyneat.Tensor.from_numpy(
        frame_bgr,
        copy=True,
        image_format=pyneat.PixelFormat.BGR,
        memory=pyneat.TensorMemory.EV74,
    )


def image_sample(frame_bgr: np.ndarray, frame_id: int):
    sample = pyneat.Sample()
    sample.kind = pyneat.SampleKind.Tensor
    sample.tensor = image_tensor(frame_bgr)
    sample.frame_id = int(frame_id)
    sample.stream_id = "webcam"
    return sample


def build_model(cfg: Config, width: int, height: int):
    options = pyneat.ModelOptions()
    options.preprocess.kind = pyneat.InputKind.Image
    options.preprocess.enable = pyneat.AutoFlag.On
    options.preprocess.color_convert.input_format = pyneat.PreprocessColorFormat.BGR
    options.preprocess.input_max_width = width
    options.preprocess.input_max_height = height
    options.preprocess.input_max_depth = 3
    options.preprocess.preset = pyneat.NormalizePreset.COCO_YOLO
    options.decode_type = pyneat.BoxDecodeType.YoloV26
    options.score_threshold = cfg.min_score
    options.nms_iou_threshold = cfg.nms_iou
    options.top_k = cfg.max_detections
    options.advanced_execution.preprocess_async = True
    options.advanced_execution.inference_async = True
    options.advanced_execution.internal_queue_depth = (
        cfg.inference_internal_queue_depth
    )
    options.advanced_execution.inference_output_buffers = cfg.inference_queue_depth
    return pyneat.Model(cfg.model_path, options)


def build_runner(cfg: Config, width: int, height: int):
    model = build_model(cfg, width, height)
    seed = image_tensor(np.zeros((height, width, 3), dtype=np.uint8))
    run_options = pyneat.RunOptions()
    run_options.queue_depth = cfg.inference_queue_depth
    run_options.overflow_policy = pyneat.OverflowPolicy.KeepLatest
    run_options.preset = pyneat.RunPreset.Realtime
    run_options.output_memory = pyneat.OutputMemory.ZeroCopy
    runner = model.build(
        [seed],
        route_options=pyneat.ModelRouteOptions(),
        run_options=run_options,
    )
    runner.run([seed], timeout_ms=cfg.timeout_ms)
    return model, runner


def build_tracker(cfg: Config) -> ObjectTracker:
    return ObjectTracker(
        TrackerConfig(
            high_score_threshold=cfg.tracker_high_score,
            new_track_threshold=cfg.tracker_new_track_score,
            match_iou_threshold=cfg.tracker_iou_threshold,
            max_center_distance=cfg.tracker_max_center_distance,
            velocity_momentum=cfg.tracker_velocity_momentum,
            max_missing_frames=cfg.tracker_max_missing,
            min_confirmed_hits=cfg.tracker_min_confirmed_hits,
            max_active_tracks=cfg.tracker_max_active,
            center_distance_enabled=cfg.tracker_center_distance_enabled,
        )
    )


def browser_track(track: TrackedDetection, state: SemanticState) -> dict:
    return {
        "track_id": track.track_id,
        "label": "person",
        "score": float(track.score),
        "bbox": [
            int(track.x1),
            int(track.y1),
            max(0, int(track.x2 - track.x1)),
            max(0, int(track.y2 - track.y1)),
        ],
        "semantic_status": state.status,
        "semantic_description": state.description,
    }


def dashboard_url(cfg: Config, actual_port: int) -> str:
    host = cfg.web_host
    if host in ("0.0.0.0", "::"):
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            probe.connect(("192.0.2.1", 9))
            host = probe.getsockname()[0]
        except OSError:
            host = socket.gethostname()
        finally:
            probe.close()
    scheme = "https" if cfg.web_tls_enabled else "http"
    return f"{scheme}://{host}:{actual_port}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Webcam semantic people tracker")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    if not args.config.is_file():
        print(f"config does not exist: {args.config}", file=sys.stderr)
        return 2

    cfg = load_config(args.config)
    if not cfg.model_path:
        print("config requires model.path", file=sys.stderr)
        return 2
    if not Path(cfg.model_path).is_file():
        print(f"model package does not exist: {cfg.model_path}", file=sys.stderr)
        return 2
    if not cfg.web_enabled:
        print("web.enabled must be true for browser webcam input", file=sys.stderr)
        return 2
    if cfg.web_tls_enabled:
        for label, value in (
            ("web.tls.cert", cfg.web_tls_cert),
            ("web.tls.key", cfg.web_tls_key),
            ("web.tls.ca_cert", cfg.web_tls_ca_cert),
        ):
            if not value or not Path(value).is_file():
                print(f"config requires an existing {label}: {value or '-'}", file=sys.stderr)
                return 2
    if cfg.genai_enabled and not cfg.genai_model:
        print("config requires genai.model or genai_server.model.name", file=sys.stderr)
        return 2

    detector_model = detector_pipeline = commenter = dashboard = dashboard_server = None
    processed = 0
    tracker_profile_frames = 0
    tracker_profile_ms = 0.0
    try:
        labels = load_labels(cfg.labels_path)
        if "person" not in {label.lower() for label in labels}:
            raise RuntimeError("model labels must contain the person class")
        tracker = build_tracker(cfg)
        visibility_gate = FullPersonVisibilityGate(
            edge_margin_ratio=cfg.description_edge_margin_ratio,
            min_clear_frames=cfg.description_min_clear_frames,
        )
        dashboard = DashboardState(
            title=cfg.web_title,
            model_name=cfg.genai_model,
            target_fps=cfg.webcam_target_fps,
            upload_max_width=cfg.webcam_upload_max_width,
        )
        dashboard_server = DashboardServer(
            dashboard,
            cfg.web_host,
            cfg.web_port,
            max_frame_bytes=cfg.webcam_max_frame_bytes,
            tls_cert=Path(cfg.web_tls_cert) if cfg.web_tls_enabled else None,
            tls_key=Path(cfg.web_tls_key) if cfg.web_tls_enabled else None,
            ca_cert=Path(cfg.web_tls_ca_cert) if cfg.web_tls_enabled else None,
        )
        dashboard_server.start()
        direct_url = dashboard_url(cfg, dashboard_server.port)
        print(
            f"[dashboard] {direct_url}\n"
            "[webcam] accept the certificate warning once, then allow camera access",
            flush=True,
        )
        dashboard.set_phase("waiting", "Allow webcam access in the browser")
        dashboard.set_component("detector", "waiting", "Waiting for first webcam frame")
        dashboard.set_component("mla", "waiting", "Waiting for first webcam frame")

        commenter = TrackAppearanceCommenter(cfg, dashboard)
        commenter.start()

        def handle_detection(context: DetectionFrame, result) -> None:
            nonlocal processed, tracker_profile_frames, tracker_profile_ms
            outputs = list(getattr(result, "tensors", []))
            tensor = getattr(result, "tensor", None)
            if not outputs and tensor is not None:
                outputs = [tensor]
            boxes = parse_boxes(outputs)
            person_boxes = person_tracker_detections(boxes, labels)
            tracker_started = time.perf_counter()
            tracks = tracker.update(person_boxes, processed)
            overlapping_people = overlapping_bboxes(
                [
                    (
                        detection["x1"],
                        detection["y1"],
                        detection["x2"],
                        detection["y2"],
                    )
                    for detection in person_boxes
                ]
            )
            tracker_ms = (time.perf_counter() - tracker_started) * 1000.0
            tracker_profile_frames += 1
            tracker_profile_ms += tracker_ms

            active_track_ids = tracker.active_track_ids()
            commenter.retain_tracks(active_track_ids)
            visibility_gate.retain_tracks(active_track_ids)
            visible: list[dict] = []
            for track in tracks:
                track_bbox = (track.x1, track.y1, track.x2, track.y2)
                if track_bbox in overlapping_people:
                    visibility_gate.reset(track.track_id)
                    eligible_for_description = False
                else:
                    eligible_for_description = visibility_gate.observe(
                        track.track_id,
                        track_bbox,
                        context.width,
                        context.height,
                    )
                if (
                    eligible_for_description
                    and commenter.needs_description(track.track_id)
                ):
                    try:
                        commenter.try_enqueue(
                            track.track_id, crop_track(context.frame_bgr, track)
                        )
                    except RuntimeError as exc:
                        if cfg.debug:
                            print(f"[tracker] crop skipped: {exc}", flush=True)
                visible.append(browser_track(track, commenter.state_for(track.track_id)))

            dashboard.record_frame(
                visible,
                width=context.width,
                height=context.height,
                tracker_latency_ms=tracker_ms,
            )
            processed += 1
            if tracker_profile_frames >= 300:
                print(
                    f"[tracker] avg_ms={tracker_profile_ms / tracker_profile_frames:.3f} "
                    f"active={tracker.active_track_count()}",
                    flush=True,
                )
                tracker_profile_frames = 0
                tracker_profile_ms = 0.0
            if cfg.debug:
                print(
                    f"[detector] frame={processed} people={len(visible)} "
                    f"tracker_ms={tracker_ms:.3f}"
                )

        frame_sequence = 0
        runner_dimensions: tuple[int, int] | None = None
        last_input_at = time.monotonic()

        while cfg.frames <= 0 or processed < cfg.frames:
            if detector_pipeline is not None:
                detector_pipeline.raise_if_failed()
            submitted = dashboard.wait_for_frame(frame_sequence, timeout=1.0)
            if submitted is None:
                if time.monotonic() - last_input_at >= 3.0:
                    dashboard.set_component(
                        "camera", "waiting", "Waiting for browser webcam frames"
                    )
                continue
            frame_sequence, payload = submitted
            last_input_at = time.monotonic()
            frame_bgr = decode_uploaded_frame(payload)
            height, width = frame_bgr.shape[:2]

            if detector_pipeline is None or runner_dimensions != (width, height):
                if detector_pipeline is not None:
                    detector_pipeline.close()
                dashboard.set_phase("initializing", "Building the Neat detection runner")
                dashboard.set_component(
                    "detector", "starting", f"Loading YOLO for {width}×{height}"
                )
                detector_model, detector_run = build_runner(cfg, width, height)
                detector_pipeline = AsyncModelPipeline(
                    detector_run,
                    handle_detection,
                    pull_timeout_ms=100,
                    max_contexts=max(8, cfg.inference_queue_depth * 4),
                )
                detector_pipeline.start()
                runner_dimensions = (width, height)
                dashboard.set_component(
                    "detector", "online", "YOLO26 async pipeline is processing"
                )
                dashboard.set_component("mla", "online", "Modalix MLA active")
                dashboard.set_phase(
                    "running", "Tracking people and describing new tracks"
                )
                print(
                    f"[detector] async webcam input {width}x{height}"
                    f"@{cfg.webcam_target_fps} queue={cfg.inference_queue_depth}",
                    flush=True,
                )

            accepted = detector_pipeline.submit(
                frame_sequence,
                image_sample(frame_bgr, frame_sequence),
                DetectionFrame(frame_bgr, width, height),
            )
            if not accepted and cfg.debug:
                print(f"[detector] dropped frame={frame_sequence}")

        return 0 if processed > 0 else 3
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    finally:
        if commenter is not None:
            commenter.close()
        if detector_pipeline is not None:
            detector_pipeline.close()
        detector_model = None
        if dashboard is not None:
            for component in ("camera", "detector", "mla", "vlm"):
                dashboard.set_component(component, "stopped", "Application stopped")
            dashboard.set_phase("stopped", "Application stopped")
        if dashboard_server is not None:
            dashboard_server.stop()


if __name__ == "__main__":
    raise SystemExit(main())
