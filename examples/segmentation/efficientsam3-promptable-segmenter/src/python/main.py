"""Single-stream RTSP promptable segmentation with EfficientSAM3 and Insight output, using pyneat."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pyneat
import yaml

from clip_tokenizer import ClipTokenizer

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "common" / "config.yaml"
PULL_TIMEOUT_MS = 20000
MODEL_SIZE = 1008
TEXT_TOKENS = 16
TEXT_DIM = 256
VOCAB_SIZE = 49408
PIPELINE_DEPTH = 3


@dataclass(frozen=True)
class Config:
    model_path: str
    text_encoder: str
    prompt: str
    rtsp_url: str
    tcp: bool
    latency_ms: int
    frames: int
    min_score: float
    max_detections: int
    mask_threshold: float
    profile: bool
    profile_interval: int
    insight_host: str
    video_port: int
    metadata_port: int
    save_dir: str
    save_every: int


def load_config(path: Path) -> Config:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    source, inference, runtime, output = raw["source"], raw["inference"], raw["runtime"], raw["output"]
    insight = output["insight"]
    return Config(
        model_path=raw["model"]["path"],
        text_encoder=raw["model"]["text_encoder"],
        prompt=raw["prompt"]["text"],
        rtsp_url=source["rtsp_url"],
        tcp=source["tcp"],
        latency_ms=source["latency_ms"],
        frames=inference["frames"],
        min_score=inference["min_score"],
        max_detections=inference["max_detections"],
        mask_threshold=inference["mask_threshold"],
        profile=runtime["profile"],
        profile_interval=runtime["profile_interval"],
        insight_host=insight["host"],
        video_port=insight["video_port"],
        metadata_port=insight["metadata_port"],
        save_dir=output["save_dir"],
        save_every=output["save_every"],
    )


def encode_prompt(cfg: Config):
    tokens = ClipTokenizer().encode(cfg.prompt, TEXT_TOKENS)
    # The MLA cannot look up token ids, so the text encoder takes them one-hot.
    one_hot = np.zeros((1, TEXT_TOKENS, VOCAB_SIZE), np.float32)
    one_hot[0, np.arange(TEXT_TOKENS), tokens] = 1.0
    encoder = pyneat.Model(cfg.text_encoder).build([ev74_tensor(one_hot)])
    encoder.push([ev74_tensor(one_hot)])
    features = encoder.pull(PULL_TIMEOUT_MS).tensors[0].to_numpy(copy=True)
    encoder.close()
    text = np.zeros((1, TEXT_TOKENS + 1, TEXT_DIM), np.float32)
    text[0, :TEXT_TOKENS] = features.reshape(TEXT_TOKENS, TEXT_DIM)
    text[0, TEXT_TOKENS, :TEXT_TOKENS] = tokens != 0
    return text, int((tokens != 0).sum())


def probe_stream(url: str, tcp: bool) -> tuple[int, int, int]:
    if tcp:
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
    capture = cv2.VideoCapture(url)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(round(capture.get(cv2.CAP_PROP_FPS)))
    capture.release()
    if width <= 0 or height <= 0 or fps <= 0:
        raise RuntimeError(f"could not read the frame size and rate of {url}")
    return width, height, fps


def build_source(cfg: Config, width: int, height: int, fps: int):
    source = pyneat.RtspDecodedInputOptions()
    source.url = cfg.rtsp_url
    source.tcp = cfg.tcp
    source.latency_ms = cfg.latency_ms
    source.fallback_h264_width = width
    source.fallback_h264_height = height
    source.fallback_h264_fps = fps

    video = pyneat.VideoSenderOptions.h264_rtp_udp_from_raw(width, height, fps)
    video.host = cfg.insight_host
    video.video_port_base = cfg.video_port

    frames = pyneat.OutputOptions.every_frame(8)
    frames.drop = True

    decoded = pyneat.groups.rtsp_decoded_input(source)
    graph = pyneat.Graph()
    graph.connect(decoded, pyneat.groups.video_sender(video))
    graph.connect(decoded, pyneat.nodes.output("frame", frames))
    run_options = pyneat.RunOptions()
    run_options.preset = pyneat.RunPreset.Realtime
    run_options.queue_depth = 3
    return graph.build(run_options)


def ev74_tensor(array):
    return pyneat.Tensor.from_numpy(array, copy=False, memory=pyneat.TensorMemory.EV74,
                                    layout=pyneat.TensorLayout.HWC)


def nv12(tensor):
    # contiguous() drops the row padding the decoder adds to some frames.
    packed = np.frombuffer(tensor.contiguous().copy_payload_bytes(), np.uint8)
    return packed.reshape(tensor.height() * 3 // 2, tensor.width())


def preprocess(frame, out) -> None:
    height, width = frame.shape[0] * 2 // 3, frame.shape[1]
    shrink = width > MODEL_SIZE or height > MODEL_SIZE
    interpolation = cv2.INTER_AREA if shrink else cv2.INTER_LINEAR
    y = cv2.resize(frame[:height], (MODEL_SIZE, MODEL_SIZE), interpolation=interpolation)
    uv = frame[height:].reshape(height // 2, width // 2, 2)
    uv = cv2.resize(uv, (MODEL_SIZE // 2, MODEL_SIZE // 2), interpolation=interpolation)
    rgb = cv2.cvtColorTwoPlane(y, uv, cv2.COLOR_YUV2RGB_NV12)
    np.multiply(rgb, np.float32(1.0 / 127.5), out=out, casting="unsafe")
    out -= np.float32(1.0)


def mask_outline(mask_logits, box, width: int, height: int, logit_threshold: float) -> list:
    mask_h, mask_w = mask_logits.shape
    cell_w, cell_h = width / mask_w, height / mask_h
    x0, y0, x1, y1 = box
    mx0, my0 = max(0, math.floor(x0 / cell_w) - 1), max(0, math.floor(y0 / cell_h) - 1)
    mx1, my1 = min(mask_w, math.ceil(x1 / cell_w) + 1), min(mask_h, math.ceil(y1 / cell_h) + 1)
    out_x0, out_y0 = round(mx0 * cell_w), round(my0 * cell_h)
    out_size = (round(mx1 * cell_w) - out_x0, round(my1 * cell_h) - out_y0)
    # Upscale the mask cells under the box before thresholding, as SAM 3 does, so the outline follows the
    # interpolated mask.
    cells = np.ascontiguousarray(mask_logits[my0:my1, mx0:mx1])
    inside = cv2.resize(cells, out_size, interpolation=cv2.INTER_LINEAR) > logit_threshold
    contours, _ = cv2.findContours(inside.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    largest = max(contours, key=cv2.contourArea)
    polygon = cv2.approxPolyDP(largest, 0.004 * cv2.arcLength(largest, True), True)
    if len(polygon) < 3:
        return []
    return [[int(x) + out_x0, int(y) + out_y0] for [[x, y]] in polygon]


def segments_of(detections, masks, cfg: Config, width: int, height: int) -> list[dict]:
    logit_threshold = np.log(np.float64(cfg.mask_threshold) / (1.0 - cfg.mask_threshold))
    best = np.argsort(-detections[:, 4], kind="stable")[:cfg.max_detections]
    sx, sy = width / MODEL_SIZE, height / MODEL_SIZE
    segments = []
    for n, query in enumerate((q for q in best if detections[q, 4] > cfg.min_score), start=1):
        x1, y1, x2, y2, score = (float(v) for v in detections[query, :5])
        x0 = max(0, min(width - 1, math.floor(x1 * sx)))
        y0 = max(0, min(height - 1, math.floor(y1 * sy)))
        box = (x0, y0, max(x0 + 1, min(width, math.ceil(x2 * sx))), max(y0 + 1, min(height, math.ceil(y2 * sy))))
        outline = mask_outline(masks[:, :, query], box, width, height, logit_threshold)
        if outline:
            segments.append({"id": f"seg_{n}", "label": cfg.prompt, "confidence": round(score, 4),
                             "bbox": [box[0], box[1], box[2] - box[0], box[3] - box[1]],
                             "mask_format": "polygon", "mask": outline})
    return segments


# Insight draws metadata only on the frame with its timestamp, and the model segments every second or
# third frame, so every frame is sent the result of the segmented frame nearest to it in time.
class OverlayClock:

    def __init__(self) -> None:
        self.waiting = deque()
        self.previous = None

    def add_frame(self, sample) -> None:
        self.waiting.append((sample.pts_ns // 1_000_000, str(sample.frame_id)))

    def add_result(self, pts_ms: int, data: str) -> list[tuple[str, int, str]]:
        messages = []
        while self.waiting and self.waiting[0][0] <= pts_ms:
            frame_pts, frame_id = self.waiting.popleft()
            closer_to_previous = self.previous and frame_pts - self.previous[0] < pts_ms - frame_pts
            messages.append((self.previous[1] if closer_to_previous else data, frame_pts, frame_id))
        self.previous = (pts_ms, data)
        return messages


def newest_frame(graph_run, overlay: OverlayClock, timeout_ms: int):
    newest = None
    while (sample := graph_run.pull("frame", timeout_ms)) is not None:
        overlay.add_frame(sample)
        newest, timeout_ms = sample, 0
    return newest


def save_frame(path: Path, frame, segments: list[dict]) -> None:
    color = (255, 191, 0)
    bgr = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_NV12)
    outlines = [np.int32(segment["mask"]) for segment in segments]
    annotated = cv2.addWeighted(cv2.fillPoly(bgr.copy(), outlines, color), 0.5, bgr, 0.5, 0.0)
    cv2.polylines(annotated, outlines, True, color, 2)
    for segment in segments:
        x, y, w, h = segment["bbox"]
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
        cv2.putText(annotated, f"{segment['label']} {segment['confidence']:.2f}", (x, max(12, y - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    cv2.imwrite(str(path), annotated)


def run(cfg: Config) -> None:
    text, tokens = encode_prompt(cfg)
    text_tensor = ev74_tensor(text)
    model_options = pyneat.RunOptions()
    model_options.queue_depth = PIPELINE_DEPTH
    image = np.zeros((MODEL_SIZE, MODEL_SIZE, 3), np.float32)
    runner = pyneat.Model(cfg.model_path).build([ev74_tensor(image), text_tensor], run_options=model_options)
    width, height, fps = probe_stream(cfg.rtsp_url, cfg.tcp)
    graph_run = build_source(cfg, width, height, fps)
    metadata_options = pyneat.MetadataSenderOptions()
    metadata_options.host = cfg.insight_host
    metadata_options.metadata_port_base = cfg.metadata_port
    metadata = pyneat.MetadataSender(metadata_options)
    print(f"rtsp={cfg.rtsp_url} stream={width}x{height}@{fps} prompt={cfg.prompt!r} tokens={tokens} "
          f"insight={cfg.insight_host} video={cfg.video_port} metadata={cfg.metadata_port} channel=0", flush=True)

    images = [np.empty((MODEL_SIZE, MODEL_SIZE, 3), np.float32) for _ in range(PIPELINE_DEPTH + 1)]
    overlay = OverlayClock()
    in_flight = deque()
    pushed = processed = window_instances = 0
    window_start = time.perf_counter()
    try:
        while cfg.frames <= 0 or processed < cfg.frames:
            if len(in_flight) < PIPELINE_DEPTH:
                sample = newest_frame(graph_run, overlay, 0 if in_flight else PULL_TIMEOUT_MS)
                if sample is None and not in_flight:
                    raise RuntimeError("timed out waiting for a frame")
                if sample is not None:
                    frame = nv12(sample.tensors[0])
                    image = images[pushed % len(images)]
                    preprocess(frame, image)
                    runner.push([ev74_tensor(image), text_tensor])
                    pushed += 1
                    save = cfg.save_dir and cfg.save_every and pushed % cfg.save_every == 0
                    in_flight.append((sample.pts_ns // 1_000_000, frame if save else None))
                    continue

            result = runner.pull(PULL_TIMEOUT_MS)
            if result is None:
                raise RuntimeError("timed out waiting for the model")
            pts_ms, saved_frame = in_flight.popleft()
            detections, masks = (tensor.to_numpy(copy=False) for tensor in result.tensors)
            segments = segments_of(detections[0], masks, cfg, width, height)
            result_json = json.dumps({"segments": segments}, separators=(",", ":"))
            for data, timestamp_ms, frame_id in overlay.add_result(pts_ms, result_json):
                metadata.send_metadata("segmentation", data, timestamp_ms, frame_id)

            processed += 1
            window_instances += len(segments)
            if saved_frame is not None:
                save_frame(Path(cfg.save_dir) / f"frame_{processed}.jpg", saved_frame, segments)
            if cfg.profile and processed % cfg.profile_interval == 0:
                elapsed = time.perf_counter() - window_start
                print(f"[profile] frames={cfg.profile_interval} "
                      f"segmentation_fps={cfg.profile_interval / elapsed:.1f} "
                      f"avg_instances={window_instances / cfg.profile_interval:.2f}", flush=True)
                window_start = time.perf_counter()
                window_instances = 0
    finally:
        graph_run.close()
        runner.close()
        print(f"processed={processed} video_sender={cfg.insight_host}:{cfg.video_port}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="EfficientSAM3 promptable segmentation on one RTSP stream, masks and video to Insight")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    cfg = load_config(parser.parse_args().config)
    if cfg.save_dir:
        Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)
    try:
        run(cfg)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
