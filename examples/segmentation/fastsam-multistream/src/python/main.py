"""FastSAM + MobileCLIP text-prompt segmentation across RTSP cameras on the SiMa MLA, via pyneat."""
import json
import os
import signal
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pyneat as neat

import fastsam
from clip import ImageEncoder, TextEncoder
from config import load_config

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "common" / "config.yaml"
PULL_TIMEOUT_MS = 100

_stop = False


def _on_sigint(signum, frame):
    global _stop
    _stop = True


def _source_options(cfg, url, width, height, fps):
    opt = neat.RtspDecodedInputOptions()
    opt.url = url
    opt.latency_ms = cfg.latency_ms
    opt.tcp = cfg.tcp
    opt.payload_type = 96
    opt.insert_queue = True
    opt.out_format = neat.Format.NV12
    opt.decoder_name = "decoder"
    opt.decoder_raw_output = True
    opt.auto_caps_from_stream = True
    opt.fallback_h264_width = width
    opt.fallback_h264_height = height
    opt.fallback_h264_fps = fps
    opt.output_caps.enable = True
    opt.output_caps.format = neat.Format.NV12
    opt.output_caps.width = width
    opt.output_caps.height = height
    opt.output_caps.fps = fps
    opt.output_caps.memory = neat.CapsMemory.Any
    return opt


def _build_run_options(cfg):
    opt = neat.RunOptions()
    opt.queue_depth = cfg.queue_depth
    opt.overflow_policy = neat.OverflowPolicy.Block
    opt.preset = neat.RunPreset.Balanced
    opt.input_timeout_ms = 30000
    opt.startup_preflight = True
    return opt


def _realtime_run_options():
    opt = neat.RunOptions()
    opt.preset = neat.RunPreset.Realtime
    opt.queue_depth = 3
    opt.overflow_policy = neat.OverflowPolicy.KeepLatest
    opt.output_memory = neat.OutputMemory.Owned
    return opt


def _build_source_run(cfg, url, width, height, fps):
    graph = neat.Graph("source")
    graph.add(neat.groups.rtsp_decoded_input(_source_options(cfg, url, width, height, fps)))
    graph.add(neat.nodes.output(neat.OutputOptions.latest()))
    return graph.build(_realtime_run_options())


def _build_video_run(cfg, index, width, height, fps):
    input_opt = neat.InputOptions()
    input_opt.payload_type = neat.PayloadType.Image
    input_opt.format = neat.Format.RGB
    input_opt.width = width
    input_opt.height = height
    input_opt.depth = 3
    input_opt.fps_n = max(1, fps)
    input_opt.fps_d = 1
    input_opt.caps_override = (
        f"video/x-raw,format=RGB,width={width},height={height},framerate={max(1, fps)}/1"
    )
    input_opt.memory_policy = neat.InputMemoryPolicy.Ev74

    sender_opt = neat.VideoSenderOptions.h264_rtp_udp_from_raw(width, height, max(1, fps))
    sender_opt.host = cfg.insight_host
    sender_opt.channel = index
    sender_opt.video_port_base = cfg.video_port_base
    sender_opt.encoder.bitrate_kbps = cfg.bitrate_kbps

    graph = neat.Graph("video")
    graph.add(neat.nodes.input(input_opt))
    graph.add(neat.groups.video_sender(sender_opt))
    seed = neat.Tensor.from_numpy(np.zeros((height, width, 3), np.uint8), copy=True,
                                  image_format=neat.PixelFormat.RGB, memory=neat.TensorMemory.EV74)
    return graph.build([seed])


def _metadata_sender(cfg, index):
    opt = neat.MetadataSenderOptions()
    opt.host = cfg.insight_host
    opt.channel = index
    opt.metadata_port_base = cfg.metadata_port_base
    return neat.MetadataSender(opt)


def _probe_rtsp(url):
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(round(cap.get(cv2.CAP_PROP_FPS)))
        ok = cap.isOpened() and width > 0 and height > 0
    finally:
        cap.release()
    if not ok:
        raise RuntimeError("failed to probe RTSP stream: " + url)
    return width, height, fps if fps > 0 else 30


def _frame_rgb(tensor, width, height):
    need = width * height * 3 // 2
    payload = tensor.contiguous().copy_payload_bytes()
    if len(payload) < need:
        raise RuntimeError("decoded frame payload smaller than expected NV12 size")
    nv12 = np.frombuffer(payload, np.uint8, count=need).reshape(height * 3 // 2, width)
    return cv2.cvtColor(nv12, cv2.COLOR_YUV2RGB_NV12)


def _push_video(video_run, rgb, sample):
    tensor = neat.Tensor.from_numpy(rgb, copy=True, image_format=neat.PixelFormat.RGB,
                                    memory=neat.TensorMemory.EV74)
    video_sample = neat.make_tensor_sample("", tensor)
    video_sample.pts_ns = sample.pts_ns
    video_sample.dts_ns = sample.dts_ns
    video_sample.duration_ns = sample.duration_ns
    video_sample.frame_id = sample.frame_id
    video_sample.stream_id = sample.stream_id
    if not video_run.push([video_sample]):
        raise RuntimeError("video push failed")


def _detect(cfg, rgb, geom, fastsam_model, image_encoder, text_query):
    seg = fastsam.decode(fastsam_model.run(rgb, cfg.timeout_ms), max(0, cfg.max_crops))
    candidates = []
    for i, mask in enumerate(seg.masks):
        crop = fastsam.object_crop(mask, rgb, geom, max_box_frac=cfg.max_box_frac)
        if crop is not None:
            candidates.append((i, crop))

    best = image_encoder.best_match(candidates, text_query, cfg.min_score, cfg.timeout_ms)
    if best is None:
        return None
    polygon = fastsam.mask_polygon(seg.masks[best], geom)
    if not polygon:
        return None
    return float(seg.boxes[best][4]), polygon


def _segments_json(match, label):
    segments = []
    if match is not None:
        confidence, polygon = match
        segments.append({
            "id": "seg_1",
            "label": label,
            "confidence": confidence,
            "mask_format": "polygon",
            "mask": [[int(p[0]), int(p[1])] for p in polygon],
        })
    return json.dumps({"segments": segments})


def _send_metadata(sender, payload, frame_id, timestamp_ms):
    try:
        sender.send_metadata("segmentation", payload, timestamp_ms, frame_id)
    except RuntimeError as ex:
        print(f"[warn] metadata send failed: {ex}", file=sys.stderr)


class _Stream:
    def __init__(self, index, source_run, video_run, geom, sender, width, height):
        self.index = index
        self.source_run = source_run
        self.video_run = video_run
        self.geom = geom
        self.sender = sender
        self.width = width
        self.height = height
        self.processed = 0
        self.closed = False
        self._lock = threading.Lock()
        self._frame = None
        self._match = None

    def hand_off(self, rgb):
        with self._lock:
            self._frame = rgb

    def take_frame(self):
        with self._lock:
            rgb, self._frame = self._frame, None
            return rgb

    def publish(self, match):
        with self._lock:
            self._match = match

    def latest_match(self):
        with self._lock:
            return self._match


def _detector_loop(streams, cfg, fastsam_model, image_encoder, text_query, stop_event):
    while not _stop and not stop_event.is_set():
        worked = False
        for stream in streams:
            if stream.closed:
                continue
            rgb = stream.take_frame()
            if rgb is None:
                continue
            worked = True
            stream.publish(_detect(cfg, rgb, stream.geom, fastsam_model, image_encoder, text_query))
        if not worked:
            time.sleep(0.001)


def main():
    signal.signal(signal.SIGINT, _on_sigint)
    try:
        cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else str(DEFAULT_CONFIG))
        run_opt = _build_run_options(cfg)

        print(f"[build] encoding prompt {cfg.text!r} via CLIP text tower", flush=True)
        text_encoder = TextEncoder(cfg.clip_text_path, cfg.clip_consts_path, run_opt)
        try:
            text_query = text_encoder.encode(cfg.text, cfg.timeout_ms)[0]
        finally:
            text_encoder.close()

        probes = []
        max_w = max_h = 0
        for url in cfg.rtsp_urls:
            width, height, fps = _probe_rtsp(url)
            print(f"[rtsp] {url} {width}x{height}@{fps}")
            max_w = max(max_w, width)
            max_h = max(max_h, height)
            probes.append((url, width, height, fps))

        fastsam_model = fastsam.Fastsam(cfg, run_opt, max_w, max_h)
        image_encoder = ImageEncoder(cfg.clip_image_path, run_opt)

        streams = []
        for index, (url, width, height, fps) in enumerate(probes):
            source_run = _build_source_run(cfg, url, width, height, fps)
            video_run = (
                _build_video_run(cfg, index, width, height, fps) if cfg.video_enabled else None
            )
            geom = fastsam.letterbox_geometry(width, height, cfg.infer_size)
            sender = _metadata_sender(cfg, index)
            streams.append(_Stream(index, source_run, video_run, geom, sender, width, height))
            print(f"[stream {index}] {url} {width}x{height}@{fps} "
                  f"metadata={sender.metadata_port()}")

        stop_event = threading.Event()
        detector = threading.Thread(
            target=_detector_loop,
            args=(streams, cfg, fastsam_model, image_encoder, text_query, stop_event),
            daemon=True,
        )
        detector.start()

        while not _stop and not all(s.closed for s in streams):
            worked = False
            for s in streams:
                if s.closed:
                    continue
                sample = s.source_run.pull(timeout_ms=PULL_TIMEOUT_MS)
                if sample is None:
                    if not s.source_run.can_pull():
                        s.closed = True
                    continue
                if not sample.tensors:
                    continue
                worked = True
                s.processed += 1
                rgb = _frame_rgb(sample.tensors[0], s.width, s.height)
                s.hand_off(rgb)
                if s.video_run is not None:
                    _push_video(s.video_run, rgb, sample)
                ts_ms = sample.pts_ns // 1_000_000 if sample.pts_ns >= 0 else -1
                frame_id = str(sample.frame_id) if sample.frame_id >= 0 else ""
                _send_metadata(s.sender, _segments_json(s.latest_match(), cfg.text), frame_id, ts_ms)
                if cfg.frames > 0 and s.processed >= cfg.frames:
                    s.closed = True
            if not worked and not all(s.closed for s in streams):
                time.sleep(0.001)

        stop_event.set()
        detector.join(timeout=2.0)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    except Exception as ex:
        print(f"Error: {ex}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
