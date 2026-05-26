#!/usr/bin/env python3
"""RTSP YOLOv8 to Insight and Gemma pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np
import pyneat

from utils.helpers import (
    Config,
    decoded_tensor_to_rgb,
    ev74_rgb_tensor,
    load_config,
    load_labels,
    metadata_json,
    parse_boxes,
    probe_rtsp,
)
from utils.openai_commenter import OpenAICommenter

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "common" / "config.yaml"

def build_rtsp_run(cfg: Config, width: int, height: int, fps: int):
    opt = pyneat.RtspDecodedInputOptions()
    opt.url = cfg.rtsp_url
    opt.fallback_h264_width = width
    opt.fallback_h264_height = height
    opt.fallback_h264_fps = fps
    opt.decoder_raw_output = False

    session = pyneat.Session()
    session.add(pyneat.groups.rtsp_decoded_input(opt))
    session.add(pyneat.nodes.output(pyneat.OutputOptions.every_frame(1)))

    run_opt = pyneat.RunOptions()
    run_opt.queue_depth = 4
    run_opt.overflow_policy = pyneat.OverflowPolicy.KeepLatest
    run_opt.output_memory = pyneat.OutputMemory.Owned
    return session, session.build(run_opt)

def build_video_run(cfg: Config, width: int, height: int, fps: int):
    input_opt = pyneat.InputOptions()
    input_opt.media_type = "video/x-raw"
    input_opt.format = "RGB"
    input_opt.use_simaai_pool = False
    input_opt.max_width = width
    input_opt.max_height = height
    input_opt.max_depth = 3

    sender_opt = pyneat.VideoSenderOptions.h264_rtp_udp_from_raw(width, height, max(1, fps))
    sender_opt.host = cfg.insight_host
    sender_opt.channel = cfg.channel
    sender_opt.video_port_base = cfg.video_port

    session = pyneat.Session()
    session.add(pyneat.nodes.input(input_opt))
    session.add(pyneat.groups.video_sender(sender_opt))
    seed = ev74_rgb_tensor(np.zeros((height, width, 3), dtype=np.uint8))
    return session, session.build(seed, pyneat.RunMode.Async)

def build_detector_run(cfg: Config, width: int, height: int):
    model = build_model(cfg, width, height)

    input_opt = model.input_appsrc_options(False)
    input_opt.media_type = "video/x-raw"
    input_opt.format = "RGB"
    input_opt.width = width
    input_opt.height = height
    input_opt.depth = 3
    for attr, value in (("max_width", width), ("max_height", height), ("max_depth", 3)):
        if hasattr(input_opt, attr):
            setattr(input_opt, attr, value)

    session = pyneat.Session()
    session.add(pyneat.nodes.input(input_opt))
    session.add(model.preprocess())
    session.add(pyneat.groups.mla(model))
    session.add(
        pyneat.nodes.sima_box_decode(
            model,
            decode_type=pyneat.BoxDecodeType.YoloV8,
            original_width=width,
            original_height=height,
            detection_threshold=cfg.min_score,
            nms_iou_threshold=cfg.nms_iou,
            top_k=cfg.max_detections,
        )
    )
    session.add(pyneat.nodes.output())

    seed = ev74_rgb_tensor(np.zeros((height, width, 3), dtype=np.uint8))
    run_opt = pyneat.RunOptions()
    run_opt.queue_depth = 1
    run_opt.overflow_policy = pyneat.OverflowPolicy.KeepLatest
    run_opt.output_memory = pyneat.OutputMemory.Owned
    return session, session.build(seed, pyneat.RunMode.Async, run_opt)

def build_model(cfg: Config, width: int, height: int):
    opt = pyneat.ModelOptions()
    opt.preprocess.kind = pyneat.InputKind.Image
    opt.preprocess.input_max_width = width
    opt.preprocess.input_max_height = height
    opt.preprocess.input_max_depth = 3
    opt.preprocess.color_convert.input_format = pyneat.PreprocessColorFormat.RGB
    opt.preprocess.preset = pyneat.NormalizePreset.COCO_YOLO
    opt.decode_type = pyneat.BoxDecodeType.YoloV8
    opt.score_threshold = cfg.min_score
    opt.nms_iou_threshold = cfg.nms_iou
    opt.top_k = cfg.max_detections
    opt.boxdecode_original_width = width
    opt.boxdecode_original_height = height
    return pyneat.Model(cfg.model_path, opt)

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

    rtsp_run = detector_run = video_run = commenter = None
    try:
        labels = load_labels(cfg.labels_path)
        width, height, fps = probe_rtsp(cfg.rtsp_url)
        _rtsp_session, rtsp_run = build_rtsp_run(cfg, width, height, fps)
        _detector_session, detector_run = build_detector_run(cfg, width, height)
        _video_session, video_run = build_video_run(cfg, width, height, fps)
        metadata = build_metadata_sender(cfg)
        commenter = OpenAICommenter(cfg, labels)
        commenter.start()
        print(
            f"rtsp={cfg.rtsp_url} stream={width}x{height}@{fps} "
            f"insight={cfg.insight_host} video={cfg.video_port} "
            f"metadata={cfg.metadata_port} channel={cfg.channel}"
        )

        frame_id = 0
        while cfg.frames <= 0 or frame_id < cfg.frames:
            tensors = rtsp_run.pull_tensors(timeout_ms=cfg.timeout_ms)
            if not tensors:
                print("RTSP stream ended or pull timed out", file=sys.stderr)
                break
            rgb_frame = decoded_tensor_to_rgb(tensors[0])
            if not detector_run.push_tensor(ev74_rgb_tensor(rgb_frame)):
                raise RuntimeError("detector push failed")
            samples = detector_run.pull_samples(timeout_ms=cfg.timeout_ms)
            if not samples:
                raise RuntimeError("detector pull timed out")
            boxes = parse_boxes(samples)
            commenter.try_enqueue(rgb_frame, boxes)
            if not video_run.push(rgb_frame, copy=True, image_format=pyneat.PixelFormat.RGB):
                raise RuntimeError("Insight video push failed")
            ok = metadata.send_metadata(
                "object-detection",
                metadata_json(boxes, labels, cfg.classes),
                int(time.time() * 1000),
                str(frame_id),
            )
            if not ok:
                raise RuntimeError("failed to send metadata to Insight")
            frame_id += 1
            if cfg.debug:
                print(f"frame={frame_id} detections={len(boxes)}")
        return 0 if frame_id > 0 else 3
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    finally:
        if commenter is not None:
            commenter.close()
        for run in (video_run, detector_run, rtsp_run):
            if run is not None:
                run.close()


if __name__ == "__main__":
    raise SystemExit(main())
