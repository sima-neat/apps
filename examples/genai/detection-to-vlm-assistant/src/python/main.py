#!/usr/bin/env python3
"""RTSP YOLO26 to Insight and Gemma pipeline."""

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
    load_config,
    load_labels,
    metadata_json,
    parse_boxes,
    probe_rtsp,
    rgb_to_bgr,
)
from utils.openai_commenter import OpenAICommenter

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "common" / "config.yaml"


def build_rtsp_run(cfg: Config, width: int, height: int, fps: int):
    opt = pyneat.RtspDecodedInputOptions()
    opt.url = cfg.rtsp_url
    opt.payload_type = 96
    opt.insert_queue = True
    opt.auto_caps_from_stream = True
    opt.fallback_h264_width = width
    opt.fallback_h264_height = height
    opt.fallback_h264_fps = fps
    opt.sima_allocator_type = 2
    opt.decoder_raw_output = False
    opt.use_videoconvert = False
    opt.use_videoscale = True
    opt.output_caps.enable = True
    opt.output_caps.width = width
    opt.output_caps.height = height
    opt.output_caps.fps = fps
    opt.output_caps.memory = pyneat.CapsMemory.SystemMemory

    graph = pyneat.Graph("rtsp_source")
    graph.add(pyneat.groups.rtsp_decoded_input(opt))
    graph.add(pyneat.nodes.output(pyneat.OutputOptions.every_frame(1)))

    run_opt = pyneat.RunOptions()
    run_opt.queue_depth = 4
    run_opt.overflow_policy = pyneat.OverflowPolicy.KeepLatest
    run_opt.output_memory = pyneat.OutputMemory.Owned
    return graph, graph.build(run_opt)


def build_video_run(cfg: Config, width: int, height: int, fps: int):
    input_opt = pyneat.InputOptions()
    input_opt.payload_type = pyneat.PayloadType.Image
    input_opt.format = "RGB"
    input_opt.width = width
    input_opt.height = height
    input_opt.depth = 3
    input_opt.fps_n = max(1, fps)
    input_opt.fps_d = 1
    input_opt.caps_override = (
        f"video/x-raw,format=RGB,width={width},height={height},framerate={max(1, fps)}/1"
    )
    input_opt.use_simaai_pool = False

    sender_opt = pyneat.VideoSenderOptions.h264_rtp_udp_from_raw(width, height, max(1, fps))
    sender_opt.host = cfg.insight_host
    sender_opt.channel = cfg.channel
    sender_opt.video_port_base = cfg.video_port

    graph = pyneat.Graph("insight_video")
    graph.add(pyneat.nodes.input(input_opt))
    graph.add(pyneat.groups.video_sender(sender_opt))
    seed = pyneat.Tensor.from_numpy(
        np.zeros((height, width, 3), dtype=np.uint8),
        copy=True,
        image_format=pyneat.PixelFormat.RGB,
        memory=pyneat.TensorMemory.CPU,
    )
    return graph, graph.build([seed])


def build_model(cfg: Config):
    opt = pyneat.ModelOptions()
    opt.preprocess.kind = pyneat.InputKind.Image
    opt.preprocess.enable = pyneat.AutoFlag.On
    opt.preprocess.color_convert.input_format = pyneat.PreprocessColorFormat.BGR
    opt.preprocess.preset = pyneat.NormalizePreset.COCO_YOLO
    opt.decode_type = pyneat.BoxDecodeType.YoloV26
    opt.score_threshold = cfg.min_score
    opt.nms_iou_threshold = cfg.nms_iou
    opt.top_k = cfg.max_detections
    return pyneat.Model(cfg.model_path, opt)


def build_detector_run(cfg: Config, width: int, height: int):
    model = build_model(cfg)

    input_opt = model.input_appsrc_options(False)
    input_opt.payload_type = pyneat.PayloadType.Image
    input_opt.format = "BGR"
    input_opt.width = width
    input_opt.height = height
    input_opt.depth = 3

    graph = pyneat.Graph("detector")
    graph.add(pyneat.nodes.input(input_opt))
    graph.add(model.graph())
    graph.add(pyneat.nodes.output())

    seed = pyneat.Tensor.from_numpy(
        np.zeros((height, width, 3), dtype=np.uint8),
        copy=True,
        image_format=pyneat.PixelFormat.BGR,
        memory=pyneat.TensorMemory.EV74,
    )
    run_opt = pyneat.RunOptions()
    run_opt.queue_depth = 4
    run_opt.overflow_policy = pyneat.OverflowPolicy.KeepLatest
    run_opt.output_memory = pyneat.OutputMemory.Owned
    return model, graph, graph.build([seed], run_opt)


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
        _model, _detector_graph, detector_run = build_detector_run(cfg, width, height)
        _rtsp_graph, rtsp_run = build_rtsp_run(cfg, width, height, fps)
        _video_graph, video_run = build_video_run(cfg, width, height, fps)
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
            decoded_frame = tensors[0]
            rgb_frame = decoded_tensor_to_rgb(decoded_frame)
            model_input = pyneat.Tensor.from_numpy(
                rgb_to_bgr(rgb_frame),
                copy=True,
                image_format=pyneat.PixelFormat.BGR,
                memory=pyneat.TensorMemory.EV74,
            )
            boxes = parse_boxes(detector_run.run([model_input], timeout_ms=cfg.timeout_ms))
            commenter.try_enqueue(rgb_frame, boxes)
            if not video_run.push(
                [rgb_frame], copy=True, image_format=pyneat.PixelFormat.RGB
            ):
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
    except KeyboardInterrupt:
        return 130
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
