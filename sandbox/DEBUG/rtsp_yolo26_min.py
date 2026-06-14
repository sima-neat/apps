#!/usr/bin/env python3
"""Minimal RTSP decode -> YOLO26 graph diagnostic.

This isolates the boundary under investigation:

  RtspDecodedInput -> Model(YOLO26) -> Output("detections")

It intentionally omits Branch, VideoSender, Insight metadata, overlays, and debug
frame saving. Run both modes to separate decoded-buffer handoff issues from
full-app fanout/video-branch issues.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import pyneat


DEFAULT_MODEL = "/workspace/sima-neat/apps/assets/models/yolo26m-det-bf16-mla_tess-b1.tar.gz"
DEFAULT_RTSP = "rtsp://192.168.2.1:8554/src1"


@dataclass(frozen=True)
class Variant:
    name: str
    next_element: bool
    output_caps: bool
    input_envelope: bool


def make_source(args: argparse.Namespace, variant: Variant):
    opt = pyneat.RtspDecodedInputOptions()
    opt.url = args.rtsp
    opt.latency_ms = args.latency_ms
    opt.tcp = True
    opt.payload_type = 96
    opt.insert_queue = True
    opt.out_format = pyneat.Format.NV12
    opt.decoder_name = "decoder"
    opt.decoder_raw_output = True
    opt.auto_caps_from_stream = True
    opt.fallback_h264_width = args.width
    opt.fallback_h264_height = args.height
    opt.fallback_h264_fps = args.fps

    if variant.next_element:
        opt.decoder_next_element = "CVU"

    if variant.output_caps:
        opt.output_caps.enable = True
        opt.output_caps.format = pyneat.Format.NV12
        opt.output_caps.width = args.width
        opt.output_caps.height = args.height
        opt.output_caps.fps = args.fps
        opt.output_caps.memory = pyneat.CapsMemory.Any

    return pyneat.groups.rtsp_decoded_input(opt)


def make_model(args: argparse.Namespace, variant: Variant):
    opt = pyneat.ModelOptions()
    opt.preprocess.kind = pyneat.InputKind.Image
    opt.preprocess.enable = pyneat.AutoFlag.On
    opt.preprocess.color_convert.input_format = pyneat.PreprocessColorFormat.NV12
    opt.preprocess.preset = pyneat.NormalizePreset.COCO_YOLO
    if variant.input_envelope:
        opt.preprocess.input_max_width = args.width
        opt.preprocess.input_max_height = args.height
        opt.preprocess.input_max_depth = args.input_max_depth
    opt.decode_type = pyneat.BoxDecodeType.YoloV26
    opt.score_threshold = args.score
    opt.nms_iou_threshold = args.nms_iou
    opt.top_k = args.top_k
    return pyneat.Model(args.model, opt)


def make_graph(args: argparse.Namespace, source, model):
    detections = pyneat.Graph("detections")
    detections.add(pyneat.nodes.output("detections", pyneat.OutputOptions.every_frame(4)))

    if args.graph_mode == "video-branch":
        branch = pyneat.graphs.branch("source", ["video", "model"])

        video_options = pyneat.VideoSenderOptions.h264_rtp_udp_from_raw(
            args.width, args.height, args.fps
        )
        video_options.host = args.insight_host
        video_options.channel = args.channel
        video_options.video_port_base = args.video_port
        video_options.encoder.bitrate_kbps = args.video_bitrate_kbps

        video_graph = pyneat.Graph("video")
        video_graph.connect(
            pyneat.nodes.input("video"), pyneat.groups.video_sender(video_options)
        )

        model_graph = pyneat.Graph("model")
        model_graph.connect(pyneat.nodes.input("model"), model)

        graph = pyneat.Graph("rtsp_decode_to_yolo26_with_video")
        graph.connect(source, branch)
        graph.connect(branch, video_graph)
        graph.connect(branch, model_graph)
        graph.connect(model_graph, detections)
        return graph

    graph = pyneat.Graph("rtsp_decode_to_yolo26")
    graph.connect(source, model)
    graph.connect(model, detections)
    return graph


def make_run_options(args: argparse.Namespace):
    opt = pyneat.RunOptions()
    opt.preset = pyneat.RunPreset.Realtime
    opt.queue_depth = args.queue_depth
    opt.overflow_policy = pyneat.OverflowPolicy.KeepLatest
    opt.output_memory = pyneat.OutputMemory.ZeroCopy
    return opt


def run_variant(args: argparse.Namespace, variant: Variant) -> int:
    print(f"=== variant={variant.name}")
    print(f"rtsp={args.rtsp} stream={args.width}x{args.height}@{args.fps}")
    print(f"model={args.model}")

    source = make_source(args, variant)
    model = make_model(args, variant)
    graph = make_graph(args, source, model)

    if args.describe_backend:
        print("Backend:")
        print(graph.describe_backend())

    run = graph.build(make_run_options(args))
    try:
        pulled = 0
        for frame_index in range(args.frames):
            sample = run.pull("detections", args.timeout_ms)
            if sample is None:
                print(f"timeout variant={variant.name} frame={frame_index}")
                continue
            pulled += 1
            print(
                f"ok variant={variant.name} frame={frame_index} "
                f"sample_frame_id={sample.frame_id}"
            )
        print(f"result variant={variant.name} pulled={pulled}/{args.frames}")
        return 0 if pulled > 0 else 2
    finally:
        run.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--rtsp", default=DEFAULT_RTSP)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--latency-ms", type=int, default=100)
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument("--timeout-ms", type=int, default=20000)
    parser.add_argument("--score", type=float, default=0.25)
    parser.add_argument("--nms-iou", type=float, default=0.45)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--queue-depth", type=int, default=3)
    parser.add_argument(
        "--graph-mode",
        choices=("minimal", "video-branch"),
        default="minimal",
        help="minimal omits the app video branch; video-branch adds Branch + VideoSender.",
    )
    parser.add_argument("--insight-host", default="192.168.2.1")
    parser.add_argument("--video-port", type=int, default=9000)
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--video-bitrate-kbps", type=int, default=1000)
    parser.add_argument(
        "--input-max-depth",
        type=int,
        default=1,
        help="Preprocess input_max_depth used by input-envelope variants.",
    )
    parser.add_argument(
        "--mode",
        choices=(
            "default",
            "next-only",
            "caps-only",
            "explicit",
            "envelope",
            "explicit-envelope",
            "matrix",
            "all",
        ),
        default="all",
        help=(
            "default matches the current app-like decoded input contract; next-only adds "
            "decoder_next_element=CVU; caps-only adds output caps; explicit adds both; envelope "
            "adds Model preprocess input_max_*; explicit-envelope adds all."
        ),
    )
    parser.add_argument(
        "--no-describe-backend",
        dest="describe_backend",
        action="store_false",
        help="Do not print graph.describe_backend().",
    )
    parser.set_defaults(describe_backend=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    variants = {
        "default": [Variant("default", False, False, False)],
        "next-only": [Variant("next-only", True, False, False)],
        "caps-only": [Variant("caps-only", False, True, False)],
        "explicit": [Variant("explicit", True, True, False)],
        "envelope": [Variant("envelope", False, False, True)],
        "explicit-envelope": [Variant("explicit-envelope", True, True, True)],
        "matrix": [
            Variant("default", False, False, False),
            Variant("next-only", True, False, False),
            Variant("caps-only", False, True, False),
            Variant("explicit", True, True, False),
        ],
        "all": [
            Variant("default", False, False, False),
            Variant("next-only", True, False, False),
            Variant("caps-only", False, True, False),
            Variant("explicit", True, True, False),
            Variant("envelope", False, False, True),
            Variant("explicit-envelope", True, True, True),
        ],
    }[args.mode]

    failures = 0
    for variant in variants:
        try:
            status = run_variant(args, variant)
        except Exception as exc:
            print(f"error variant={variant.name}: {exc}")
            status = 2
        failures += 1 if status != 0 else 0
    return 0 if failures == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
