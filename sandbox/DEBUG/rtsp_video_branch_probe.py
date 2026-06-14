#!/usr/bin/env python3
"""Probe RTSP decoded branching and VideoSender behavior.

The cases intentionally differ by one graph boundary at a time:

  direct-model              RTSP -> Model -> Output
  branch-model              RTSP -> Branch(model) -> Model -> Output
  branch-model-raw-unpulled RTSP -> Branch(model, raw) -> Model + raw Output
  branch-model-raw-pulled   same as above, but pulls raw frames too
  video-only                RTSP -> VideoSender
  branch-model-video        RTSP -> Branch(model, video) -> Model + VideoSender

This is a diagnostic script for Modalix. It is not part of the example contract.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import time

import pyneat


DEFAULT_MODEL = (
    "/workspace/sima-neat/apps/assets/models/YOLO26-DETECTION/"
    "yolo26m-det-int8-b1.tar.gz"
)
DEFAULT_STREAMS = [
    "rtsp://192.168.2.1:8554/src1",
    "rtsp://192.168.2.1:8554/src2",
    "rtsp://192.168.2.1:8554/src3",
    "rtsp://192.168.2.1:8554/src4",
]


@dataclass
class ProbeRun:
    stream_index: int
    case: str
    run: object
    has_detections: bool
    has_raw: bool


def make_source(args: argparse.Namespace, url: str):
    opt = pyneat.RtspDecodedInputOptions()
    opt.url = url
    opt.latency_ms = args.latency_ms
    opt.tcp = True
    opt.payload_type = 96
    opt.insert_queue = True
    opt.decoder_name = "decoder"
    opt.decoder_raw_output = True
    opt.auto_caps_from_stream = True
    opt.fallback_h264_width = args.width
    opt.fallback_h264_height = args.height
    opt.fallback_h264_fps = args.fps
    opt.output_caps.enable = True
    opt.output_caps.format = pyneat.Format.NV12
    opt.output_caps.width = args.width
    opt.output_caps.height = args.height
    opt.output_caps.fps = args.fps
    opt.output_caps.memory = pyneat.CapsMemory.Any
    return pyneat.groups.rtsp_decoded_input(opt)


def make_model(args: argparse.Namespace):
    opt = pyneat.ModelOptions()
    opt.preprocess.kind = pyneat.InputKind.Image
    opt.preprocess.enable = pyneat.AutoFlag.On
    opt.preprocess.color_convert.input_format = pyneat.PreprocessColorFormat.NV12
    opt.preprocess.preset = pyneat.NormalizePreset.COCO_YOLO
    opt.decode_type = pyneat.BoxDecodeType.YoloV26
    opt.score_threshold = args.score
    opt.nms_iou_threshold = args.nms_iou
    opt.top_k = args.top_k
    return pyneat.Model(args.model, opt)


def detections_output(name: str = "detections"):
    graph = pyneat.Graph(name)
    graph.add(pyneat.nodes.output(name, pyneat.OutputOptions.every_frame(4)))
    return graph


def raw_output(name: str = "raw"):
    graph = pyneat.Graph(name)
    graph.add(pyneat.nodes.output(name, pyneat.OutputOptions.every_frame(4)))
    return graph


def make_video_sender(args: argparse.Namespace, stream_index: int):
    options = pyneat.VideoSenderOptions.h264_rtp_udp_from_raw(
        args.width, args.height, args.fps
    )
    options.host = args.video_host
    options.channel = stream_index
    options.video_port_base = args.video_port_base
    options.encoder.bitrate_kbps = args.video_bitrate_kbps

    graph = pyneat.Graph("video")
    graph.connect(pyneat.nodes.input("video"), pyneat.groups.video_sender(options))
    return graph


def make_graph(args: argparse.Namespace, case: str, stream_index: int, url: str):
    source = make_source(args, url)
    graph = pyneat.Graph(f"{case}_{stream_index}")

    if case == "direct-model":
        model = make_model(args)
        graph.connect(source, model)
        graph.connect(model, detections_output())
        return graph, True, False

    if case == "branch-model":
        model = make_model(args)
        branch = pyneat.graphs.branch("source", ["model"])
        model_graph = pyneat.Graph("model")
        model_graph.connect(pyneat.nodes.input("model"), model)
        graph.connect(source, branch)
        graph.connect(branch, model_graph)
        graph.connect(model_graph, detections_output())
        return graph, True, False

    if case in ("branch-model-raw-unpulled", "branch-model-raw-pulled"):
        model = make_model(args)
        branch = pyneat.graphs.branch("source", ["model", "raw"])
        model_graph = pyneat.Graph("model")
        model_graph.connect(pyneat.nodes.input("model"), model)
        graph.connect(source, branch)
        graph.connect(branch, model_graph)
        graph.connect(model_graph, detections_output())
        graph.connect(branch, raw_output())
        return graph, True, case == "branch-model-raw-pulled"

    if case == "video-only":
        video_graph = make_video_sender(args, stream_index)
        graph.connect(source, video_graph)
        return graph, False, False

    if case == "branch-model-video":
        model = make_model(args)
        branch = pyneat.graphs.branch("source", ["model", "video"])
        model_graph = pyneat.Graph("model")
        model_graph.connect(pyneat.nodes.input("model"), model)
        graph.connect(source, branch)
        graph.connect(branch, model_graph)
        graph.connect(model_graph, detections_output())
        graph.connect(branch, make_video_sender(args, stream_index))
        return graph, True, False

    raise ValueError(f"unknown case: {case}")


def make_run_options(args: argparse.Namespace):
    opt = pyneat.RunOptions()
    opt.preset = pyneat.RunPreset.Realtime
    opt.queue_depth = args.queue_depth
    opt.overflow_policy = pyneat.OverflowPolicy.KeepLatest
    opt.output_memory = pyneat.OutputMemory.ZeroCopy
    return opt


def print_runtime_state(run: object) -> None:
    for method_name in ("stats", "report", "metrics_report"):
        method = getattr(run, method_name, None)
        if not callable(method):
            continue
        try:
            print(f"--- run.{method_name}()")
            print(method())
        except Exception as exc:
            print(f"--- run.{method_name}() failed: {exc}")


def build_probe(args: argparse.Namespace, case: str, stream_index: int, url: str) -> ProbeRun:
    print(f"=== build case={case} stream={stream_index} url={url}", flush=True)
    graph, has_detections, has_raw = make_graph(args, case, stream_index, url)
    if args.describe_backend:
        print(graph.describe_backend(), flush=True)
    run = graph.build(make_run_options(args))
    return ProbeRun(stream_index, case, run, has_detections, has_raw)


def pull_probe(args: argparse.Namespace, probe: ProbeRun) -> tuple[int, int]:
    detections = 0
    raw = 0
    for frame in range(args.frames):
        if probe.has_raw:
            raw_sample = probe.run.pull("raw", args.raw_timeout_ms)
            if raw_sample is not None:
                raw += 1
                print(
                    f"ok case={probe.case} stream={probe.stream_index} "
                    f"raw_frame={frame} sample_frame_id={raw_sample.frame_id}",
                    flush=True,
                )
            else:
                print(
                    f"timeout case={probe.case} stream={probe.stream_index} raw_frame={frame}",
                    flush=True,
                )

        sample = probe.run.pull("detections", args.timeout_ms)
        if sample is not None:
            detections += 1
            print(
                f"ok case={probe.case} stream={probe.stream_index} "
                f"det_frame={frame} sample_frame_id={sample.frame_id}",
                flush=True,
            )
        else:
            print(
                f"timeout case={probe.case} stream={probe.stream_index} det_frame={frame}",
                flush=True,
            )
    return detections, raw


def run_case(args: argparse.Namespace, case: str) -> int:
    streams = args.streams[: args.stream_count]
    probes: list[ProbeRun] = []
    try:
        for stream_index, url in enumerate(streams):
            probes.append(build_probe(args, case, stream_index, url))

        if case == "video-only":
            print(
                f"sleep case={case} streams={len(probes)} seconds={args.video_seconds}",
                flush=True,
            )
            time.sleep(args.video_seconds)
            for probe in probes:
                print_runtime_state(probe.run)
            print(f"result case={case} built={len(probes)}", flush=True)
            return 0

        total_detections = 0
        total_raw = 0
        for probe in probes:
            detections, raw = pull_probe(args, probe)
            total_detections += detections
            total_raw += raw
            print_runtime_state(probe.run)

        expected = args.frames * len(probes)
        print(
            f"result case={case} detections={total_detections}/{expected} raw={total_raw}",
            flush=True,
        )
        return 0 if total_detections > 0 else 2
    finally:
        for probe in reversed(probes):
            try:
                probe.run.close()
            except Exception as exc:
                print(
                    f"close failed case={probe.case} stream={probe.stream_index}: {exc}",
                    flush=True,
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        choices=(
            "direct-model",
            "branch-model",
            "branch-model-raw-unpulled",
            "branch-model-raw-pulled",
            "video-only",
            "branch-model-video",
            "all",
        ),
        default="all",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--streams", nargs="+", default=DEFAULT_STREAMS)
    parser.add_argument("--stream-count", type=int, default=1)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--latency-ms", type=int, default=100)
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument("--timeout-ms", type=int, default=20000)
    parser.add_argument("--raw-timeout-ms", type=int, default=2000)
    parser.add_argument("--video-seconds", type=float, default=10.0)
    parser.add_argument("--score", type=float, default=0.30)
    parser.add_argument("--nms-iou", type=float, default=0.60)
    parser.add_argument("--top-k", type=int, default=24)
    parser.add_argument("--queue-depth", type=int, default=3)
    parser.add_argument("--video-host", default="192.168.2.1")
    parser.add_argument("--video-port-base", type=int, default=9000)
    parser.add_argument("--video-bitrate-kbps", type=int, default=1000)
    parser.add_argument("--no-describe-backend", dest="describe_backend", action="store_false")
    parser.set_defaults(describe_backend=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.stream_count < 1 or args.stream_count > len(args.streams):
        raise SystemExit("--stream-count must be between 1 and len(--streams)")

    cases = (
        [
            "direct-model",
            "branch-model",
            "branch-model-raw-unpulled",
            "branch-model-raw-pulled",
            "video-only",
            "branch-model-video",
        ]
        if args.case == "all"
        else [args.case]
    )

    failures = 0
    for case in cases:
        try:
            failures += 1 if run_case(args, case) != 0 else 0
        except Exception as exc:
            failures += 1
            print(f"error case={case}: {exc}", flush=True)
    return 0 if failures == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
