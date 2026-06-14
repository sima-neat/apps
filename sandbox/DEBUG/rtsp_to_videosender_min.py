#!/usr/bin/env python3
"""Minimal RTSP decoded input to VideoSender probe.

This isolates the raw VideoSender encoder path from model, metadata, BoxDecode,
and app scheduling logic.
"""

from __future__ import annotations

import argparse
import time

import pyneat


DEFAULT_RTSP_URL = "rtsp://192.168.2.1:25689/src1"
DEFAULT_INSIGHT_HOST = "192.168.2.1"


def make_source_options(args: argparse.Namespace) -> pyneat.RtspDecodedInputOptions:
    opt = pyneat.RtspDecodedInputOptions()
    opt.url = args.rtsp_url
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
    return opt


def make_video_options(
    args: argparse.Namespace, channel: int
) -> pyneat.VideoSenderOptions:
    opt = pyneat.VideoSenderOptions.h264_rtp_udp_from_raw(
        args.width, args.height, args.fps
    )
    opt.host = args.insight_host
    opt.channel = channel
    opt.video_port_base = args.video_port_base
    opt.encoder.bitrate_kbps = args.bitrate_kbps
    return opt


def make_run_options() -> pyneat.RunOptions:
    opt = pyneat.RunOptions()
    opt.preset = pyneat.RunPreset.Realtime
    opt.queue_depth = 3
    opt.overflow_policy = pyneat.OverflowPolicy.KeepLatest
    opt.output_memory = pyneat.OutputMemory.ZeroCopy
    return opt


def make_link_options(args: argparse.Namespace) -> pyneat.GraphLinkOptions:
    opt = pyneat.GraphLinkOptions()
    if args.link_policy == "realtime":
        opt.policy = pyneat.GraphLinkPolicy.RealtimeLatestByStream
    return opt


def video_sender_graph(
    args: argparse.Namespace, input_name: str, options: pyneat.VideoSenderOptions
) -> pyneat.Graph:
    graph = pyneat.Graph(input_name)
    video_input = pyneat.nodes.input(input_name)
    if args.sender == "group":
        graph.connect(video_input, pyneat.groups.video_sender(options))
        return graph

    input_format = "" if args.sender == "manual-any" else "NV12"
    input_caps = pyneat.nodes.caps_raw(
        input_format, width=args.width, height=args.height, fps=args.fps
    )
    convert = pyneat.nodes.video_convert()
    encoder_caps = pyneat.nodes.caps_raw(
        "NV12", width=args.width, height=args.height, fps=args.fps
    )
    encoder = pyneat.nodes.h264_encode_sima(
        args.width,
        args.height,
        args.fps,
        bitrate_kbps=options.encoder.bitrate_kbps,
        profile=options.encoder.profile,
        level=options.encoder.level,
    )
    parser = pyneat.nodes.h264_parse(options.rtp.config_interval)
    packetizer = pyneat.nodes.h264_packetize(
        payload_type=options.rtp.payload_type,
        config_interval=options.rtp.config_interval,
    )
    udp_options = pyneat.UdpOutputOptions()
    udp_options.host = options.host
    udp_options.port = options.video_port
    udp_options.sync = options.sync
    udp_options.async_ = options.async_
    udp = pyneat.nodes.udp_output(udp_options)

    graph.connect(video_input, input_caps)
    graph.connect(input_caps, convert)
    graph.connect(convert, encoder_caps)
    graph.connect(encoder_caps, encoder)
    graph.connect(encoder, parser)
    graph.connect(parser, packetizer)
    graph.connect(packetizer, udp)
    return graph


def output_graph(name: str) -> pyneat.Graph:
    graph = pyneat.Graph(name)
    graph.add(pyneat.nodes.output(name, pyneat.OutputOptions.every_frame(4)))
    return graph


def make_graph(
    args: argparse.Namespace,
) -> tuple[pyneat.Graph, list[pyneat.VideoSenderOptions], str | None]:
    graph = pyneat.Graph("rtsp_to_videosender_min")
    source = pyneat.groups.rtsp_decoded_input(make_source_options(args))
    link_options = make_link_options(args)

    if args.mode == "single":
        video_options = make_video_options(args, args.channel)
        graph.connect(source, pyneat.groups.video_sender(video_options))
        return graph, [video_options], None

    if args.mode == "branch-one":
        branch = pyneat.graphs.branch("source", ["video0"])
        video_options = make_video_options(args, args.channel)
        graph.connect(source, branch)
        graph.connect(branch, video_sender_graph(args, "video0", video_options), link_options)
        return graph, [video_options], None

    if args.mode == "branch-video-probe":
        branch = pyneat.graphs.branch("source", ["video0", "probe"])
        video_options = make_video_options(args, args.channel)
        graph.connect(source, branch)
        graph.connect(branch, video_sender_graph(args, "video0", video_options), link_options)
        graph.connect(branch, output_graph("probe"), link_options)
        return graph, [video_options], "probe"

    if args.mode == "branch-probe-video":
        branch = pyneat.graphs.branch("source", ["probe", "video0"])
        video_options = make_video_options(args, args.channel)
        graph.connect(source, branch)
        graph.connect(branch, output_graph("probe"), link_options)
        graph.connect(branch, video_sender_graph(args, "video0", video_options), link_options)
        return graph, [video_options], "probe"

    branch = pyneat.graphs.branch("source", ["video0", "video1"])
    first = make_video_options(args, args.channel)
    second = make_video_options(args, args.channel + 1)

    graph.connect(source, branch)
    graph.connect(branch, video_sender_graph(args, "video0", first), link_options)
    graph.connect(branch, video_sender_graph(args, "video1", second), link_options)
    return graph, [first, second], None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=(
            "single",
            "branch-one",
            "branch-two",
            "branch-video-probe",
            "branch-probe-video",
        ),
        default="single",
    )
    parser.add_argument("--link-policy", choices=("default", "realtime"), default="default")
    parser.add_argument(
        "--sender", choices=("group", "manual-any", "manual-nv12"), default="group"
    )
    parser.add_argument("--rtsp-url", default=DEFAULT_RTSP_URL)
    parser.add_argument("--insight-host", default=DEFAULT_INSIGHT_HOST)
    parser.add_argument("--video-port-base", type=int, default=19187)
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--latency-ms", type=int, default=100)
    parser.add_argument("--bitrate-kbps", type=int, default=1000)
    parser.add_argument("--no-describe-backend", dest="describe_backend", action="store_false")
    parser.set_defaults(describe_backend=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    graph, video_options, probe_output = make_graph(args)

    print(f"pyneat_version={getattr(pyneat, '__version__', '<unknown>')}", flush=True)
    print(f"pyneat_file={getattr(pyneat, '__file__', '<unknown>')}", flush=True)
    print(f"mode={args.mode}", flush=True)
    print(f"link_policy={args.link_policy}", flush=True)
    print(f"sender={args.sender}", flush=True)
    print(f"rtsp={args.rtsp_url}", flush=True)
    print(f"source_caps=NV12 {args.width}x{args.height}@{args.fps}", flush=True)
    for opt in video_options:
        print(
            f"destination channel={opt.channel} host={opt.host} port={opt.video_port}",
            flush=True,
        )
    print(f"sleep_seconds={args.seconds}", flush=True)
    if args.describe_backend:
        print("graph_backend:", flush=True)
        print(graph.describe_backend(), flush=True)

    run = graph.build(make_run_options())
    try:
        if probe_output is None:
            time.sleep(args.seconds)
        else:
            pulled = 0
            deadline = time.monotonic() + args.seconds
            while time.monotonic() < deadline:
                sample = run.pull(probe_output, 250)
                if sample is None:
                    continue
                pulled += 1
                print(
                    f"probe_pull count={pulled} frame_id={sample.frame_id}",
                    flush=True,
                )
            print(f"probe_pull_total={pulled}", flush=True)
        for method_name in ("stats", "input_stats", "report"):
            method = getattr(run, method_name, None)
            if callable(method):
                print(f"run.{method_name}():", flush=True)
                print(method(), flush=True)
    finally:
        run.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
