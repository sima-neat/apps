#!/usr/bin/env python3
"""Stress RtspDecodedInput without model, branches, or VideoSender."""

from __future__ import annotations

import argparse
import statistics
import sys
import time

import pyneat


DEFAULT_RTSP_URL = "rtsp://192.168.2.1:25689/src1"


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


def make_run_options(args: argparse.Namespace) -> pyneat.RunOptions:
    opt = pyneat.RunOptions()
    opt.preset = pyneat.RunPreset.Realtime
    opt.queue_depth = args.queue_depth
    opt.overflow_policy = pyneat.OverflowPolicy.KeepLatest
    opt.output_memory = pyneat.OutputMemory.ZeroCopy
    return opt


def make_graph(args: argparse.Namespace) -> pyneat.Graph:
    graph = pyneat.Graph("rtsp_decoded_input_stress")
    graph.add(pyneat.groups.rtsp_decoded_input(make_source_options(args)))
    graph.add(pyneat.nodes.output("frames", pyneat.OutputOptions.every_frame(1)))
    return graph


def print_runtime_state(run: object) -> None:
    for method_name in ("stats", "input_stats", "report"):
        method = getattr(run, method_name, None)
        if not callable(method):
            continue
        try:
            print(f"--- run.{method_name}()", flush=True)
            print(method(), flush=True)
        except Exception as exc:
            print(f"--- run.{method_name}() failed: {exc}", flush=True)


def run_once(args: argparse.Namespace, run_index: int) -> tuple[int, float]:
    graph = make_graph(args)
    print(f"run_index={run_index} rtsp={args.rtsp_url}", flush=True)
    if args.describe_backend:
        print("graph_backend:", flush=True)
        print(graph.describe_backend(), flush=True)

    run = graph.build(make_run_options(args))
    pulled = 0
    timeouts = 0
    latencies_ms: list[float] = []
    start = time.perf_counter()
    try:
        while pulled < args.frames:
            t0 = time.perf_counter()
            sample = run.pull("frames", args.timeout_ms)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            if sample is None:
                timeouts += 1
                print(
                    f"timeout run={run_index} pulled={pulled} "
                    f"timeouts={timeouts} latency_ms={latency_ms:.3f}",
                    flush=True,
                )
                if timeouts > args.max_timeouts:
                    print_runtime_state(run)
                    return pulled, time.perf_counter() - start
                continue

            pulled += 1
            latencies_ms.append(latency_ms)
            if pulled <= 5 or pulled % 25 == 0:
                print(
                    f"progress run={run_index} pulled={pulled} "
                    f"frame_id={sample.frame_id} latency_ms={latency_ms:.3f}",
                    flush=True,
                )
    finally:
        print_runtime_state(run)
        run.close()

    elapsed_s = time.perf_counter() - start
    p50 = statistics.median(latencies_ms)
    p95 = sorted(latencies_ms)[int(0.95 * (len(latencies_ms) - 1))]
    print(
        f"RESULT ok run={run_index} frames={pulled} elapsed_s={elapsed_s:.3f} "
        f"avg_pull_ms={statistics.fmean(latencies_ms):.3f} "
        f"p50_pull_ms={p50:.3f} p95_pull_ms={p95:.3f} "
        f"max_pull_ms={max(latencies_ms):.3f}",
        flush=True,
    )
    return pulled, elapsed_s


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rtsp-url", default=DEFAULT_RTSP_URL)
    parser.add_argument("--frames", type=int, default=500)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--latency-ms", type=int, default=100)
    parser.add_argument("--timeout-ms", type=int, default=20000)
    parser.add_argument("--max-timeouts", type=int, default=0)
    parser.add_argument("--queue-depth", type=int, default=8)
    parser.add_argument("--no-describe-backend", dest="describe_backend", action="store_false")
    parser.set_defaults(describe_backend=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.frames < 1 or args.runs < 1:
        print("--frames and --runs must be >= 1", file=sys.stderr)
        return 2

    print(f"pyneat_version={getattr(pyneat, '__version__', '<unknown>')}", flush=True)
    print(f"pyneat_file={getattr(pyneat, '__file__', '<unknown>')}", flush=True)
    print(
        f"target=RtspDecodedInput frames={args.frames} runs={args.runs} "
        f"caps=NV12 {args.width}x{args.height}@{args.fps}",
        flush=True,
    )

    total = 0
    total_elapsed = 0.0
    for run_index in range(1, args.runs + 1):
        pulled, elapsed_s = run_once(args, run_index)
        total += pulled
        total_elapsed += elapsed_s
        if pulled != args.frames:
            print(
                f"RESULT fail target=RtspDecodedInput run={run_index} "
                f"frames={pulled}/{args.frames}",
                flush=True,
            )
            return 1

    print(
        f"RESULT ok target=RtspDecodedInput total_frames={total} "
        f"runs={args.runs} total_elapsed_s={total_elapsed:.3f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
