#!/usr/bin/env python3
"""Stress the explicit H264 encoder chain without the VideoSender group."""

from __future__ import annotations

import argparse
import socket
import statistics
import sys
import threading
import time

import numpy as np
import pyneat


class UdpCounter:
    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self.packets = 0
        self.bytes = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._sock: socket.socket | None = None

    def start(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.settimeout(0.1)
        sock.bind((self.host, self.port))
        self._sock = sock
        self._thread = threading.Thread(target=self._run, name="udp-counter", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._sock is not None:
            self._sock.close()

    def _run(self) -> None:
        assert self._sock is not None
        while not self._stop.is_set():
            try:
                payload, _ = self._sock.recvfrom(65536)
            except TimeoutError:
                continue
            except OSError:
                break
            self.packets += 1
            self.bytes += len(payload)


def effective_channel(args: argparse.Namespace, run_index: int) -> int:
    return args.channel + run_index - 1 if args.increment_channel else args.channel


def effective_port(args: argparse.Namespace, run_index: int) -> int:
    return args.video_port_base + effective_channel(args, run_index)


def make_input_options(args: argparse.Namespace) -> pyneat.InputOptions:
    opt = pyneat.InputOptions()
    opt.payload_type = pyneat.PayloadType.Image
    opt.format = pyneat.Format.RGB
    opt.width = args.width
    opt.height = args.height
    opt.depth = 3
    opt.fps_n = args.fps
    opt.fps_d = 1
    opt.use_simaai_pool = False
    return opt


def make_run_options(args: argparse.Namespace) -> pyneat.RunOptions:
    opt = pyneat.RunOptions()
    opt.preset = pyneat.RunPreset.Realtime
    opt.queue_depth = args.queue_depth
    opt.overflow_policy = pyneat.OverflowPolicy.Block
    return opt


def make_rgb_frame(args: argparse.Namespace, index: int) -> np.ndarray:
    frame = np.zeros((args.height, args.width, 3), dtype=np.uint8)
    frame[:, :, 0] = 32
    frame[:, :, 1] = (index * 3) % 255
    frame[:, :, 2] = 180
    x0 = (index * 17) % max(1, args.width - 64)
    frame[:, x0 : x0 + 64, :] = (240, 240, 32)
    return np.ascontiguousarray(frame)


def make_tensor(args: argparse.Namespace, index: int) -> pyneat.Tensor:
    return pyneat.Tensor.from_numpy(
        make_rgb_frame(args, index),
        copy=True,
        image_format=pyneat.PixelFormat.RGB,
        memory=pyneat.TensorMemory.CPU,
    )


def add_encoder_frontend(graph: pyneat.Graph, args: argparse.Namespace) -> None:
    graph.add(pyneat.nodes.input(make_input_options(args)))
    graph.add(pyneat.nodes.caps_raw("", width=args.width, height=args.height, fps=args.fps))
    graph.add(pyneat.nodes.video_convert())
    graph.add(pyneat.nodes.caps_raw("NV12", width=args.width, height=args.height, fps=args.fps))
    graph.add(
        pyneat.nodes.h264_encode_sima(
            args.width,
            args.height,
            args.fps,
            bitrate_kbps=args.bitrate_kbps,
        )
    )


def make_udp_output(args: argparse.Namespace, run_index: int):
    udp_options = pyneat.UdpOutputOptions()
    udp_options.host = args.host
    udp_options.port = effective_port(args, run_index)
    return pyneat.nodes.udp_output(udp_options)


def make_graph(args: argparse.Namespace, run_index: int) -> pyneat.Graph:
    graph = pyneat.Graph(f"encoder_input_stress_{args.mode}")
    add_encoder_frontend(graph, args)

    if args.mode == "encode-output":
        graph.add(pyneat.nodes.output("encoded", pyneat.OutputOptions.every_frame(1)))
        return graph

    graph.add(pyneat.nodes.h264_parse(args.config_interval))
    if args.mode == "parse-output":
        graph.add(pyneat.nodes.output("parsed", pyneat.OutputOptions.every_frame(1)))
        return graph

    graph.add(
        pyneat.nodes.h264_packetize(
            payload_type=args.payload_type,
            config_interval=args.config_interval,
        )
    )
    graph.add(make_udp_output(args, run_index))
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


def run_output_mode(args: argparse.Namespace, run, run_index: int) -> tuple[int, int]:
    output_name = "encoded" if args.mode == "encode-output" else "parsed"
    latencies_ms: list[float] = []
    for index in range(1, args.frames + 1):
        tensor = make_tensor(args, index)
        t0 = time.perf_counter()
        if not run.push([tensor]):
            print(f"RESULT fail mode={args.mode} run={run_index} frame={index} reason=push_false")
            return 1, index - 1
        try:
            sample = run.pull(output_name, args.timeout_ms)
        except Exception as exc:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            print(
                f"RESULT fail mode={args.mode} run={run_index} frame={index} "
                f"reason=pull_exception latency_ms={latency_ms:.3f} error={exc}",
                flush=True,
            )
            return 3, index - 1
        latency_ms = (time.perf_counter() - t0) * 1000.0
        if sample is None:
            print(
                f"RESULT fail mode={args.mode} run={run_index} frame={index} "
                f"reason=pull_timeout latency_ms={latency_ms:.3f}",
                flush=True,
            )
            return 2, index - 1
        latencies_ms.append(latency_ms)
        if index <= 5 or index % 25 == 0:
            print(
                f"progress mode={args.mode} run={run_index} frame={index} "
                f"latency_ms={latency_ms:.3f} sample_frame_id={sample.frame_id}",
                flush=True,
            )

    p50 = statistics.median(latencies_ms)
    p95 = sorted(latencies_ms)[int(0.95 * (len(latencies_ms) - 1))]
    print(
        f"RESULT ok mode={args.mode} run={run_index} frames={args.frames} "
        f"avg_ms={statistics.fmean(latencies_ms):.3f} p50_ms={p50:.3f} "
        f"p95_ms={p95:.3f} max_ms={max(latencies_ms):.3f}",
        flush=True,
    )
    return 0, args.frames


def run_udp_mode(
    args: argparse.Namespace, run, receiver: UdpCounter | None, run_index: int
) -> tuple[int, int, int]:
    push_latencies_ms: list[float] = []
    start = time.perf_counter()
    for index in range(1, args.frames + 1):
        frame_start = time.perf_counter()
        tensor = make_tensor(args, index)
        t0 = time.perf_counter()
        if not run.push([tensor]):
            print(f"RESULT fail mode=rtp-udp run={run_index} frame={index} reason=push_false")
            packets = receiver.packets if receiver is not None else -1
            return 1, index - 1, packets
        latency_ms = (time.perf_counter() - t0) * 1000.0
        push_latencies_ms.append(latency_ms)
        if index <= 5 or index % 25 == 0:
            packets = receiver.packets if receiver is not None else -1
            print(
                f"progress mode=rtp-udp run={run_index} frame={index} "
                f"push_ms={latency_ms:.3f} udp_packets={packets}",
                flush=True,
            )

        if args.pace:
            target = start + index / args.fps
            sleep_s = target - time.perf_counter()
            if sleep_s > 0:
                time.sleep(sleep_s)
        elif args.push_interval_ms > 0:
            elapsed_s = time.perf_counter() - frame_start
            sleep_s = args.push_interval_ms / 1000.0 - elapsed_s
            if sleep_s > 0:
                time.sleep(sleep_s)

    run.close_input()
    time.sleep(args.drain_seconds)

    p50 = statistics.median(push_latencies_ms)
    p95 = sorted(push_latencies_ms)[int(0.95 * (len(push_latencies_ms) - 1))]
    packets = receiver.packets if receiver is not None else -1
    packet_bytes = receiver.bytes if receiver is not None else -1
    print(
        f"RESULT ok mode=rtp-udp run={run_index} frames={args.frames} "
        f"avg_push_ms={statistics.fmean(push_latencies_ms):.3f} "
        f"p50_push_ms={p50:.3f} p95_push_ms={p95:.3f} "
        f"max_push_ms={max(push_latencies_ms):.3f} "
        f"udp_packets={packets} udp_bytes={packet_bytes}",
        flush=True,
    )
    if args.receiver and packets <= 0:
        print("RESULT fail mode=rtp-udp reason=no_udp_packets", flush=True)
        return 2, args.frames, packets
    return 0, args.frames, packets


def run_once(args: argparse.Namespace, run_index: int) -> tuple[int, int, int]:
    port = effective_port(args, run_index)
    receiver = None
    if args.mode == "rtp-udp" and args.receiver:
        receiver = UdpCounter(args.receiver_host, port)
        receiver.start()

    graph = make_graph(args, run_index)
    print(
        f"run_index={run_index} mode={args.mode} destination={args.host}:{port} "
        f"receiver={receiver is not None}",
        flush=True,
    )
    if args.describe_backend:
        print("graph_backend:", flush=True)
        print(graph.describe_backend(), flush=True)

    run = graph.build([make_tensor(args, 0)], make_run_options(args))
    try:
        if args.mode in ("encode-output", "parse-output"):
            rc, frames = run_output_mode(args, run, run_index)
            return rc, frames, -1
        return run_udp_mode(args, run, receiver, run_index)
    finally:
        print_runtime_state(run)
        run.close()
        if receiver is not None:
            receiver.stop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("encode-output", "parse-output", "rtp-udp"),
        default="encode-output",
    )
    parser.add_argument("--frames", type=int, default=500)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--bitrate-kbps", type=int, default=1000)
    parser.add_argument("--payload-type", type=int, default=96)
    parser.add_argument("--config-interval", type=int, default=1)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--receiver-host", default="127.0.0.1")
    parser.add_argument("--video-port-base", type=int, default=9000)
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--increment-channel", action="store_true")
    parser.add_argument("--queue-depth", type=int, default=8)
    parser.add_argument("--timeout-ms", type=int, default=20000)
    parser.add_argument("--drain-seconds", type=float, default=1.0)
    parser.add_argument("--push-interval-ms", type=float, default=0.0)
    parser.add_argument("--no-pace", dest="pace", action="store_false")
    parser.add_argument("--no-receiver", dest="receiver", action="store_false")
    parser.add_argument("--no-describe-backend", dest="describe_backend", action="store_false")
    parser.set_defaults(pace=True, receiver=True, describe_backend=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.frames < 1 or args.runs < 1:
        print("--frames and --runs must be >= 1", file=sys.stderr)
        return 2
    if args.fps < 1:
        print("--fps must be >= 1", file=sys.stderr)
        return 2

    print(f"pyneat_version={getattr(pyneat, '__version__', '<unknown>')}", flush=True)
    print(f"pyneat_file={getattr(pyneat, '__file__', '<unknown>')}", flush=True)
    print(
        f"target=explicit-encoder mode={args.mode} frames={args.frames} runs={args.runs} "
        f"caps=RGB {args.width}x{args.height}@{args.fps}",
        flush=True,
    )

    total_frames = 0
    total_packets = 0
    for run_index in range(1, args.runs + 1):
        rc, frames, packets = run_once(args, run_index)
        total_frames += frames
        if packets > 0:
            total_packets += packets
        if rc != 0:
            return rc

    print(
        f"RESULT ok target=explicit-encoder mode={args.mode} "
        f"total_frames={total_frames} runs={args.runs} "
        f"total_udp_packets={total_packets}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
