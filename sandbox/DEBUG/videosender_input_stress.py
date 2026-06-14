#!/usr/bin/env python3
"""Stress VideoSender from generated raw frames without RTSP or model logic."""

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


def make_input_options(args: argparse.Namespace) -> pyneat.InputOptions:
    opt = pyneat.InputOptions()
    opt.payload_type = pyneat.PayloadType.Image
    opt.format = pyneat.Format.NV12 if args.format == "NV12" else pyneat.Format.RGB
    opt.width = args.width
    opt.height = args.height
    opt.fps_n = args.fps
    opt.fps_d = 1
    opt.use_simaai_pool = False
    if args.format == "RGB":
        opt.depth = 3
    return opt


def effective_channel(args: argparse.Namespace, run_index: int) -> int:
    return args.channel + run_index - 1 if args.increment_channel else args.channel


def make_video_options(args: argparse.Namespace, run_index: int) -> pyneat.VideoSenderOptions:
    opt = pyneat.VideoSenderOptions.h264_rtp_udp_from_raw(
        args.width, args.height, args.fps
    )
    opt.host = args.host
    opt.channel = effective_channel(args, run_index)
    opt.video_port_base = args.video_port_base
    opt.encoder.bitrate_kbps = args.bitrate_kbps
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


def make_nv12_frame(args: argparse.Namespace, index: int) -> np.ndarray:
    frame = np.empty((args.height + args.height // 2, args.width), dtype=np.uint8)
    frame[: args.height, :] = 64
    frame[args.height :, :] = 128
    x0 = (index * 17) % max(1, args.width - 64)
    frame[: args.height, x0 : x0 + 64] = 200
    return np.ascontiguousarray(frame)


def make_tensor(args: argparse.Namespace, index: int) -> pyneat.Tensor:
    if args.format == "NV12":
        return pyneat.Tensor.from_numpy(
            make_nv12_frame(args, index),
            copy=True,
            image_format=pyneat.PixelFormat.NV12,
            memory=pyneat.TensorMemory.CPU,
        )
    return pyneat.Tensor.from_numpy(
        make_rgb_frame(args, index),
        copy=True,
        image_format=pyneat.PixelFormat.RGB,
        memory=pyneat.TensorMemory.CPU,
    )


def make_graph(args: argparse.Namespace, run_index: int) -> pyneat.Graph:
    graph = pyneat.Graph("videosender_input_stress")
    graph.add(pyneat.nodes.input(make_input_options(args)))
    graph.add(pyneat.groups.video_sender(make_video_options(args, run_index)))
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


def run_once(args: argparse.Namespace, run_index: int) -> tuple[int, int, int]:
    video_options = make_video_options(args, run_index)
    receiver = UdpCounter(args.receiver_host, video_options.video_port) if args.receiver else None
    if receiver is not None:
        receiver.start()

    graph = make_graph(args, run_index)
    print(
        f"run_index={run_index} destination host={video_options.host} port={video_options.video_port} "
        f"receiver={args.receiver}",
        flush=True,
    )
    if args.describe_backend:
        print("graph_backend:", flush=True)
        print(graph.describe_backend(), flush=True)

    seed = make_tensor(args, 0)
    run = graph.build([seed], make_run_options(args))
    push_latencies_ms: list[float] = []
    start = time.perf_counter()
    try:
        for index in range(1, args.frames + 1):
            frame_start = time.perf_counter()
            tensor = make_tensor(args, index)
            t0 = time.perf_counter()
            if not run.push([tensor]):
                print(f"RESULT fail target=VideoSender frame={index} reason=push_false")
                return 1, index - 1, receiver.packets if receiver is not None else -1
            latency_ms = (time.perf_counter() - t0) * 1000.0
            push_latencies_ms.append(latency_ms)
            if index <= 5 or index % 25 == 0:
                packets = receiver.packets if receiver is not None else -1
                print(
                    f"progress run={run_index} frame={index} push_ms={latency_ms:.3f} "
                    f"udp_packets={packets}",
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
    finally:
        print_runtime_state(run)
        run.close()
        if receiver is not None:
            receiver.stop()

    elapsed_s = time.perf_counter() - start
    p50 = statistics.median(push_latencies_ms)
    p95 = sorted(push_latencies_ms)[int(0.95 * (len(push_latencies_ms) - 1))]
    packets = receiver.packets if receiver is not None else -1
    packet_bytes = receiver.bytes if receiver is not None else -1
    print(
        f"RESULT ok target=VideoSender run={run_index} frames={args.frames} format={args.format} "
        f"elapsed_s={elapsed_s:.3f} avg_push_ms={statistics.fmean(push_latencies_ms):.3f} "
        f"p50_push_ms={p50:.3f} p95_push_ms={p95:.3f} "
        f"max_push_ms={max(push_latencies_ms):.3f} "
        f"udp_packets={packets} udp_bytes={packet_bytes}",
        flush=True,
    )
    if args.receiver and packets <= 0:
        print("RESULT fail target=VideoSender reason=no_udp_packets", flush=True)
        return 2, args.frames, packets
    return 0, args.frames, packets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=500)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--format", choices=("RGB", "NV12"), default="RGB")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--receiver-host", default="127.0.0.1")
    parser.add_argument("--video-port-base", type=int, default=9000)
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--increment-channel", action="store_true")
    parser.add_argument("--bitrate-kbps", type=int, default=1000)
    parser.add_argument("--queue-depth", type=int, default=8)
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
        f"target=VideoSender frames={args.frames} format={args.format} "
        f"runs={args.runs} caps={args.width}x{args.height}@{args.fps} pace={args.pace}",
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
        f"RESULT ok target=VideoSender total_frames={total_frames} "
        f"runs={args.runs} total_udp_packets={total_packets}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
