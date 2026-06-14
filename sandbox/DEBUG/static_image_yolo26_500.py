#!/usr/bin/env python3
"""Run repeated YOLO26 inference from one static image without saving outputs."""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pyneat


DEFAULT_IMAGE = "/workspace/sima-neat/apps/assets/images/image2.png"
DEFAULT_MODEL = (
    "/workspace/sima-neat/apps/assets/models/YOLO26-DETECTION/"
    "yolo26m-det-int8-b1.tar.gz"
)


def make_model(args: argparse.Namespace) -> pyneat.Model:
    opt = pyneat.ModelOptions()
    opt.preprocess.kind = pyneat.InputKind.Image
    opt.preprocess.enable = pyneat.AutoFlag.On
    opt.preprocess.color_convert.input_format = pyneat.PreprocessColorFormat.BGR
    opt.preprocess.preset = pyneat.NormalizePreset.COCO_YOLO
    opt.decode_type = pyneat.BoxDecodeType.YoloV26
    opt.score_threshold = args.score
    opt.nms_iou_threshold = args.nms_iou
    opt.top_k = args.top_k
    return pyneat.Model(str(args.model), opt)


def make_tensor(bgr: np.ndarray) -> pyneat.Tensor:
    return pyneat.Tensor.from_numpy(
        bgr,
        copy=True,
        image_format=pyneat.PixelFormat.BGR,
        memory=pyneat.TensorMemory.EV74,
    )


def make_run_options(args: argparse.Namespace) -> pyneat.RunOptions:
    opt = pyneat.RunOptions()
    opt.queue_depth = args.queue_depth
    opt.overflow_policy = pyneat.OverflowPolicy.Block
    opt.preset = pyneat.RunPreset.Balanced
    return opt


def print_progress(label: str, index: int, latency_ms: float, outputs: int | None) -> None:
    if index > 5 and index % 25 != 0:
        return
    suffix = "" if outputs is None else f" outputs={outputs}"
    print(f"{label} progress iteration={index} latency_ms={latency_ms:.3f}{suffix}", flush=True)


def print_result(label: str, latencies_ms: list[float], elapsed_s: float) -> None:
    p50 = statistics.median(latencies_ms)
    p95 = sorted(latencies_ms)[int(0.95 * (len(latencies_ms) - 1))]
    print(
        f"RESULT ok mode={label} iterations={len(latencies_ms)} "
        f"elapsed_s={elapsed_s:.3f} avg_ms={statistics.fmean(latencies_ms):.3f} "
        f"p50_ms={p50:.3f} p95_ms={p95:.3f} max_ms={max(latencies_ms):.3f}",
        flush=True,
    )


def run_model_runner(args: argparse.Namespace, bgr: np.ndarray) -> None:
    print("mode=runner topology=Model.build()->runner.run()", flush=True)
    model = make_model(args)
    seed = make_tensor(bgr)
    runner = model.build(
        [seed],
        route_options=pyneat.ModelRouteOptions(),
        run_options=make_run_options(args),
    )
    runner.run([seed], timeout_ms=args.timeout_ms)
    print("warmup_done=true", flush=True)

    latencies_ms: list[float] = []
    start = time.perf_counter()
    try:
        for index in range(1, args.iterations + 1):
            tensor = make_tensor(bgr)
            t0 = time.perf_counter()
            outputs = runner.run([tensor], timeout_ms=args.timeout_ms)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            if outputs is None:
                print(f"FAIL mode=runner iteration={index} reason=runner_returned_none")
                sys.exit(3)
            latencies_ms.append(latency_ms)
            print_progress("runner", index, latency_ms, len(outputs))
    finally:
        close = getattr(runner, "close", None)
        if callable(close):
            close()

    print_result("runner", latencies_ms, time.perf_counter() - start)


def make_graph_input_options(
    model: pyneat.Model, width: int, height: int
) -> pyneat.InputOptions:
    opt = model.input_appsrc_options(False)
    opt.payload_type = pyneat.PayloadType.Image
    opt.format = pyneat.Format.BGR
    opt.width = width
    opt.height = height
    opt.depth = 3
    return opt


def run_explicit_graph(args: argparse.Namespace, bgr: np.ndarray) -> None:
    print("mode=graph topology=Input(BGR)->model.graph()->Output", flush=True)
    model = make_model(args)
    height, width = bgr.shape[:2]
    graph = pyneat.Graph("static_image_yolo26_500")
    graph.add(pyneat.nodes.input(make_graph_input_options(model, width, height)))
    graph.add(model.graph())
    graph.add(pyneat.nodes.output())

    if args.describe_backend:
        print("graph_backend:", flush=True)
        print(graph.describe_backend(), flush=True)

    run = graph.build([make_tensor(bgr)], make_run_options(args))
    print("build_done=true", flush=True)

    latencies_ms: list[float] = []
    start = time.perf_counter()
    try:
        for index in range(1, args.iterations + 1):
            tensor = make_tensor(bgr)
            t0 = time.perf_counter()
            if not run.push([tensor]):
                print(f"FAIL mode=graph iteration={index} reason=push_returned_false")
                sys.exit(4)
            sample = run.pull(timeout_ms=args.timeout_ms)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            if sample is None:
                print(
                    f"FAIL mode=graph iteration={index} reason=pull_timeout "
                    f"latency_ms={latency_ms:.3f}",
                    flush=True,
                )
                sys.exit(5)
            latencies_ms.append(latency_ms)
            print_progress("graph", index, latency_ms, None)
    finally:
        run.close()

    print_result("graph", latencies_ms, time.perf_counter() - start)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=Path(DEFAULT_IMAGE))
    parser.add_argument("--model", type=Path, default=Path(DEFAULT_MODEL))
    parser.add_argument("--mode", choices=("graph", "runner", "both"), default="graph")
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--timeout-ms", type=int, default=20000)
    parser.add_argument("--queue-depth", type=int, default=8)
    parser.add_argument("--score", type=float, default=0.25)
    parser.add_argument("--nms-iou", type=float, default=0.45)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--no-describe-backend", dest="describe_backend", action="store_false")
    parser.set_defaults(describe_backend=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.iterations < 1:
        print("--iterations must be >= 1", file=sys.stderr)
        return 2
    if not args.image.is_file():
        print(f"image does not exist: {args.image}", file=sys.stderr)
        return 2
    if not args.model.is_file():
        print(f"model does not exist: {args.model}", file=sys.stderr)
        return 2

    bgr = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if bgr is None:
        print(f"cv2 failed to read image: {args.image}", file=sys.stderr)
        return 2
    bgr = np.ascontiguousarray(bgr, dtype=np.uint8)

    print(f"pyneat_version={getattr(pyneat, '__version__', '<unknown>')}", flush=True)
    print(f"pyneat_file={getattr(pyneat, '__file__', '<unknown>')}", flush=True)
    print(f"image={args.image}", flush=True)
    print(f"model={args.model}", flush=True)
    print(f"image_shape_hwc={bgr.shape} dtype={bgr.dtype}", flush=True)
    print(f"iterations={args.iterations}", flush=True)

    if args.mode in ("runner", "both"):
        run_model_runner(args, bgr)
    if args.mode in ("graph", "both"):
        run_explicit_graph(args, bgr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
