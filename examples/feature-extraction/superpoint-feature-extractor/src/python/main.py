"""Run SuperPoint on an RTSP stream and send the feature-point overlay to Insight."""

from __future__ import annotations

import argparse
import os
import signal
import sys
from pathlib import Path

import yaml

DESCRIPTOR_DIM = 256
MAX_POINTS = 600


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SuperPoint video feature extractor")
    default_config = Path(__file__).resolve().parents[1] / "common" / "config.yaml"
    parser.add_argument(
        "--config", type=Path, default=default_config, help="Path to YAML configuration"
    )
    return parser.parse_args()


def load_config(path: Path) -> argparse.Namespace:
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    model = raw.get("model", {})
    source = raw.get("source", {})
    insight = raw.get("output", {}).get("insight", {})
    runtime = raw.get("runtime", {})
    config = argparse.Namespace(
        model=Path(model.get("path", "models/superpoint_mpk.tar.gz")),
        rtsp_url=str(source.get("rtsp_url", "")),
        tcp=bool(source.get("tcp", True)),
        latency_ms=int(source.get("latency_ms", 100)),
        insight_host=str(insight.get("host", "127.0.0.1")),
        video_port=int(insight.get("video_port", 9000)),
        channel=int(insight.get("channel", 0)),
        bitrate_kbps=int(insight.get("bitrate_kbps", 1000)),
        frames=int(runtime.get("frames", 0)),
        timeout_ms=int(runtime.get("timeout_ms", 20000)),
    )
    if config.frames < 0:
        raise ValueError("runtime.frames must be >= 0")
    if config.timeout_ms <= 0:
        raise ValueError("runtime.timeout_ms must be > 0")
    if not config.rtsp_url or config.latency_ms < 0:
        raise ValueError("set source.rtsp_url and a non-negative source.latency_ms")
    if not config.insight_host:
        raise ValueError("output.insight.host must be set")
    if not 0 < config.video_port <= 65535:
        raise ValueError("output.insight.video_port must be in [1, 65535]")
    if config.channel < 0 or config.video_port + config.channel > 65535:
        raise ValueError("output.insight channel selects an invalid UDP port")
    if config.bitrate_kbps <= 0:
        raise ValueError("output.insight.bitrate_kbps must be > 0")
    return config


def model_options(pyneat, input_width, input_height):
    options = pyneat.ModelOptions()
    options.preprocess.kind = pyneat.InputKind.Image
    options.preprocess.enable = pyneat.AutoFlag.On
    options.preprocess.input_max_width = input_width
    options.preprocess.input_max_height = input_height
    options.preprocess.input_max_depth = 3
    options.preprocess.resize.enable = pyneat.AutoFlag.On
    options.preprocess.resize.mode = pyneat.ResizeMode.Stretch
    options.preprocess.color_convert.enable = pyneat.AutoFlag.On
    options.preprocess.color_convert.input_format = pyneat.PreprocessColorFormat.BGR
    options.preprocess.color_convert.output_format = pyneat.PreprocessColorFormat.GRAY8
    options.preprocess.normalize.enable = pyneat.AutoFlag.On
    options.preprocess.normalize.mean = (0.0, 0.0, 0.0)
    options.preprocess.normalize.stddev = (1.0, 1.0, 1.0)
    options.preprocess.normalize.has_explicit_stats = True
    options.decode_type = pyneat.BoxDecodeType.SuperPoint
    options.superpoint.profile = pyneat.SuperPointProfile.A65V1
    options.superpoint.output_format = pyneat.SuperPointOutputFormat.FeaturePointsV1
    options.superpoint.descriptor_output_dtype = pyneat.TensorDType.Float32
    options.processcvu.post_run_target = "A65"
    return options


def feature_points(output, source_width, source_height, np, pyneat):
    decoded = pyneat.decode_superpoint(list(output))
    if len(decoded) != 1:
        raise RuntimeError("SuperPoint must return one feature set per frame")

    features = decoded[0]
    points = np.asarray(features.keypoints.to_numpy(copy=True), dtype=np.float32)
    scores_shape = tuple(int(value) for value in features.scores.shape)
    descriptor_shape = tuple(int(value) for value in features.descriptors.shape)
    if (
        features.keypoints.dtype != pyneat.TensorDType.Float32
        or points.ndim != 2
        or points.shape[1:] != (2,)
        or points.shape[0] > MAX_POINTS
        or scores_shape != (points.shape[0],)
        or descriptor_shape != (points.shape[0], DESCRIPTOR_DIM)
        or features.scores.dtype != pyneat.TensorDType.Float32
        or features.descriptors.dtype != pyneat.TensorDType.Float32
    ):
        raise RuntimeError("invalid SuperPoint output contract")
    if not points.size:
        return points

    # The frame edges are exclusive coordinates. Discard isolated decoder outliers
    # instead of aborting the stream when inverse-affine rounding places a point on
    # or beyond an edge.
    drawable = (
        np.all(np.isfinite(points), axis=1)
        & (points[:, 0] >= 0)
        & (points[:, 0] < source_width)
        & (points[:, 1] >= 0)
        & (points[:, 1] < source_height)
    )
    return points[drawable]


def input_tensor(frame, np, pyneat):
    return pyneat.Tensor.from_numpy(
        np.ascontiguousarray(frame, dtype=np.uint8),
        copy=True,
        image_format=pyneat.PixelFormat.BGR,
        memory=pyneat.TensorMemory.EV74,
    )


def probe_stream(config, cv2):
    key = "OPENCV_FFMPEG_CAPTURE_OPTIONS"
    previous = os.environ.get(key)
    os.environ[key] = f"rtsp_transport;{'tcp' if config.tcp else 'udp'}"
    try:
        video = cv2.VideoCapture(config.rtsp_url)
    finally:
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = round(video.get(cv2.CAP_PROP_FPS))
    video.release()
    if width <= 0 or height <= 0 or fps <= 0:
        raise RuntimeError(f"could not read the frame size and rate of {config.rtsp_url}")
    return width, height, fps


def build_source(config, width, height, fps, pyneat):
    options = pyneat.RtspDecodedInputOptions()
    options.url = config.rtsp_url
    options.tcp = config.tcp
    options.latency_ms = config.latency_ms
    options.fallback_h264_width = width
    options.fallback_h264_height = height
    options.fallback_h264_fps = fps
    output = pyneat.OutputOptions.latest()
    output.max_buffers = 4
    graph = pyneat.Graph("rtsp")
    graph.connect(pyneat.groups.rtsp_decoded_input(options), pyneat.nodes.output("frame", output))
    run_options = pyneat.RunOptions()
    run_options.preset = pyneat.RunPreset.Realtime
    run_options.queue_depth = 4
    run_options.overflow_policy = pyneat.OverflowPolicy.KeepLatest
    run_options.output_memory = pyneat.OutputMemory.ZeroCopy
    return graph, graph.build(run_options)


def pull_frame(run, timeout_ms, np, cv2, pyneat):
    sample = run.pull("frame", timeout_ms)
    if sample is None:
        raise RuntimeError("timed out waiting for an RTSP frame")
    field = sample
    if sample.fields:
        field = next(part for part in sample.fields if part.stream_label == "frame")
    tensor = field.tensor if field.kind == pyneat.SampleKind.Tensor else field.tensors[0]
    width, height = tensor.width(), tensor.height()
    payload = tensor.copy_payload_bytes()
    planes = {plane.role: plane for plane in tensor.planes}
    packed = np.empty((height * 3 // 2, width), np.uint8)
    for role, rows, row0 in ((pyneat.PlaneRole.Y, height, 0),
                            (pyneat.PlaneRole.UV, height // 2, height)):
        plane = planes[role]
        stride = plane.strides_bytes[0] if plane.strides_bytes else width
        packed[row0:row0 + rows] = np.ndarray((rows, width), np.uint8, buffer=payload,
                                            offset=plane.byte_offset, strides=(stride, 1))
    # Return owned pixels so the decoder loan is released before model execution.
    return cv2.cvtColor(packed, cv2.COLOR_YUV2BGR_NV12)


def draw_points(frame, points, cv2) -> None:
    for x, y in points:
        cv2.circle(
            frame,
            (round(float(x)), round(float(y))),
            2,
            (0, 255, 0),
            -1,
            cv2.LINE_AA,
        )
    cv2.putText(
        frame,
        f"points: {len(points)}",
        (12, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )


def build_video_sender(config, fps, width, height, np, pyneat):
    output_fps = max(1, round(fps))
    input_options = pyneat.InputOptions()
    input_options.payload_type = pyneat.PayloadType.Image
    input_options.format = pyneat.Format.BGR
    input_options.width = width
    input_options.height = height
    input_options.depth = 3
    input_options.fps_n = output_fps
    input_options.fps_d = 1

    encode = pyneat.SimaEncodeOptions()
    encode.fps = output_fps
    sender_options = pyneat.VideoSenderOptions.from_raw(encode)
    sender_options.host = config.insight_host
    sender_options.channel = config.channel
    sender_options.video_port_base = config.video_port
    sender_options.encoder.bitrate_kbps = config.bitrate_kbps

    graph = pyneat.Graph("insight")
    graph.add(pyneat.nodes.input(input_options))
    graph.add(pyneat.groups.video_sender(sender_options))
    seed = pyneat.Tensor.from_numpy(
        np.zeros((height, width, 3), dtype=np.uint8),
        copy=True,
        image_format=pyneat.PixelFormat.BGR,
        memory=pyneat.TensorMemory.CPU,
    )
    return graph, graph.build([seed]), sender_options.video_port


def stream_frame(run, frame, pyneat) -> None:
    tensor = pyneat.Tensor.from_numpy(
        frame,
        copy=True,
        image_format=pyneat.PixelFormat.BGR,
        memory=pyneat.TensorMemory.CPU,
    )
    if not run.push([tensor]):
        raise RuntimeError("Insight video push failed")


def validate_frame(frame, np, expected_shape=None) -> None:
    if (
        frame is None
        or frame.ndim != 3
        or frame.shape[0] <= 0
        or frame.shape[1] <= 0
        or frame.shape[2] != 3
        or frame.dtype != np.uint8
    ):
        raise RuntimeError("SuperPoint input must be a non-empty BGR video frame")
    if expected_shape is not None and frame.shape != expected_shape:
        raise RuntimeError(
            "SuperPoint input resolution changed after pipeline construction"
        )


def main() -> int:
    args = parse_args()
    try:
        config = load_config(args.config)
    except (OSError, TypeError, ValueError, yaml.YAMLError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2

    if not config.model.is_file():
        print(f"Error: model does not exist: {config.model}", file=sys.stderr)
        return 2

    import cv2
    import numpy as np
    import pyneat

    runner = None
    video_run = None
    source_run = None
    stop_requested = False

    def request_stop(_signum, _frame):
        nonlocal stop_requested
        stop_requested = True

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, request_stop)
    try:
        input_width, input_height, fps = probe_stream(config, cv2)
        _source_graph, source_run = build_source(config, input_width, input_height, fps, pyneat)
        frame = pull_frame(source_run, config.timeout_ms, np, cv2, pyneat)
        validate_frame(frame, np, (input_height, input_width, 3))
        input_shape = frame.shape
        model = pyneat.Model(
            str(config.model), model_options(pyneat, input_width, input_height)
        )
        model_input = input_tensor(frame, np, pyneat)
        runner = model.build([model_input])
        _video_graph, video_run, video_port = build_video_sender(
            config, fps, input_width, input_height, np, pyneat
        )

        processed = 0
        total_points = 0
        while not stop_requested and (config.frames <= 0 or processed < config.frames):
            output = runner.run([model_input], timeout_ms=config.timeout_ms)
            points = feature_points(
                output,
                input_width,
                input_height,
                np,
                pyneat,
            )
            total_points += len(points)
            draw_points(frame, points, cv2)
            stream_frame(video_run, frame, pyneat)
            processed += 1

            if stop_requested or (config.frames > 0 and processed >= config.frames):
                break
            frame = pull_frame(source_run, config.timeout_ms, np, cv2, pyneat)
            validate_frame(frame, np, input_shape)
            model_input = input_tensor(frame, np, pyneat)

        average = total_points / processed if processed else 0.0
        print(
            f"frames={processed} average_points={average:.1f} "
            f"descriptor_dim={DESCRIPTOR_DIM} "
            f"video_sender={config.insight_host}:{video_port}"
        )
        return 0
    except Exception as error:  # noqa: BLE001
        print(f"Error: {error}", file=sys.stderr)
        return 2
    finally:
        if runner is not None:
            runner.close()
        if video_run is not None:
            video_run.close()
        if source_run is not None:
            source_run.close()


if __name__ == "__main__":
    raise SystemExit(main())
