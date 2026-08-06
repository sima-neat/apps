"""Run SuperPoint on a video and stream the feature-point overlay to Insight."""

from __future__ import annotations

import argparse
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
    io = raw.get("io", {})
    insight = raw.get("output", {}).get("insight", {})
    runtime = raw.get("runtime", {})
    config = argparse.Namespace(
        model=Path(model.get("path", "models/superpoint_mpk.tar.gz")),
        input=Path(io.get("input", "assets/datasets/tum-rgbd/freiburg1-desk.mp4")),
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


def model_frame(model):
    shape = tuple(int(value) for value in model.preprocess_requirements().output_shape)
    if len(shape) < 3 or shape[-3] <= 0 or shape[-2] <= 0:
        raise RuntimeError("SuperPoint Preproc must expose a positive HWC output shape")
    return shape[-2], shape[-3]


def remap_points(points, model_width, model_height, output_width, output_height, np):
    remapped = np.array(points, dtype=np.float32, copy=True)
    if remapped.size:
        remapped[:, 0] *= np.float32(output_width / model_width)
        remapped[:, 1] *= np.float32(output_height / model_height)
    return remapped


def feature_points(
    output, model_width, model_height, output_width, output_height, np, pyneat
):
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
    if points.size and (
        not np.all(np.isfinite(points))
        or np.any(points[:, 0] < 0)
        or np.any(points[:, 0] >= model_width)
        or np.any(points[:, 1] < 0)
        or np.any(points[:, 1] >= model_height)
    ):
        raise RuntimeError("SuperPoint returned an invalid keypoint coordinate")
    return remap_points(
        points, model_width, model_height, output_width, output_height, np
    )


def input_tensor(frame, np, pyneat):
    return pyneat.Tensor.from_numpy(
        np.ascontiguousarray(frame, dtype=np.uint8),
        copy=True,
        image_format=pyneat.PixelFormat.BGR,
        memory=pyneat.TensorMemory.EV74,
    )


def draw_points(frame, points, cv2) -> None:
    for x, y in points:
        cv2.circle(
            frame,
            (int(round(float(x))), int(round(float(y)))),
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
    output_fps = max(1, int(round(fps)))
    input_options = pyneat.InputOptions()
    input_options.payload_type = pyneat.PayloadType.Image
    input_options.format = pyneat.Format.RGB
    input_options.width = width
    input_options.height = height
    input_options.depth = 3
    input_options.fps_n = output_fps
    input_options.fps_d = 1
    input_options.memory_policy = pyneat.InputMemoryPolicy.Ev74

    sender_options = pyneat.VideoSenderOptions.h264_rtp_udp_from_raw(
        width, height, output_fps
    )
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
        image_format=pyneat.PixelFormat.RGB,
        memory=pyneat.TensorMemory.EV74,
    )
    return graph, graph.build([seed]), sender_options.video_port


def stream_frame(run, frame, cv2, pyneat) -> None:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = pyneat.Tensor.from_numpy(
        rgb,
        copy=True,
        image_format=pyneat.PixelFormat.RGB,
        memory=pyneat.TensorMemory.EV74,
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
    if not config.input.is_file():
        print(f"Error: input video does not exist: {config.input}", file=sys.stderr)
        return 2

    import cv2
    import numpy as np
    import pyneat

    video = cv2.VideoCapture(str(config.input))
    ok, frame = video.read()
    if not video.isOpened() or not ok:
        print(f"Error: failed to read input video: {config.input}", file=sys.stderr)
        return 2

    fps = video.get(cv2.CAP_PROP_FPS)
    fps = fps if np.isfinite(fps) and fps > 0 else 30.0

    runner = None
    video_run = None
    try:
        validate_frame(frame, np)
        input_shape = frame.shape
        input_height, input_width = input_shape[:2]
        model = pyneat.Model(
            str(config.model), model_options(pyneat, input_width, input_height)
        )
        model_width, model_height = model_frame(model)
        model_input = input_tensor(frame, np, pyneat)
        runner = model.build(
            [model_input],
            route_options=pyneat.ModelRouteOptions(),
            run_options=pyneat.RunOptions(),
        )
        _video_graph, video_run, video_port = build_video_sender(
            config, fps, input_width, input_height, np, pyneat
        )

        processed = 0
        total_points = 0
        while True:
            output = runner.run([model_input], timeout_ms=config.timeout_ms)
            points = feature_points(
                output,
                model_width,
                model_height,
                frame.shape[1],
                frame.shape[0],
                np,
                pyneat,
            )
            total_points += len(points)
            draw_points(frame, points, cv2)
            stream_frame(video_run, frame, cv2, pyneat)
            processed += 1

            if config.frames > 0 and processed >= config.frames:
                break
            ok, frame = video.read()
            if not ok:
                break
            validate_frame(frame, np, input_shape)
            model_input = input_tensor(frame, np, pyneat)

        average = total_points / processed
        print(
            f"frames={processed} average_points={average:.1f} "
            f"descriptor_dim={DESCRIPTOR_DIM} "
            f"video_sender={config.insight_host}:{video_port}"
        )
        return 0
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    finally:
        if runner is not None:
            runner.close()
        if video_run is not None:
            video_run.close()
        video.release()


if __name__ == "__main__":
    raise SystemExit(main())
