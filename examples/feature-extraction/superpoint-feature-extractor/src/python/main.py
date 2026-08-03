"""Run SuperPoint on a video and draw the extracted feature points."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


WIDTH = 640
HEIGHT = 480
DESCRIPTOR_DIM = 256
MAX_POINTS = 600


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SuperPoint video feature extractor")
    default_config = Path(__file__).resolve().parents[1] / "common" / "config.yaml"
    parser.add_argument("--config", type=Path, default=default_config, help="Path to YAML configuration")
    return parser.parse_args()


def load_config(path: Path) -> argparse.Namespace:
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    model = raw.get("model", {})
    io = raw.get("io", {})
    runtime = raw.get("runtime", {})
    config = argparse.Namespace(
        model=Path(model.get("path", "models/superpoint_mpk.tar.gz")),
        input=Path(io.get("input", "assets/datasets/tum-rgbd/freiburg1-desk.mp4")),
        output=Path(io.get("output", "sandbox/superpoint-feature-extractor.mp4")),
        frames=int(runtime.get("frames", 0)),
        timeout_ms=int(runtime.get("timeout_ms", 20000)),
    )
    if config.frames < 0:
        raise ValueError("runtime.frames must be >= 0")
    if config.timeout_ms <= 0:
        raise ValueError("runtime.timeout_ms must be > 0")
    return config


def model_options(pyneat):
    options = pyneat.ModelOptions()
    options.preprocess.enable = pyneat.AutoFlag.Off
    options.decode_type = pyneat.BoxDecodeType.SuperPoint
    options.superpoint.profile = pyneat.SuperPointProfile.A65V1
    options.superpoint.output_format = pyneat.SuperPointOutputFormat.FeaturePointsV1
    options.superpoint.descriptor_output_dtype = pyneat.TensorDType.Float32
    options.boxdecode_original_width = WIDTH
    options.boxdecode_original_height = HEIGHT
    options.boxdecode_resize_mode = pyneat.ResizeMode.Stretch
    options.processcvu.post_run_target = "A65"
    return options


def feature_points(output, np, pyneat):
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
        or np.any(points[:, 0] >= WIDTH)
        or np.any(points[:, 1] < 0)
        or np.any(points[:, 1] >= HEIGHT)
    ):
        raise RuntimeError("SuperPoint returned an invalid keypoint coordinate")
    return points


def input_tensor(frame, dtype, cv2, np, pyneat):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    values = np.ascontiguousarray(gray[:, :, None], dtype=np.float32)
    values *= np.float32(1.0 / 255.0)
    if dtype == pyneat.TensorDType.Float32:
        return pyneat.Tensor.from_numpy(
            values,
            copy=True,
            layout=pyneat.TensorLayout.HWC,
            memory=pyneat.TensorMemory.EV74,
        )
    if dtype == pyneat.TensorDType.BFloat16:
        bits = values.view(np.uint32)
        rounded = ((bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) >> 16).astype(np.uint16)
        tensor = pyneat.Tensor.from_numpy(
            rounded,
            copy=True,
            layout=pyneat.TensorLayout.HWC,
            memory=pyneat.TensorMemory.EV74,
        )
        tensor.dtype = pyneat.TensorDType.BFloat16
        return tensor
    raise RuntimeError("SuperPoint model input must be Float32 or BFloat16")


def select_input_dtype(spec, pyneat):
    for dtype in spec.dtypes:
        if dtype in (pyneat.TensorDType.Float32, pyneat.TensorDType.BFloat16):
            return dtype
    raise RuntimeError("SuperPoint model input must support Float32 or BFloat16")


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


def validate_frame(frame, np) -> None:
    if frame.shape != (HEIGHT, WIDTH, 3) or frame.dtype != np.uint8:
        raise RuntimeError("SuperPoint input must be 640x480 BGR video")


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
    if config.input.resolve() == config.output.resolve():
        print("Error: input and output paths must differ", file=sys.stderr)
        return 2
    if config.output.suffix != ".mp4":
        print("Error: io.output must use the .mp4 extension", file=sys.stderr)
        return 2

    import cv2
    import numpy as np
    import pyneat

    video = cv2.VideoCapture(str(config.input))
    ok, frame = video.read()
    if not video.isOpened() or not ok:
        print(f"Error: failed to read input video: {config.input}", file=sys.stderr)
        return 2

    config.output.parent.mkdir(parents=True, exist_ok=True)
    fps = video.get(cv2.CAP_PROP_FPS)
    fps = fps if np.isfinite(fps) and fps > 0 else 30.0
    writer = cv2.VideoWriter(
        str(config.output),
        cv2.CAP_FFMPEG,
        cv2.VideoWriter_fourcc(*"avc1"),
        fps,
        (WIDTH, HEIGHT),
    )
    if not writer.isOpened():
        video.release()
        print(f"Error: failed to open H.264 output video: {config.output}", file=sys.stderr)
        return 2

    runner = None
    try:
        validate_frame(frame, np)
        model = pyneat.Model(str(config.model), model_options(pyneat))
        input_specs = model.input_specs()
        if len(input_specs) != 1:
            raise RuntimeError("SuperPoint model must expose exactly one input")
        input_dtype = select_input_dtype(input_specs[0], pyneat)
        model_input = input_tensor(frame, input_dtype, cv2, np, pyneat)
        runner = model.build(
            [model_input],
            route_options=pyneat.ModelRouteOptions(),
            run_options=pyneat.RunOptions(),
        )

        processed = 0
        total_points = 0
        while True:
            output = runner.run([model_input], timeout_ms=config.timeout_ms)
            points = feature_points(output, np, pyneat)
            total_points += len(points)
            draw_points(frame, points, cv2)
            writer.write(frame)
            processed += 1

            if (config.frames > 0 and processed >= config.frames):
                break
            ok, frame = video.read()
            if not ok:
                break
            validate_frame(frame, np)
            model_input = input_tensor(frame, input_dtype, cv2, np, pyneat)

        average = total_points / processed
        print(
            f"frames={processed} average_points={average:.1f} "
            f"descriptor_dim={DESCRIPTOR_DIM} output={config.output}"
        )
        return 0
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    finally:
        if runner is not None:
            runner.close()
        video.release()
        writer.release()


if __name__ == "__main__":
    raise SystemExit(main())
