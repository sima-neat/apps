"""Local model setup, decoding and numerical helpers for this example."""
from __future__ import annotations

import argparse

import os

import signal

import sys

from collections import namedtuple

from pathlib import Path

import yaml

DESCRIPTOR_DIM = 256

MAX_POINTS = 600

FrameSize = namedtuple("FrameSize", ["width", "height"])

def load_config(path: Path) -> argparse.Namespace:
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    model = raw.get("model", {})
    source = raw.get("source", {})
    insight = raw.get("output", {}).get("insight", {})
    runtime = raw.get("runtime", {})
    config = argparse.Namespace(
        model=Path(model.get("path", "models/superpoint/w8a8/superpoint_w8a8_mpk.tar.gz")),
        rtsp_url=str(source.get("rtsp_url", "")),
        tcp=bool(source.get("tcp", True)),
        latency_ms=int(source.get("latency_ms", 100)),
        insight_host=str(insight.get("host", "127.0.0.1")),
        video_port=int(insight.get("video_port", 9000)),
        channel=int(insight.get("channel", 0)),
        bitrate_kbps=int(insight.get("bitrate_kbps", 1000)),
        frames=int(runtime.get("frames", 0)),
    )
    if config.frames < 0:
        raise ValueError("runtime.frames must be >= 0")
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

def model_input_size(model):
    """Size of the tensor Preproc hands to the MLA: the frame of the decoded keypoints."""
    contract = model.resolved_preprocess_plan().mla_contract
    if contract.width <= 0 or contract.height <= 0:
        raise RuntimeError("SuperPoint model input size is unresolved")
    return FrameSize(int(contract.width), int(contract.height))

def feature_points(output, model_size, source_size, np, pyneat):
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

    # The decoder reports model-input pixels (sima-neat/core#1328). Preproc stretched
    # the source frame to the model input, so each axis maps back by its own ratio.
    points = points * np.asarray(
        [
            source_size.width / model_size.width,
            source_size.height / model_size.height,
        ],
        dtype=np.float32,
    )

    # The frame edges are exclusive coordinates. Discard isolated decoder outliers
    # instead of aborting the stream when a point lands on or beyond an edge.
    drawable = (
        np.all(np.isfinite(points), axis=1)
        & (points[:, 0] >= 0)
        & (points[:, 0] < source_size.width)
        & (points[:, 1] >= 0)
        & (points[:, 1] < source_size.height)
    )
    return points[drawable]

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
