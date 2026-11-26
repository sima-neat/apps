"""Find SuperPoint feature points in one H.264 RTSP stream, shown in Insight.

The decoded frames feed the model. Insight receives the video as received and every frame's
feature points.
"""

from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path

import numpy as np
import pyneat

# The helpers the examples share live in <apps root>/support/python.
for apps_root in Path(__file__).resolve().parents:
    if (apps_root / "support" / "python" / "sima_examples").is_dir():
        sys.path.insert(0, str(apps_root / "support" / "python"))
        break
else:
    raise SystemExit("support/python not found above this example; reinstall the Apps package")

from sima_examples.config import Config
from sima_examples.rtsp_probe import RtspStreamInfo, probe_rtsp_stream_info
from sima_examples.stop_signals import stop_on_signals
from utils import feature_points

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "common" / "config.yaml"
PULL_TIMEOUT_MS = 100


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args(argv)


def model_options(stream: RtspStreamInfo) -> pyneat.ModelOptions:
    """SuperPoint on the gray frames, stretched to the model input, with its keypoints and
    descriptors decoded."""
    options = pyneat.ModelOptions()
    options.preprocess.kind = pyneat.InputKind.Image
    options.preprocess.enable = pyneat.AutoFlag.On
    options.preprocess.color_convert.input_format = pyneat.PreprocessColorFormat.GRAY8
    options.preprocess.input_max_width = stream.width
    options.preprocess.input_max_height = stream.height
    options.preprocess.resize.enable = pyneat.AutoFlag.On
    options.preprocess.resize.mode = pyneat.ResizeMode.Stretch
    # The default mean 0 and stddev 1, marked explicit, turn normalization on: pixels / 255.
    options.preprocess.normalize.has_explicit_stats = True
    options.decode_type = pyneat.BoxDecodeType.SuperPoint
    # A65V1 sets the decoder's numerics and its defaults: score threshold 0.1 while
    # score_threshold is 0, and border margin 0. Auto would also give A65V1 for this package,
    # which names no profile, but would follow a package that does. The 600-point limit is
    # Core's default top_k for every profile.
    options.superpoint.profile = pyneat.SuperPointProfile.A65V1
    return options


def model_size(model: pyneat.Model) -> tuple[int, int]:
    """The frame the decoded keypoints are in: the model input, `(width, height)`."""
    contract = model.resolved_preprocess_plan().mla_contract
    if contract.width <= 0 or contract.height <= 0:
        raise RuntimeError("SuperPoint model input size is unresolved")
    return contract.width, contract.height


def rtsp_options(config: Config, stream: RtspStreamInfo) -> pyneat.RtspEncodedInputOptions:
    options = pyneat.RtspEncodedInputOptions()
    options.url = config.string("input.source")
    options.tcp = config.boolean("input.rtsp.tcp")
    options.latency_ms = config.integer("input.rtsp.latency_ms")
    # With input.fps at 0, Core also takes the caps from the stream: above 60 fps, caps set from
    # the probed values alone failed to negotiate in some runs.
    options.auto_caps_from_stream = config.integer("input.fps") == 0
    options.source_fps = stream.fps
    options.fallback_h264_width = stream.width
    options.fallback_h264_height = stream.height
    return options


def decoder_graph(stream: RtspStreamInfo) -> pyneat.Graph:
    """Hardware decode of the H.264 stream to NV12 frames. The "frames" output keeps only the
    newest, so a pull loop slower than the stream cannot hold back the video sender."""
    decode = pyneat.SimaDecodeOptions()
    decode.dec_width = stream.width
    decode.dec_height = stream.height
    decode.dec_fps = stream.fps
    graph = pyneat.Graph("decoder")
    graph.add(pyneat.nodes.sima_decode(decode))
    graph.add(pyneat.nodes.output("frames", pyneat.OutputOptions.latest()))
    return graph


def video_options(config: Config) -> pyneat.VideoSenderOptions:
    """The H.264 stream goes to Insight as received."""
    options = pyneat.VideoSenderOptions.passthrough(pyneat.RtspCodec.H264)
    options.host = config.string("output.insight.host")
    options.video_port_base = config.integer("output.insight.video_port_base")
    return options


def metadata_options(config: Config) -> pyneat.MetadataSenderOptions:
    options = pyneat.MetadataSenderOptions()
    options.host = config.string("output.insight.host")
    options.metadata_port_base = config.integer("output.insight.metadata_port_base")
    return options


def run_options() -> pyneat.RunOptions:
    options = pyneat.RunOptions()
    options.preset = pyneat.RunPreset.Realtime
    options.output_memory = pyneat.OutputMemory.ZeroCopy
    return options


def gray_tensor(gray: np.ndarray) -> pyneat.Tensor:
    """A gray image as the model's input tensor, copied into EV74 memory, from_numpy's default."""
    return pyneat.Tensor.from_numpy(gray, image_format=pyneat.PixelFormat.GRAY8)


def luma(sample: pyneat.Sample, stream: RtspStreamInfo) -> pyneat.Tensor:
    """The Y plane of a decoded NV12 frame as the model's input: the gray image SuperPoint
    reads."""
    # Core candidate: Preproc converts no YUV format to gray, and a VideoConvert before the model
    # drops the frame metadata Preproc needs, so the host hands the model the Y plane.
    # Core candidate: pyneat has no NV12 plane view. copy_payload_bytes copies the frame twice
    # inside Core, the Y plane is copied once more (below), and gray_tensor copies it into EV74.
    tensors = [sample.tensor] if sample.kind == pyneat.SampleKind.Tensor else sample.tensors
    if not tensors:
        raise RuntimeError("decoded frame has no tensor")
    tensor = tensors[0]
    if not tensor.is_nv12():
        raise RuntimeError("decoded frame is not NV12")
    if (tensor.width(), tensor.height()) != (stream.width, stream.height):
        raise RuntimeError(
            f"frame size {tensor.width()}x{tensor.height()} differs from the stream's "
            f"{stream.width}x{stream.height}; the stream must keep one size"
        )
    plane = next(plane for plane in tensor.planes if plane.role == pyneat.PlaneRole.Y)
    stride = plane.strides_bytes[0] if plane.strides_bytes else tensor.width()
    y_plane = np.ndarray(
        (tensor.height(), tensor.width()),
        np.uint8,
        buffer=tensor.copy_payload_bytes(),
        offset=plane.byte_offset,
        strides=(stride, 1),
    )
    # Core candidate: a view of the bytes is read-only, and from_numpy exports it through
    # __dlpack__() without max_version, which NumPy refuses for a read-only array (BufferError),
    # so the plane is copied.
    return gray_tensor(y_plane.copy())


def send_feature_points(
    run: pyneat.Run,
    runner: pyneat.ModelRunner,
    metadata_sender: pyneat.MetadataSender,
    model_size: tuple[int, int],
    stream: RtspStreamInfo,
    frames: int,
    stop: threading.Event,
) -> tuple[int, int, int]:
    """Run the model on every frame and send its feature points to Insight until `frames`
    (0: no limit) or a stop signal; return the frame count, the point count, and the descriptor
    length."""
    processed = points_sent = descriptor_dim = 0
    while not stop.is_set() and (frames == 0 or processed < frames):
        sample = run.pull("frames", PULL_TIMEOUT_MS)
        if sample is None:
            # None is a timeout or a closed output. With one output, can_pull() tells which.
            if run.last_error() or not run.running() or not run.can_pull():
                raise RuntimeError(f"frames output closed: {run.last_error()}")
            continue
        outputs = runner.run([luma(sample, stream)])
        features = pyneat.decode_superpoint(list(outputs))[0]
        points = feature_points.source_points(
            features.keypoints.to_numpy(), model_size, (stream.width, stream.height)
        )
        # A nonblocking send drops the message while the socket buffer is full; the drops are
        # counted in the sender's stats and reported when the run ends. Any other failure raises.
        # The frame's timestamp lets Insight draw the points on the video frame they belong to.
        metadata_sender.send_metadata(
            "pose-estimation",
            feature_points.points_json(points),
            sample.pts_ns // 1_000_000 if sample.pts_ns >= 0 else -1,
            str(sample.frame_id) if sample.frame_id >= 0 else "",
        )
        processed += 1
        points_sent += len(points)
        descriptor_dim = int(features.descriptors.shape[1])
    return processed, points_sent, descriptor_dim


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = Config.load(args.config)
    # The stream's size and rate come from the probe for each value the config leaves at 0.
    stream = probe_rtsp_stream_info(
        config.string("input.source"),
        config.boolean("input.rtsp.tcp"),
        RtspStreamInfo(
            config.integer("input.width"),
            config.integer("input.height"),
            config.integer("input.fps"),
        ),
    )
    model_path = args.config.absolute().parent / config.string("model.path")
    model = pyneat.Model(str(model_path), model_options(stream))
    runner = model.build([gray_tensor(np.zeros((stream.height, stream.width), np.uint8))])
    run = None
    try:
        video = pyneat.groups.video_sender(video_options(config))
        graph = pyneat.Graph()
        source = pyneat.groups.rtsp_encoded_input(rtsp_options(config, stream))
        graph.connect(source, decoder_graph(stream))
        # A plain link lets Core tee the encoded stream to the decoder and the video sender.
        graph.connect(source, video)

        metadata_sender = pyneat.MetadataSender(metadata_options(config))
        stop = stop_on_signals()
        run = graph.build(run_options())
        host = config.string("output.insight.host")
        video_port = config.integer("output.insight.video_port_base")
        print(
            f"source={config.string('input.source')} "
            f"stream={stream.width}x{stream.height}@{stream.fps} insight={host} "
            f"video={video_port} metadata={config.integer('output.insight.metadata_port_base')}"
        )
        processed, points_sent, descriptor_dim = send_feature_points(
            run,
            runner,
            metadata_sender,
            model_size(model),
            stream,
            config.integer("inference.frames"),
            stop,
        )
    finally:
        if run is not None:
            run.close()
        runner.close()

    average = points_sent / processed if processed else 0.0
    print(
        f"frames={processed} average_points={average:.1f} descriptor_dim={descriptor_dim} "
        f"video_sender={host}:{video_port}"
    )
    if metadata_sender.stats().send_failures:
        print(f"{metadata_sender.stats().send_failures} metadata messages dropped")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:  # noqa: BLE001 - report the cause in one line, not a traceback
        print(f"[ERR] {error}", file=sys.stderr)
        sys.exit(1)
