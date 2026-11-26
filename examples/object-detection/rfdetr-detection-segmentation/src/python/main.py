"""Detect or segment objects in one RTSP stream with RF-DETR and show them in Insight.

The backbone runs in the graph; the host picks its best proposals and hands them, with the
backbone's features, to the transformer. Insight receives the video and, for every result,
the detection boxes or the segmentation polygons.
"""

from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path

import numpy as np
import pyneat

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "common" / "config.yaml"

# The helpers the examples share live in <apps root>/support/python.
for apps_root in Path(__file__).resolve().parents:
    if (apps_root / "support" / "python" / "sima_examples").is_dir():
        sys.path.insert(0, str(apps_root / "support" / "python"))
        break
else:
    raise SystemExit("support/python not found above this example; reinstall the Apps package")

from sima_examples.config import Config
from sima_examples.labels import load_labels
from sima_examples.rtsp_probe import RtspStreamInfo, probe_rtsp_stream_info
from sima_examples.stop_signals import stop_on_signals
from utils import postprocess

RTSP_CODECS = {
    "h264": pyneat.RtspCodec.H264,
    "h265": pyneat.RtspCodec.H265,
    "mjpeg": pyneat.RtspCodec.MJPEG,
}
DECODE_TYPES = {
    "h264": pyneat.SimaDecodeType.H264,
    "h265": pyneat.SimaDecodeType.H265,
    "mjpeg": pyneat.SimaDecodeType.MJPEG,
}
PULL_TIMEOUT_MS = 100
# The frame identity a transformer input carries over from its backbone result.
FRAME_IDENTITY = (
    "frame_id",
    "stream_id",
    "stream_label",
    "input_seq",
    "orig_input_seq",
    "pts_ns",
    "dts_ns",
    "duration_ns",
    "attributes",
)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args(argv)


def backbone_options(stream: RtspStreamInfo) -> pyneat.ModelOptions:
    """EV74 converts each NV12 frame to RGB, stretches it to the model input, and normalizes it."""
    options = pyneat.ModelOptions()
    options.preprocess.kind = pyneat.InputKind.Image
    options.preprocess.enable = pyneat.AutoFlag.On
    options.preprocess.input_max_width = stream.width
    options.preprocess.input_max_height = stream.height
    options.preprocess.input_max_depth = 3
    options.preprocess.resize.enable = pyneat.AutoFlag.On
    options.preprocess.resize.mode = pyneat.ResizeMode.Stretch
    options.preprocess.color_convert.enable = pyneat.AutoFlag.On
    options.preprocess.color_convert.input_format = pyneat.PreprocessColorFormat.NV12
    options.preprocess.color_convert.output_format = pyneat.PreprocessColorFormat.RGB
    options.preprocess.preset = pyneat.NormalizePreset.ImageNet
    options.processcvu.pre_run_target = "EV74"
    options.processcvu.post_run_target = "A65"
    return options


def transformer_options() -> pyneat.ModelOptions:
    options = pyneat.ModelOptions()
    options.processcvu.pre_run_target = "A65"
    options.processcvu.post_run_target = "A65"
    return options


def rtsp_options(config: Config, stream: RtspStreamInfo) -> pyneat.RtspEncodedInputOptions:
    options = pyneat.RtspEncodedInputOptions()
    options.url = config.string("input.source")
    options.codec = config.choice("input.codec", RTSP_CODECS)
    options.tcp = config.boolean("input.rtsp.tcp")
    options.latency_ms = config.integer("input.rtsp.latency_ms")
    if options.tcp:
        # Keep the RTP timestamps; arrival-time corrections drift during high-rate replay.
        options.buffer_mode = "none"
    options.source_fps = stream.fps
    if options.codec == pyneat.RtspCodec.H264:
        options.fallback_h264_width = stream.width
        options.fallback_h264_height = stream.height
    return options


def decoder_graph(config: Config, stream: RtspStreamInfo) -> pyneat.Graph:
    """Hardware decode to NV12 frames at the stream's size."""
    decode = pyneat.SimaDecodeOptions()
    decode.type = config.choice("input.codec", DECODE_TYPES)
    decode.dec_width = stream.width
    decode.dec_height = stream.height
    decode.dec_fps = stream.fps
    graph = pyneat.Graph("decoder")
    graph.add(pyneat.nodes.sima_decode(decode))
    return graph


def video_options(config: Config) -> pyneat.VideoSenderOptions:
    """Send the stream to Insight as received."""
    options = pyneat.VideoSenderOptions.passthrough(config.choice("input.codec", RTSP_CODECS))
    options.host = config.string("output.insight.host")
    options.video_port_base = config.integer("output.insight.video_port_base")
    return options


def backbone_graph(backbone: pyneat.Model) -> pyneat.Graph:
    """Newest decoded frames -> backbone -> "backbone" output.

    The queue replaces its oldest frame instead of slowing the decoder when inference lags.
    """
    newest_frames = pyneat.QueueOptions()
    newest_frames.max_buffers = 4
    newest_frames.overflow_policy = pyneat.OverflowPolicy.KeepLatest
    graph = pyneat.Graph("inference")
    graph.add(pyneat.nodes.queue(newest_frames))
    graph.add(backbone.graph())
    graph.add(pyneat.nodes.output("backbone", pyneat.OutputOptions.every_frame(8)))
    return graph


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


def transformer_run_options() -> pyneat.RunOptions:
    """Copy each result to host memory, so reading it holds none of the transformer's buffers."""
    options = pyneat.RunOptions()
    options.output_memory = pyneat.OutputMemory.Owned
    return options


def transformer_seed(transformer: pyneat.Model) -> list[pyneat.Tensor]:
    """Zero tensors shaped like the transformer's inputs, which the build runs once."""
    return [
        pyneat.Tensor.from_numpy(
            np.zeros(tuple(spec.shape), dtype=np.float32),
            copy=True,
            memory=pyneat.TensorMemory.EV74,
        )
        for spec in transformer.input_specs()
    ]


def postprocess_options(config: Config, stream: RtspStreamInfo) -> postprocess.PostprocessOptions:
    return postprocess.PostprocessOptions(
        width=stream.width,
        height=stream.height,
        min_score=config.number("inference.min_score"),
        max_detections=config.integer("inference.max_detections"),
        mask_threshold=config.number("inference.mask_threshold"),
        mask_grid_size=config.integer("app.mask_grid_size"),
    )


def transformer_sample(
    backbone_result: pyneat.Sample, proposals_shape: list[int]
) -> pyneat.Sample:
    """The transformer's input for one frame: the backbone features and the best proposals."""
    feature, scores, proposals = backbone_result.tensors
    if feature.storage.kind in (
        pyneat.StorageKind.CpuOwned,
        pyneat.StorageKind.CpuExternal,
    ):
        # The transformer reads its features from device memory; a device-backed feature map
        # passes without a copy.
        feature = feature.cvu()
    best = postprocess.top_proposals(
        scores.to_numpy(copy=False), proposals.to_numpy(copy=False), proposals_shape[1]
    )
    sample = pyneat.Sample()
    sample.kind = pyneat.SampleKind.TensorSet
    sample.tensors = [
        feature,
        pyneat.Tensor.from_numpy(
            best.reshape(proposals_shape), copy=True, memory=pyneat.TensorMemory.EV74
        ),
    ]
    for name in FRAME_IDENTITY:
        setattr(sample, name, getattr(backbone_result, name))
    return sample


def result_json(
    result: pyneat.Sample,
    labels: list[str],
    mask_size: int,
    options: postprocess.PostprocessOptions,
) -> str:
    """The metadata of one transformer result: detection boxes, or segmentation polygons."""
    values = [tensor.to_numpy(copy=False) for tensor in result.tensors]
    if mask_size:
        return postprocess.segmentation_json(
            values[0], values[1], values[2], mask_size, labels, options
        )
    return postprocess.detection_json(values[0], values[1], labels, options)


def feed_transformer(
    source_run: pyneat.Run,
    transformer_runner: pyneat.ModelRunner,
    proposals_shape: list[int],
    stop: threading.Event,
    errors: list[Exception],
) -> None:
    """Hand every backbone result to the transformer until the run stops.

    A push waits while the transformer's input queue is full, so no backbone result is lost.
    """
    try:
        while not stop.is_set():
            sample = source_run.pull("backbone", PULL_TIMEOUT_MS)
            if sample is None:
                if source_run.last_error() or not source_run.running():
                    raise RuntimeError(
                        f"backbone output closed: {source_run.last_error()}"
                    )
                continue
            if not transformer_runner.push_samples(
                transformer_sample(sample, proposals_shape)
            ):
                raise RuntimeError("the transformer rejected its input")
    except Exception as error:  # noqa: BLE001 - main reports the feeder's failure
        if not stop.is_set():
            errors.append(error)
    finally:
        stop.set()


def send_results(
    transformer_runner: pyneat.ModelRunner,
    metadata_sender: pyneat.MetadataSender,
    labels: list[str],
    mask_size: int,
    options: postprocess.PostprocessOptions,
    frames: int,
    stop: threading.Event,
) -> int:
    """Send every transformer result to Insight until `frames` (0: no limit) or a stop."""
    metadata_type = "segmentation" if mask_size else "object-detection"
    sent = 0
    while not stop.is_set() and (frames == 0 or sent < frames):
        result = transformer_runner.pull(timeout_ms=PULL_TIMEOUT_MS)
        if not result:
            continue
        # A nonblocking send drops the message while the socket buffer is full; the drops
        # are counted in the sender's stats and reported when the run ends. Any other
        # failure raises.
        metadata_sender.send_metadata(
            metadata_type,
            result_json(result, labels, mask_size, options),
            result.pts_ns // 1_000_000 if result.pts_ns >= 0 else -1,
            str(result.frame_id) if result.frame_id >= 0 else "",
        )
        sent += 1
    return sent


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = Config.load(args.config)
    stream = probe_rtsp_stream_info(
        config.string("input.source"),
        config.boolean("input.rtsp.tcp"),
        RtspStreamInfo(
            config.integer("input.width"),
            config.integer("input.height"),
            config.integer("input.fps"),
        ),
    )
    model_id = config.string("model.id")
    # A model comes as two packages: the backbone that model.path names, and the decoder,
    # which the code calls the transformer, next to it.
    backbone_path = args.config.resolve().parent / config.string("model.path")
    transformer_path = backbone_path.with_name(
        backbone_path.name.replace("_backbone_mpk", "_decoder_mpk")
    )
    backbone = pyneat.Model(str(backbone_path), backbone_options(stream))
    transformer = pyneat.Model(str(transformer_path), transformer_options())
    proposals_shape = list(transformer.input_specs()[1].shape)
    segmentation = model_id.startswith("seg-")
    mask_size = int(transformer.output_specs()[2].shape[1]) if segmentation else 0
    # The model's class IDs index the labels directly.
    labels = load_labels(Path(config.string("model.labels")))

    rtsp = pyneat.groups.rtsp_encoded_input(rtsp_options(config, stream))
    decoder = decoder_graph(config, stream)
    video = pyneat.groups.video_sender(video_options(config))
    graph = pyneat.Graph()
    graph.connect(rtsp, decoder)
    # A plain link lets Core tee the encoded stream to the decoder and the video sender.
    graph.connect(rtsp, video)
    graph.connect(decoder, backbone_graph(backbone))

    transformer_runner = transformer.build(
        transformer_seed(transformer), run_options=transformer_run_options()
    )
    source_run = graph.build(run_options())
    metadata_sender = pyneat.MetadataSender(metadata_options(config))
    print(
        f"RF-DETR {model_id} {config.string('input.codec')}: {config.string('input.source')} "
        f"({stream.width}x{stream.height}@{stream.fps}) "
        f"-> Insight {config.string('output.insight.host')} "
        f"video={config.integer('output.insight.video_port_base')} "
        f"metadata={config.integer('output.insight.metadata_port_base')}"
    )

    stop = stop_on_signals()
    feed_errors: list[Exception] = []
    feeder = threading.Thread(
        target=feed_transformer,
        args=(source_run, transformer_runner, proposals_shape, stop, feed_errors),
    )
    feeder.start()
    try:
        sent = send_results(
            transformer_runner,
            metadata_sender,
            labels,
            mask_size,
            postprocess_options(config, stream),
            config.integer("inference.frames"),
            stop,
        )
    finally:
        stop.set()
        transformer_runner.close_input()  # Ends a push the feeder may be blocked in.
        feeder.join()
        transformer_runner.close()
        source_run.close()
    if feed_errors:
        raise feed_errors[0]

    print(f"RF-DETR {model_id}: sent {sent} results to Insight")
    if metadata_sender.stats().send_failures:
        print(f"{metadata_sender.stats().send_failures} metadata messages dropped")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:  # noqa: BLE001 - report the cause in one line, not a traceback
        print(f"[ERR] {error}", file=sys.stderr)
        sys.exit(1)
