"""Detect objects on many RTSP streams with one shared YOLO26 model.

Every stream's video and its detection boxes go to Insight, one channel per stream.
"""

from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path

import pyneat

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "common" / "config.yaml"

# The helpers the examples share live in <apps root>/support/python.
for apps_root in Path(__file__).resolve().parents:
    if (apps_root / "support" / "python" / "sima_examples").is_dir():
        sys.path.insert(0, str(apps_root / "support" / "python"))
        break
else:
    raise SystemExit("support/python not found above this example; reinstall the Apps package")

from sima_examples import detection_egress
from sima_examples.config import Config
from sima_examples.labels import load_labels
from sima_examples.rtsp_probe import RtspStreamInfo, probe_rtsp_stream_info
from sima_examples.stop_signals import stop_on_signals

RTSP_CODECS = {"h264": pyneat.RtspCodec.H264, "h265": pyneat.RtspCodec.H265}
DECODE_TYPES = {"h264": pyneat.SimaDecodeType.H264, "h265": pyneat.SimaDecodeType.H265}
PULL_TIMEOUT_MS = 100


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args(argv)


def model_options(config: Config, max_detections: int) -> pyneat.ModelOptions:
    options = pyneat.ModelOptions()
    options.verbose = pyneat.VerboseOptions.quiet()
    options.preprocess.kind = pyneat.InputKind.Image
    options.preprocess.enable = pyneat.AutoFlag.On
    options.preprocess.color_convert.input_format = pyneat.PreprocessColorFormat.NV12
    options.preprocess.preset = pyneat.NormalizePreset.COCO_YOLO
    options.decode_type = pyneat.BoxDecodeType.YoloV26
    options.score_threshold = config.number("inference.min_score")
    options.nms_iou_threshold = config.number("inference.nms_iou")
    options.top_k = max_detections
    return options


def detector_graph(model: pyneat.Model, config: Config) -> pyneat.Graph:
    """The model every stream shares: "detector_frame" input -> model -> "detections" output."""
    frames = pyneat.Graph()
    frames.add(pyneat.nodes.input("detector_frame", model.input_appsrc_options(False)))

    detections = pyneat.Graph()
    detections.add(
        pyneat.nodes.output(
            "detections", pyneat.OutputOptions.every_frame(config.integer("app.queue_depth"))
        )
    )

    model_graph = model.graph()
    graph = pyneat.Graph()
    graph.connect(frames, model_graph)
    graph.connect(model_graph, detections)
    return graph


def rtsp_options(
    config: Config, url: str, stream: RtspStreamInfo
) -> pyneat.RtspEncodedInputOptions:
    options = pyneat.RtspEncodedInputOptions()
    options.url = url
    options.codec = config.choice("input.codec", RTSP_CODECS)
    options.tcp = config.boolean("input.rtsp.tcp")
    options.latency_ms = config.integer("input.rtsp.latency_ms")
    options.source_fps = config.integer("input.fps")
    options.auto_caps_from_stream = options.source_fps == 0
    if options.codec == pyneat.RtspCodec.H264:
        options.fallback_h264_width = stream.width
        options.fallback_h264_height = stream.height
        options.fallback_h264_fps = stream.fps
    return options


def decoder_graph(config: Config, stream: RtspStreamInfo) -> pyneat.Graph:
    """Hardware decode to NV12 frames at the stream's size, ending at "detector_frame"."""
    decode = pyneat.SimaDecodeOptions()
    decode.type = config.choice("input.codec", DECODE_TYPES)
    decode.decoder_name = "decoder"
    decode.next_element = "CVU"
    decode.dec_width = stream.width
    decode.dec_height = stream.height
    decode.dec_fps = stream.fps
    decode.num_buffers = config.integer("app.decoder_buffers")
    decode.input_buffers = 2
    decode.decoder_tuning = "throughput-low-latency"
    decode.memory_opt = True
    graph = pyneat.Graph("decoder")
    graph.add(pyneat.nodes.sima_decode(decode))
    graph.add(
        pyneat.nodes.caps_raw(
            "NV12",
            stream.width,
            stream.height,
            config.integer("input.fps"),
            pyneat.CapsMemory.Any,
        )
    )
    graph.add(pyneat.nodes.output("detector_frame"))
    return graph


def stream_id(index: int) -> str:
    return f"stream{index}"


def detector_link(config: Config, index: int) -> pyneat.GraphLinkOptions:
    """Feed the shared model the latest frame of each stream, with bounded frames in flight."""
    link = pyneat.GraphLinkOptions()
    link.policy = pyneat.GraphLinkPolicy.RealtimeLatestByStream
    link.stream_id = stream_id(index)
    link.max_inflight_per_stream = config.integer("inference.max_inflight_per_stream")
    link.max_inflight_total = config.integer("inference.max_inflight_total")
    return link


def video_link() -> pyneat.GraphLinkOptions:
    link = pyneat.GraphLinkOptions()
    link.policy = pyneat.GraphLinkPolicy.RealtimeLatestByStream
    return link


def video_options(config: Config, index: int) -> pyneat.VideoSenderOptions:
    """Send the encoded stream to Insight unchanged, on video port base + index."""
    options = pyneat.VideoSenderOptions.passthrough(config.choice("input.codec", RTSP_CODECS))
    options.host = config.string("output.insight.host")
    options.channel = index
    options.video_port_base = config.integer("output.insight.video_port_base")
    return options


def metadata_options(config: Config, index: int) -> pyneat.MetadataSenderOptions:
    options = pyneat.MetadataSenderOptions()
    options.host = config.string("output.insight.host")
    options.channel = index
    options.metadata_port_base = config.integer("output.insight.metadata_port_base")
    return options


def graph_options(config: Config) -> pyneat.GraphOptions:
    options = pyneat.GraphOptions()
    options.advanced_execution.internal_queue_depth = config.integer("app.internal_queue_depth")
    return options


def run_options(config: Config) -> pyneat.RunOptions:
    options = pyneat.RunOptions()
    options.preset = pyneat.RunPreset.Realtime
    options.queue_depth = config.integer("app.queue_depth")
    options.output_memory = pyneat.OutputMemory.ZeroCopy
    return options


def stream_info(config: Config, sources: list[str]) -> list[RtspStreamInfo]:
    """Each stream's frame size and rate: the configured values, and a probe's for each 0.

    Every stream has the same size, because the streams share one model input.
    """
    configured = RtspStreamInfo(
        config.integer("input.width"),
        config.integer("input.height"),
        config.integer("input.fps"),
    )
    tcp = config.boolean("input.rtsp.tcp")
    by_url = {url: probe_rtsp_stream_info(url, tcp, configured) for url in set(sources)}
    # Core gap: a model shared by several streams accepts one input size, and a stream of
    # another size fails the build with "input spec mismatch".
    (first_url, first), *others = by_url.items()
    for url, stream in others:
        if (stream.width, stream.height) != (first.width, first.height):
            raise ValueError(
                "every input.source stream needs the same frame size, because the streams "
                f"share one model: {first_url} is {first.width}x{first.height}, "
                f"{url} is {stream.width}x{stream.height}"
            )
    return [by_url[url] for url in sources]


def decode_boxes(sample: pyneat.Sample, stream: RtspStreamInfo, max_detections: int) -> list:
    """The boxes in a detections sample, whose first tensor is the BBOX payload."""
    if sample.kind == pyneat.SampleKind.TensorSet and sample.tensors:
        bbox = sample.tensors[0]
    elif sample.kind == pyneat.SampleKind.Tensor:
        bbox = sample.tensor
    else:
        raise RuntimeError(f"expected a BBOX tensor, got {sample.kind}")
    result = pyneat.detections.decode_bbox_tensor(
        bbox, stream.width, stream.height, max_detections, strict=True
    )
    return result.boxes


def send_detections(
    run: pyneat.Run,
    metadata_senders: list[pyneat.MetadataSender],
    streams: list[RtspStreamInfo],
    labels: list[str],
    max_detections: int,
    stop: threading.Event,
) -> None:
    """Send every detections sample to its stream's Insight channel until a stop signal."""
    while not stop.is_set():
        sample = run.pull("detections", PULL_TIMEOUT_MS)
        if sample is None:
            if run.last_error() or not run.running():
                raise RuntimeError(f"detections output closed: {run.last_error()}")
            continue

        # With one source, Core names the sample after the decoder instead of the stream.
        index = 0
        if len(metadata_senders) > 1:
            index = int(sample.stream_id.removeprefix("stream"))
        stream = streams[index]
        payload = detection_egress.serialize(
            decode_boxes(sample, stream, max_detections),
            labels,
            stream.width,
            stream.height,
            detection_egress.frame_metadata(sample, index),
        )
        # A nonblocking send drops the message while the socket buffer is full; the drops are
        # counted in the sender's stats and reported when the run ends. Any other failure raises.
        metadata_senders[index].send_raw_json(payload)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = Config.load(args.config)
    sources = config.string_list("input.source")
    streams = stream_info(config, sources)
    labels = load_labels(Path(config.string("model.labels")))
    model_path = args.config.resolve().parent / config.string("model.path")
    # Used by both the model and the box decode.
    max_detections = config.integer("inference.max_detections")
    model = pyneat.Model(str(model_path), model_options(config, max_detections))
    detector = detector_graph(model, config)

    graph = pyneat.Graph(options=graph_options(config))
    metadata_senders = []
    for index, url in enumerate(sources):
        rtsp = pyneat.groups.rtsp_encoded_input(rtsp_options(config, url, streams[index]))
        decoder = decoder_graph(config, streams[index])
        graph.connect(rtsp, decoder)
        graph.connect(decoder, detector, detector_link(config, index))
        if config.boolean("output.video_enabled"):
            graph.connect(
                rtsp,
                pyneat.groups.video_sender(video_options(config, index)),
                video_link(),
            )
        metadata_senders.append(pyneat.MetadataSender(metadata_options(config, index)))
    print(
        f"streams={len(sources)} insight={config.string('output.insight.host')} "
        f"video_ports={config.integer('output.insight.video_port_base')}+ "
        f"metadata_ports={config.integer('output.insight.metadata_port_base')}+"
    )

    stop = stop_on_signals()
    run = graph.build(run_options(config))
    try:
        send_detections(
            run,
            metadata_senders,
            streams,
            labels,
            max_detections,
            stop,
        )
    finally:
        run.close()

    for index, sender in enumerate(metadata_senders):
        if sender.stats().send_failures:
            print(
                f"stream {index}: {sender.stats().send_failures} metadata messages dropped"
            )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:  # noqa: BLE001 - report the cause in one line, not a traceback
        print(f"[ERR] {error}", file=sys.stderr)
        sys.exit(1)
