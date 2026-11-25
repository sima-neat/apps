"""Track tiny drones in up to four RTSP streams with one shared YOLO26 model, shown in Insight.

Each stream's decoded frames feed the model. Each stream's tracker gives every drone an ID, and
Insight draws each frame's tracks over that stream's video, one channel per stream.
"""

from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path

import pyneat

# The helpers the examples share live in <apps root>/support/python.
for apps_root in Path(__file__).resolve().parents:
    if (apps_root / "support" / "python" / "sima_examples").is_dir():
        sys.path.insert(0, str(apps_root / "support" / "python"))
        break
else:
    raise SystemExit("support/python not found above this example; reinstall the Apps package")

from sima_examples import detection_egress
from sima_examples.config import Config
from sima_examples.rtsp_probe import RtspStreamInfo, probe_rtsp_stream_info
from sima_examples.stop_signals import stop_on_signals
from utils import tracker

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "common" / "config.yaml"

RTSP_CODECS = {"h264": pyneat.RtspCodec.H264, "h265": pyneat.RtspCodec.H265}
DECODE_TYPES = {"h264": pyneat.SimaDecodeType.H264, "h265": pyneat.SimaDecodeType.H265}
# The model's one class.
LABELS = ["drone"]
# Frames all streams together may have in the shared model: 4, one stream's default
# (RealtimeLinkOptions.h). Core's default total, min(4 * streams, 8) (EdgeRouter.cpp), holds more
# from two streams on. Whether more frames in flight give more results per second is not measured.
MAX_INFLIGHT_TOTAL = 4
PULL_TIMEOUT_MS = 100


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args(argv)


def model_options(
    config: Config, stream: RtspStreamInfo, max_detections: int
) -> pyneat.ModelOptions:
    """YOLO26 on the decoded NV12 frames of every stream, which all have the size of `stream`."""
    options = pyneat.ModelOptions()
    options.preprocess.kind = pyneat.InputKind.Image
    options.preprocess.enable = pyneat.AutoFlag.On
    options.preprocess.color_convert.input_format = pyneat.PreprocessColorFormat.NV12
    # Lets streams above Core's 1920x1080 default in (PreprocessPlan.h).
    options.preprocess.input_max_width = stream.width
    options.preprocess.input_max_height = stream.height
    options.preprocess.preset = pyneat.NormalizePreset.COCO_YOLO
    options.decode_type = pyneat.BoxDecodeType.YoloV26
    # The split heads of a single-class YOLO may not give Core the class count, so
    # Model::Options::num_classes (model/Model.h) sets it.
    options.num_classes = len(LABELS)
    options.score_threshold = config.number("inference.min_score")
    options.nms_iou_threshold = config.number("inference.nms_iou")
    options.top_k = max_detections
    return options


def detector_graph(model: pyneat.Model) -> pyneat.Graph:
    """The model every stream shares: "detector_frame" input -> model -> "detections" output."""
    frames = pyneat.Graph()
    frames.add(pyneat.nodes.input("detector_frame", model.input_appsrc_options(False)))

    detections = pyneat.Graph()
    detections.add(pyneat.nodes.output("detections"))

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
    # With input.fps at 0, Core also takes the H.264 caps from the stream: above 60 fps, caps set
    # from the probed values alone failed to negotiate in some runs. The H.265 path does not read
    # auto_caps_from_stream (RtspEncodedInput.cpp).
    options.auto_caps_from_stream = config.integer("input.fps") == 0
    # The stream's rate in the encoded caps. With input.fps set, the H.264 explicit caps need it
    # (RtspEncodedInput.cpp), and H.265 writes it into its caps (H265Depacketize.cpp).
    options.source_fps = stream.fps
    if options.codec == pyneat.RtspCodec.H264:
        options.fallback_h264_width = stream.width
        options.fallback_h264_height = stream.height
    return options


def decoder_graph(config: Config, stream: RtspStreamInfo) -> pyneat.Graph:
    """Hardware decode to NV12 frames at the stream's size, ending at "detector_frame"."""
    decode = pyneat.SimaDecodeOptions()
    decode.type = config.choice("input.codec", DECODE_TYPES)
    # Names the output buffer "decoder", the default Model::Options::upstream_name.
    decode.decoder_name = "decoder"
    decode.dec_width = stream.width
    decode.dec_height = stream.height
    decode.dec_fps = stream.fps
    graph = pyneat.Graph("decoder")
    graph.add(pyneat.nodes.sima_decode(decode))
    graph.add(
        pyneat.nodes.caps_raw(
            "NV12", stream.width, stream.height, stream.fps, pyneat.CapsMemory.Any
        )
    )
    graph.add(pyneat.nodes.output("detector_frame"))
    return graph


def stream_id(index: int) -> str:
    return f"stream{index}"


def detector_link(index: int) -> pyneat.GraphLinkOptions:
    """Feed the shared model the latest frames of each stream, tagged with the stream's ID."""
    link = pyneat.GraphLinkOptions()
    link.policy = pyneat.GraphLinkPolicy.RealtimeLatestByStream
    link.stream_id = stream_id(index)
    link.max_inflight_total = MAX_INFLIGHT_TOTAL
    return link


def video_link() -> pyneat.GraphLinkOptions:
    """Send the newest encoded frame, so a slow video sender cannot hold back the decoder."""
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


def run_options() -> pyneat.RunOptions:
    options = pyneat.RunOptions()
    options.preset = pyneat.RunPreset.Realtime
    # Auto may copy an output when the sink queue backs up (Core RunCoreGraphStart.cpp).
    options.output_memory = pyneat.OutputMemory.ZeroCopy
    return options


def tracker_settings(config: Config) -> tracker.TrackerSettings:
    return tracker.TrackerSettings(
        high_score_threshold=config.number("app.tracking.high_score_threshold"),
        new_track_threshold=config.number("app.tracking.new_track_threshold"),
        max_missing_frames=config.integer("app.tracking.max_missing_frames"),
        min_confirmed_hits=config.integer("app.tracking.min_confirmed_hits"),
    )


def stream_info(config: Config, sources: list[str]) -> list[RtspStreamInfo]:
    """Each stream's frame size and rate: the configured values, and a probe's for each 0.

    Every stream has the same size, because the streams share one model input.
    """
    # Behaves as probe_rtsp_streams, which #793 adds to sima_examples.rtsp_probe. Whichever of
    # #785 and #793 merges second calls that instead.
    configured = RtspStreamInfo(
        config.integer("input.width"),
        config.integer("input.height"),
        config.integer("input.fps"),
    )
    tcp = config.boolean("input.rtsp.tcp")
    by_url = {url: probe_rtsp_stream_info(url, tcp, configured) for url in dict.fromkeys(sources)}
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
    # Core candidate: decode_bbox_tensor takes a Tensor, so each example picks the BBOX tensor
    # out of the Sample itself.
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


def send_tracks(
    run: pyneat.Run,
    metadata_senders: list[pyneat.MetadataSender],
    trackers: list[tracker.ObjectTracker],
    streams: list[RtspStreamInfo],
    max_detections: int,
    frames: int,
    stop: threading.Event,
) -> list[int]:
    """Track every frame's boxes and send the tracks to its stream's Insight channel until each
    stream sent `frames` (0: no limit) or a stop signal; return each stream's count."""
    processed = [0] * len(metadata_senders)
    while not stop.is_set() and (frames == 0 or min(processed) < frames):
        sample = run.pull("detections", PULL_TIMEOUT_MS)
        if sample is None:
            if run.last_error() or not run.running():
                raise RuntimeError(f"detections output closed: {run.last_error()}")
            continue

        # With one source, Core names the sample after the decoder instead of the stream.
        index = 0
        if len(metadata_senders) > 1:
            index = int(sample.stream_id.removeprefix("stream"))
        if frames and processed[index] >= frames:
            continue
        stream = streams[index]
        tracks = trackers[index].update(
            decode_boxes(sample, stream, max_detections), sample.pts_ns
        )
        payload = detection_egress.serialize_tracks(
            tracks,
            LABELS,
            stream.width,
            stream.height,
            detection_egress.frame_metadata(sample, index),
        )
        # A nonblocking send drops the message while the socket buffer is full; the drops are
        # counted in the sender's stats and reported when the run ends. Any other failure raises.
        metadata_senders[index].send_raw_json(payload)
        processed[index] += 1
    return processed


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = Config.load(args.config)
    sources = config.string_list("input.source")
    streams = stream_info(config, sources)
    # Used by both the model and the box decode.
    max_detections = config.integer("inference.max_detections")
    model_path = args.config.absolute().parent / config.string("model.path")
    model = pyneat.Model(str(model_path), model_options(config, streams[0], max_detections))
    detector = detector_graph(model)

    graph = pyneat.Graph()
    metadata_senders = []
    trackers = []
    for index, url in enumerate(sources):
        stream = streams[index]
        rtsp = pyneat.groups.rtsp_encoded_input(rtsp_options(config, url, stream))
        decoder = decoder_graph(config, stream)
        # A plain link keeps every encoded frame. A dropped one corrupts the decoded frames
        # after it until the next keyframe.
        graph.connect(rtsp, decoder)
        graph.connect(decoder, detector, detector_link(index))
        graph.connect(rtsp, pyneat.groups.video_sender(video_options(config, index)), video_link())
        metadata_senders.append(pyneat.MetadataSender(metadata_options(config, index)))
        trackers.append(tracker.ObjectTracker(tracker_settings(config), stream.fps))
        print(
            f"[stream {index}] rtsp={url} stream={stream.width}x{stream.height}@{stream.fps} "
            f"insight={config.string('output.insight.host')} "
            f"video={config.integer('output.insight.video_port_base') + index} "
            f"metadata={metadata_senders[index].metadata_port()}"
        )

    stop = stop_on_signals()
    run = graph.build(run_options())
    try:
        processed = send_tracks(
            run,
            metadata_senders,
            trackers,
            streams,
            max_detections,
            config.integer("inference.frames"),
            stop,
        )
    finally:
        run.close()

    for index, sender in enumerate(metadata_senders):
        print(
            f"[stream {index}] processed={processed[index]} "
            f"tracks_started={trackers[index].tracks_started()}"
        )
        if sender.stats().send_failures:
            print(f"[stream {index}] {sender.stats().send_failures} metadata messages dropped")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:  # noqa: BLE001 - report the cause in one line, not a traceback
        print(f"[ERR] {error}", file=sys.stderr)
        sys.exit(1)
