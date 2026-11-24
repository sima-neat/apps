"""Track tiny drones in one RTSP stream with YOLO26, shown in Insight.

The decoded frames feed the model. The tracker gives each drone an ID, and Insight draws every
frame's tracks over the video.
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
PULL_TIMEOUT_MS = 100


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args(argv)


def model_options(
    config: Config, stream: RtspStreamInfo, max_detections: int
) -> pyneat.ModelOptions:
    """YOLO26 on the decoded NV12 frames."""
    options = pyneat.ModelOptions()
    options.preprocess.kind = pyneat.InputKind.Image
    options.preprocess.enable = pyneat.AutoFlag.On
    options.preprocess.color_convert.input_format = pyneat.PreprocessColorFormat.NV12
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


def rtsp_options(config: Config, stream: RtspStreamInfo) -> pyneat.RtspEncodedInputOptions:
    options = pyneat.RtspEncodedInputOptions()
    options.url = config.string("input.source")
    options.codec = config.choice("input.codec", RTSP_CODECS)
    options.tcp = config.boolean("input.rtsp.tcp")
    options.latency_ms = config.integer("input.rtsp.latency_ms")
    # With input.fps at 0, Core also takes the caps from the stream: above 60 fps, caps set from
    # the probed values alone failed to negotiate in some runs.
    options.auto_caps_from_stream = config.integer("input.fps") == 0
    options.source_fps = stream.fps
    if options.codec == pyneat.RtspCodec.H264:
        options.fallback_h264_width = stream.width
        options.fallback_h264_height = stream.height
    return options


def decoder_graph(config: Config, stream: RtspStreamInfo) -> pyneat.Graph:
    """Hardware decode of the RTSP stream to NV12 frames."""
    decode = pyneat.SimaDecodeOptions()
    decode.type = config.choice("input.codec", DECODE_TYPES)
    decode.dec_width = stream.width
    decode.dec_height = stream.height
    decode.dec_fps = stream.fps
    graph = pyneat.Graph("decoder")
    graph.add(pyneat.nodes.sima_decode(decode))
    return graph


def detector_graph(model: pyneat.Model) -> pyneat.Graph:
    """Decoded frames -> model -> "detections" output."""
    graph = pyneat.Graph("detector")
    graph.add(model.graph())
    graph.add(pyneat.nodes.output("detections"))
    return graph


def video_options(config: Config) -> pyneat.VideoSenderOptions:
    """Send the encoded stream to Insight unchanged."""
    options = pyneat.VideoSenderOptions.passthrough(config.choice("input.codec", RTSP_CODECS))
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


def tracker_settings(config: Config) -> tracker.TrackerSettings:
    return tracker.TrackerSettings(
        high_score_threshold=config.number("app.tracking.high_score_threshold"),
        new_track_threshold=config.number("app.tracking.new_track_threshold"),
        max_missing_frames=config.integer("app.tracking.max_missing_frames"),
        min_confirmed_hits=config.integer("app.tracking.min_confirmed_hits"),
    )


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
    metadata_sender: pyneat.MetadataSender,
    object_tracker: tracker.ObjectTracker,
    stream: RtspStreamInfo,
    max_detections: int,
    frames: int,
    stop: threading.Event,
) -> int:
    """Track every frame's boxes and send the tracks to Insight until `frames` (0: no limit) or a
    stop signal; return the count."""
    processed = 0
    while not stop.is_set() and (frames == 0 or processed < frames):
        sample = run.pull("detections", PULL_TIMEOUT_MS)
        if sample is None:
            if run.last_error() or not run.running():
                raise RuntimeError(f"detections output closed: {run.last_error()}")
            continue
        tracks = object_tracker.update(
            decode_boxes(sample, stream, max_detections), processed
        )
        payload = detection_egress.serialize_tracks(
            tracks, LABELS, stream.width, stream.height, detection_egress.frame_metadata(sample, 0)
        )
        # A nonblocking send drops the message while the socket buffer is full; the drops are
        # counted in the sender's stats and reported when the run ends. Any other failure raises.
        metadata_sender.send_raw_json(payload)
        processed += 1
    return processed


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
    # Used by both the model and the box decode.
    max_detections = config.integer("inference.max_detections")
    model_path = args.config.absolute().parent / config.string("model.path")
    model = pyneat.Model(str(model_path), model_options(config, stream, max_detections))
    detector = detector_graph(model)

    graph = pyneat.Graph()
    source = pyneat.groups.rtsp_encoded_input(rtsp_options(config, stream))
    decoder = decoder_graph(config, stream)
    graph.connect(source, decoder)
    graph.connect(decoder, detector)
    # A plain link lets Core tee the encoded stream to the decoder and the video sender.
    graph.connect(source, pyneat.groups.video_sender(video_options(config)))

    metadata_sender = pyneat.MetadataSender(metadata_options(config))
    object_tracker = tracker.ObjectTracker(tracker_settings(config))
    stop = stop_on_signals()
    run = graph.build(run_options())
    host = config.string("output.insight.host")
    print(
        f"source={config.string('input.source')} codec={config.string('input.codec')} "
        f"stream={stream.width}x{stream.height}@{stream.fps} insight={host} "
        f"video={config.integer('output.insight.video_port_base')} "
        f"metadata={config.integer('output.insight.metadata_port_base')}"
    )
    try:
        processed = send_tracks(
            run,
            metadata_sender,
            object_tracker,
            stream,
            max_detections,
            config.integer("inference.frames"),
            stop,
        )
    finally:
        run.close()

    print(f"processed={processed} tracks_started={object_tracker.tracks_started()}")
    if metadata_sender.stats().send_failures:
        print(f"{metadata_sender.stats().send_failures} metadata messages dropped")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:  # noqa: BLE001 - report the cause in one line, not a traceback
        print(f"[ERR] {error}", file=sys.stderr)
        sys.exit(1)
