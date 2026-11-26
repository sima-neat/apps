"""Detect oriented boxes on several aerial RTSP streams with one shared YOLO OBB model.

Every stream's video and its rotated boxes go to Insight, one channel per stream.
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

from sima_examples.config import Config
from sima_examples.labels import load_labels
from sima_examples.rtsp_probe import RtspStreamInfo, probe_rtsp_streams
from sima_examples.stop_signals import stop_on_signals
from utils import oriented_boxes

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
# YOLOv8 and YOLO11 share one oriented-box decoder.
BOX_DECODE_TYPES = {
    "yolo26": pyneat.BoxDecodeType.YoloV26Obb,
    "yolov8": pyneat.BoxDecodeType.YoloV8Obb,
    "yolo11": pyneat.BoxDecodeType.YoloV8Obb,
}
# Ultralytics' IoU for rotated NMS on OBB predictions.
YOLOV8_OBB_NMS_IOU = 0.7
PULL_TIMEOUT_MS = 100


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args(argv)


def model_options(config: Config) -> pyneat.ModelOptions:
    options = pyneat.ModelOptions()
    options.preprocess.kind = pyneat.InputKind.Image
    options.preprocess.enable = pyneat.AutoFlag.On
    options.preprocess.color_convert.input_format = pyneat.PreprocessColorFormat.NV12
    # COCO_YOLO letterboxes, so rotated boxes map back when both axes scale by the same factor.
    options.preprocess.preset = pyneat.NormalizePreset.COCO_YOLO
    options.decode_type = config.choice("model.id", BOX_DECODE_TYPES)
    # YOLO26 OBB runs no NMS and rejects a nonzero threshold. YOLOv8 and YOLO11 OBB need one.
    if options.decode_type == pyneat.BoxDecodeType.YoloV8Obb:
        options.nms_iou_threshold = YOLOV8_OBB_NMS_IOU
    options.score_threshold = config.number("inference.min_score")
    options.top_k = config.integer("inference.max_detections")
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
    graph.add(pyneat.nodes.caps_raw("NV12", stream.width, stream.height, stream.fps))
    graph.add(pyneat.nodes.output("detector_frame"))
    return graph


def stream_id(index: int) -> str:
    return f"stream{index}"


def detector_link(index: int, streams: int) -> pyneat.GraphLinkOptions:
    """Feed the shared model the latest frames of each stream, tagged with the stream's ID."""
    link = pyneat.GraphLinkOptions()
    link.policy = pyneat.GraphLinkPolicy.RealtimeLatestByStream
    link.stream_id = stream_id(index)
    # Core's default total cap is 8, so above two streams each would get fewer than its 4 frames.
    link.max_inflight_total = 4 * streams
    return link


def video_link() -> pyneat.GraphLinkOptions:
    link = pyneat.GraphLinkOptions()
    link.policy = pyneat.GraphLinkPolicy.RealtimeLatestByStream
    return link


def video_options(config: Config, index: int) -> pyneat.VideoSenderOptions:
    """Send the encoded stream to Insight unchanged, on video port base + index."""
    # Passthrough(MJPEG) carries RTP/JPEG: each side a multiple of 8, at most 2040 pixels.
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
    # Auto copies a result that waits in the output queue unless it is in DMA-BUF or framework
    # memory. ZeroCopy never copies.
    options.output_memory = pyneat.OutputMemory.ZeroCopy
    return options


def decode_boxes(sample: pyneat.Sample) -> pyneat.OrientedDecodeTensors:
    """The oriented boxes in a detections sample, already in source-frame pixels."""
    # Core candidate: decode_obb_tensor takes a Tensor, so each example picks the tensor out of
    # the Sample itself.
    if sample.kind == pyneat.SampleKind.TensorSet and sample.tensors:
        tensor = sample.tensors[0]
    elif sample.kind == pyneat.SampleKind.Tensor:
        tensor = sample.tensor
    else:
        raise RuntimeError("expected a BBOX_OBB_V1 tensor in the detections sample")
    return pyneat.decode_obb_tensor(tensor)


def send_detections(
    run: pyneat.Run,
    metadata_senders: list[pyneat.MetadataSender],
    labels: list[str],
    frames: int,
    stop: threading.Event,
) -> list[int]:
    """Send every frame's boxes to its stream's Insight channel until each stream sent `frames`
    (0: no limit) or a stop signal; return each stream's count."""
    processed = [0] * len(metadata_senders)
    while not stop.is_set() and (frames == 0 or min(processed) < frames):
        sample = run.pull("detections", PULL_TIMEOUT_MS)
        if sample is None:
            if run.last_error() or not run.running():
                raise RuntimeError(f"detections output closed: {run.last_error()}")
            continue

        # Core candidate: with one source, Core names the sample after the decoder instead of the
        # link's stream_id, so the index is 0 here.
        index = 0
        if len(metadata_senders) > 1:
            index = int(sample.stream_id.removeprefix("stream"))
        if frames and processed[index] >= frames:
            continue
        data = oriented_boxes.detections_json(decode_boxes(sample), labels)
        timestamp_ms = sample.pts_ns // 1_000_000 if sample.pts_ns >= 0 else -1
        frame_id = str(sample.frame_id) if sample.frame_id >= 0 else ""
        # A nonblocking send drops the message while the socket buffer is full; the drops are
        # counted in the sender's stats and reported when the run ends. Any other failure raises.
        metadata_senders[index].send_metadata("object-detection", data, timestamp_ms, frame_id)
        processed[index] += 1
    return processed


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = Config.load(args.config)
    sources = config.string_list("input.source")
    # Each stream's frame size and rate: the configured values, and a probe's for each 0.
    streams = probe_rtsp_streams(
        sources,
        config.boolean("input.rtsp.tcp"),
        RtspStreamInfo(
            config.integer("input.width"),
            config.integer("input.height"),
            config.integer("input.fps"),
        ),
    )
    labels = load_labels(Path(config.string("model.labels")))
    model_path = args.config.absolute().parent / config.string("model.path")
    model = pyneat.Model(str(model_path), model_options(config))
    detector = detector_graph(model)

    graph = pyneat.Graph()
    metadata_senders = []
    for index, url in enumerate(sources):
        stream = streams[index]
        rtsp = pyneat.groups.rtsp_encoded_input(rtsp_options(config, url, stream))
        decoder = decoder_graph(config, stream)
        # A plain link keeps every encoded frame. A dropped one corrupts the decoded frames
        # after it until the next keyframe.
        graph.connect(rtsp, decoder)
        graph.connect(decoder, detector, detector_link(index, len(sources)))
        graph.connect(rtsp, pyneat.groups.video_sender(video_options(config, index)), video_link())
        metadata_senders.append(pyneat.MetadataSender(metadata_options(config, index)))
        print(
            f"[stream {index}] rtsp={url} stream={stream.width}x{stream.height}@{stream.fps} "
            f"insight={config.string('output.insight.host')} "
            f"video={config.integer('output.insight.video_port_base') + index} "
            f"metadata={metadata_senders[index].metadata_port()}"
        )

    stop = stop_on_signals()
    run = graph.build(run_options())
    try:
        processed = send_detections(
            run, metadata_senders, labels, config.integer("inference.frames"), stop
        )
    finally:
        run.close()

    # Core candidate: Core forwards fd 1 through a detached thread (start_stdio_filter) that the
    # process exit can stop before it writes the last lines. flush=True puts each line into the
    # pipe earlier, which narrows the race but does not close it.
    for index, sender in enumerate(metadata_senders):
        print(f"[stream {index}] processed={processed[index]}", flush=True)
        if sender.stats().send_failures:
            print(
                f"[stream {index}] {sender.stats().send_failures} metadata messages dropped",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:  # noqa: BLE001 - report the cause in one line, not a traceback
        print(f"[ERR] {error}", file=sys.stderr)
        sys.exit(1)
