"""Run a different YOLO model on each of up to four RTSP streams, shown in Insight.

Each stream's model detects objects, segments them, or estimates poses. Insight draws the results
over that stream's video, one channel per stream.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from dataclasses import dataclass
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

from sima_examples.config import Config, lookup
from sima_examples.labels import load_labels
from sima_examples.rtsp_probe import RtspStreamInfo, probe_rtsp_stream_info
from sima_examples.stop_signals import stop_on_signals
from utils import insight_metadata
from utils.pending_results import PendingResults

RTSP_CODECS = {"h264": pyneat.RtspCodec.H264, "h265": pyneat.RtspCodec.H265}
DECODE_TYPES = {"h264": pyneat.SimaDecodeType.H264, "h265": pyneat.SimaDecodeType.H265}
# Insight metadata types: Insight draws boxes, mask outlines, or skeletons for each.
DETECTION = "object-detection"
SEGMENTATION = "segmentation"
POSE = "pose-estimation"
# Each app.model_ids value: the BoxDecode type of the package's heads, and the Insight
# metadata type of its results. YOLO11 packages carry the YOLOv8 heads.
MODEL_IDS = {
    "yolov8": (pyneat.BoxDecodeType.YoloV8, DETECTION),
    "yolov8-seg": (pyneat.BoxDecodeType.YoloV8Seg, SEGMENTATION),
    "yolov8-pose": (pyneat.BoxDecodeType.YoloV8Pose, POSE),
    "yolo26": (pyneat.BoxDecodeType.YoloV26, DETECTION),
    "yolo26-seg": (pyneat.BoxDecodeType.YoloV26Seg, SEGMENTATION),
    "yolo26-pose": (pyneat.BoxDecodeType.YoloV26Pose, POSE),
}
# The key that holds a message's entries, for each metadata type.
DATA_KEYS = {DETECTION: "objects", SEGMENTATION: "segments", POSE: "poses"}
# How long a stream's decoded frames may arrive with no result before the app stops with an
# error. Not measured on a DevKit: a result takes a fraction of a second.
RESULT_DEADLINE_S = 10.0
# The same for the first result, which also waits for Core to build the rest of the pipeline on
# the first frame (EdgeRouter.cpp). Not measured on a DevKit.
FIRST_RESULT_DEADLINE_S = 30.0
# Decoded frames each stream's decoder keeps. 12 as before the conversion, not re-measured with
# this graph.
DECODER_BUFFERS = 12
# The pull loop polls every output without waiting, and sleeps this long after a round that
# pulled no result.
IDLE_SLEEP_S = 0.002


@dataclass
class Stream:
    """What the pull loop keeps for one stream."""

    info: RtspStreamInfo
    metadata_type: str
    metadata_sender: pyneat.MetadataSender
    pending: PendingResults
    processed: int = 0
    left_out: int = 0


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args(argv)


def model_options(
    config: Config,
    stream: RtspStreamInfo,
    decode_type: pyneat.BoxDecodeType,
    max_detections: int,
) -> pyneat.ModelOptions:
    """A YOLO model on the decoded NV12 frames of one stream, decoded by `decode_type`."""
    options = pyneat.ModelOptions()
    options.preprocess.kind = pyneat.InputKind.Image
    options.preprocess.enable = pyneat.AutoFlag.On
    options.preprocess.color_convert.input_format = pyneat.PreprocessColorFormat.NV12
    # Lets streams above Core's 1920x1080 default in (PreprocessPlan.h).
    options.preprocess.input_max_width = stream.width
    options.preprocess.input_max_height = stream.height
    options.preprocess.preset = pyneat.NormalizePreset.COCO_YOLO
    options.decode_type = decode_type
    # The pose heads score one class, person, while the packages declare the 80 COCO classes.
    if decode_type in (pyneat.BoxDecodeType.YoloV8Pose, pyneat.BoxDecodeType.YoloV26Pose):
        options.num_classes = 1
    options.score_threshold = config.number("inference.min_score")
    options.nms_iou_threshold = config.number("inference.nms_iou")
    options.top_k = max_detections
    return options


def results_output(index: int) -> str:
    return f"results_{index}"


def frames_output(index: int) -> str:
    return f"frames_{index}"


def model_graph(model: pyneat.Model, index: int) -> pyneat.Graph:
    """One stream's model: "model_frame" input -> model -> "results_<index>" output."""
    frames = pyneat.Graph()
    frames.add(pyneat.nodes.input("model_frame", model.input_appsrc_options(False)))

    results = pyneat.Graph()
    results.add(pyneat.nodes.output(results_output(index)))

    stages = model.graph()
    graph = pyneat.Graph()
    graph.connect(frames, stages)
    graph.connect(stages, results)
    return graph


def rtsp_options(
    config: Config, url: str, stream: RtspStreamInfo
) -> pyneat.RtspEncodedInputOptions:
    options = pyneat.RtspEncodedInputOptions()
    options.url = url
    options.codec = config.choice("input.codec", RTSP_CODECS)
    options.tcp = config.boolean("input.rtsp.tcp")
    options.latency_ms = config.integer("input.rtsp.latency_ms")
    # 0 accepts every RTP payload type. The default accepts only 96, and a camera may send
    # another dynamic type.
    options.payload_type = 0
    options.source_fps = config.integer("input.fps")
    if options.codec == pyneat.RtspCodec.H264:
        options.fallback_h264_width = stream.width
        options.fallback_h264_height = stream.height
        if options.source_fps > 0:
            # H.264 caps from input.fps and the stream's size, instead of the caps the stream sends.
            options.auto_caps_from_stream = False
        else:
            # The rate the stream's caps fall back to; with input.fps set it has no effect.
            options.fallback_h264_fps = stream.fps
    return options


def decoder_graph(config: Config, stream: RtspStreamInfo) -> pyneat.Graph:
    """Hardware decode to NV12 frames at the stream's size, ending at "model_frame"."""
    decode = pyneat.SimaDecodeOptions()
    decode.type = config.choice("input.codec", DECODE_TYPES)
    # Names the output buffer "decoder", the default Model::Options::upstream_name.
    decode.decoder_name = "decoder"
    decode.dec_width = stream.width
    decode.dec_height = stream.height
    decode.dec_fps = stream.fps
    decode.num_buffers = DECODER_BUFFERS
    graph = pyneat.Graph("decoder")
    graph.add(pyneat.nodes.sima_decode(decode))
    graph.add(
        pyneat.nodes.caps_raw("NV12", stream.width, stream.height, config.integer("input.fps"))
    )
    graph.add(pyneat.nodes.output("model_frame"))
    return graph


def frames_graph(index: int) -> pyneat.Graph:
    """The newest decoded frame of one stream, which tells the pull loop the decoder is sending."""
    # Core gap: a Run reports no count of the frames a model holds, so the pull loop watches
    # each decoder's frames to tell a stalled model from a paused camera.
    graph = pyneat.Graph()
    graph.add(pyneat.nodes.output(frames_output(index), pyneat.OutputOptions.latest()))
    return graph


def realtime_link() -> pyneat.GraphLinkOptions:
    """Pass on the newest frame, so a slow consumer cannot hold back the stream.

    Core caps the decoded frames in flight on one link at min(4 * streams, 8), which is 4 for the
    one stream per model (EdgeRouter.cpp). Not verified on a DevKit: the frames output receives
    the same decoded frames, and Core counts one credit per frame, so the model may hold more.
    """
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
    # Auto copies a result that waits at an output while another one is queued there, unless it
    # is DMA-BUF or framework memory (RunCoreGraphStart.cpp, GraphRuntimeUtil.cpp).
    options.output_memory = pyneat.OutputMemory.ZeroCopy
    return options


def stream_info(config: Config, sources: list[str]) -> list[RtspStreamInfo]:
    """Each stream's frame size and rate: the configured values, and a probe's for each 0."""
    configured = RtspStreamInfo(
        config.integer("input.width"),
        config.integer("input.height"),
        config.integer("input.fps"),
    )
    tcp = config.boolean("input.rtsp.tcp")
    by_url = {url: probe_rtsp_stream_info(url, tcp, configured) for url in dict.fromkeys(sources)}
    return [by_url[url] for url in sources]


def per_stream(config: Config, key: str, streams: int) -> list[str]:
    """The list at `key`, which holds one entry per input.source stream."""
    values = config.string_list(key)
    if len(values) != streams:
        raise ValueError(
            f"{key} has {len(values)} entries and input.source has {streams}: give one per stream"
        )
    return values


def sample_tensors(sample: pyneat.Sample) -> list[pyneat.Tensor]:
    # Core candidate: Python has no binding of tensors_from_sample (GraphOptions.h), so both
    # languages pick the tensors out of the Sample here.
    if sample.kind == pyneat.SampleKind.TensorSet and sample.tensors:
        return list(sample.tensors)
    if sample.kind == pyneat.SampleKind.Tensor and sample.tensor is not None:
        return [sample.tensor]
    raise RuntimeError("expected a tensor result")


def metadata_entries(
    sample: pyneat.Sample, stream: Stream, labels: list[str], max_detections: int
) -> list[dict]:
    """The objects, segments, or poses of one results sample, in the stream's frame pixels."""
    width, height = stream.info.width, stream.info.height
    tensors = sample_tensors(sample)
    if stream.metadata_type == DETECTION:
        result = pyneat.detections.decode_bbox_tensor(
            tensors[0], width, height, max_detections, strict=True
        )
        return insight_metadata.objects(result.boxes, labels, width, height)
    # A frame with nothing found decodes to tensors with zero rows.
    entries = []
    if stream.metadata_type == SEGMENTATION:
        for item in pyneat.decode_segmentation(
            tensors, clamp_to=(width, height), top_k=max_detections, strict=True
        ):
            entries += insight_metadata.segments(
                item.boxes.to_numpy(), item.masks.to_numpy(), labels, width, height
            )
        return entries
    for item in pyneat.decode_pose(
        tensors, clamp_to=(width, height), top_k=max_detections, strict=True
    ):
        entries += insight_metadata.poses(item.boxes.to_numpy(), item.keypoints.to_numpy())
    return entries


def send_results(
    run: pyneat.Run,
    streams: list[Stream],
    labels: list[str],
    max_detections: int,
    frames: int,
    stop: threading.Event,
) -> None:
    """Send every stream's results to its Insight channel until each stream sent `frames`
    (0 or less: no limit) or a stop signal. A stream whose decoded frames arrive with no result
    for its deadline raises with its model."""
    while not stop.is_set() and (frames <= 0 or min(s.processed for s in streams) < frames):
        pulled = False
        now_s = time.monotonic()
        for index, stream in enumerate(streams):
            if frames > 0 and stream.processed >= frames:
                continue
            sample = run.pull(results_output(index), 0)
            # Core gap: Python's pull returns None both on a timeout and on a closed output, so
            # an output that closed shows only once the run stopped. A source that ends cleanly
            # leaves the run running, and the loop waits for it. C++ reads the PullStatus.
            # Core candidate: a Python pull that returns the PullStatus (RunPull.cpp).
            if sample is None and (run.last_error() or not run.running()):
                error = run.last_error()
                raise RuntimeError(
                    f"{results_output(index)} output closed unexpectedly"
                    + (f": {error}" if error else "")
                )
            frame_arrived = run.pull(frames_output(index), 0) is not None
            stream.pending.observe(frame_arrived, sample is not None, now_s)
            if sample is None:
                continue
            pulled = True
            data, left_out = insight_metadata.data_json(
                DATA_KEYS[stream.metadata_type],
                metadata_entries(sample, stream, labels, max_detections),
            )
            # A nonblocking send drops the message while the socket buffer is full; the drops
            # are counted in the sender's stats and reported when the run ends. Any other
            # failure raises.
            stream.metadata_sender.send_metadata(
                stream.metadata_type,
                data,
                sample.pts_ns // 1_000_000 if sample.pts_ns >= 0 else -1,
                str(sample.frame_id) if sample.frame_id >= 0 else "",
            )
            stream.processed += 1
            stream.left_out += left_out
        if not pulled:
            time.sleep(IDLE_SLEEP_S)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = Config.load(args.config)
    sources = config.string_list("input.source")
    model_paths = per_stream(config, "app.model_paths", len(sources))
    model_ids = per_stream(config, "app.model_ids", len(sources))
    # Each stream's BoxDecode type and metadata type, read before the probe and any model load.
    model_types = [lookup("app.model_ids", model_id, MODEL_IDS) for model_id in model_ids]
    infos = stream_info(config, sources)
    labels = load_labels(Path(config.string("model.labels")))
    # Used by both the models and the box decode.
    max_detections = config.integer("inference.max_detections")
    # Results each stream sends before the app stops. 0 or less runs until stopped.
    frames = config.integer("inference.frames")

    graph = pyneat.Graph()
    models = []
    streams = []
    for index, url in enumerate(sources):
        info = infos[index]
        decode_type, metadata_type = model_types[index]
        model_path = args.config.absolute().parent / model_paths[index]
        models.append(
            pyneat.Model(str(model_path), model_options(config, info, decode_type, max_detections))
        )
        rtsp = pyneat.groups.rtsp_encoded_input(rtsp_options(config, url, info))
        decoder = decoder_graph(config, info)
        # A plain link keeps every encoded frame. A dropped one corrupts the decoded frames
        # after it until the next keyframe. Core runs an RTSP input with exactly two consumers,
        # a decoder on a plain link and a synchronous VideoSender of the same codec, as one
        # pipeline (ExecutionGraphCompiler.cpp), so the frames output taps the decoder instead.
        graph.connect(rtsp, decoder)
        graph.connect(decoder, model_graph(models[-1], index), realtime_link())
        graph.connect(
            rtsp, pyneat.groups.video_sender(video_options(config, index)), realtime_link()
        )
        graph.connect(decoder, frames_graph(index), realtime_link())
        streams.append(
            Stream(
                info,
                metadata_type,
                pyneat.MetadataSender(metadata_options(config, index)),
                PendingResults(
                    index, model_paths[index], RESULT_DEADLINE_S, FIRST_RESULT_DEADLINE_S
                ),
            )
        )
        print(
            f"[stream {index}] {model_ids[index]} model={model_paths[index]} rtsp={url} "
            f"stream={info.width}x{info.height}@{info.fps} "
            f"insight={config.string('output.insight.host')} "
            f"video={config.integer('output.insight.video_port_base') + index} "
            f"metadata={streams[-1].metadata_sender.metadata_port()}"
        )

    stop = stop_on_signals()
    run = graph.build(run_options())
    try:
        send_results(run, streams, labels, max_detections, frames, stop)
    finally:
        run.close()

    for index, stream in enumerate(streams):
        print(f"[stream {index}] processed={stream.processed} left_out={stream.left_out}")
        if stream.metadata_sender.stats().send_failures:
            print(
                f"[stream {index}] {stream.metadata_sender.stats().send_failures} "
                "metadata messages dropped"
            )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:  # noqa: BLE001 - report the cause in one line, not a traceback
        print(f"[ERR] {error}", file=sys.stderr)
        sys.exit(1)
