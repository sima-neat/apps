"""Detect faces and their five landmarks in one thermal RTSP stream with yolov5s-face, shown in
Insight.

The decoded frames branch to Insight's video and to the model. The model's six raw heads are
decoded on the host, and Insight receives every frame's faces as a pose overlay.
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

from sima_examples.config import Config
from sima_examples.labels import load_labels
from sima_examples.rtsp_probe import RtspStreamInfo, probe_rtsp_stream_info
from sima_examples.stop_signals import stop_on_signals
from utils import yolov5_face

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "common" / "config.yaml"
PULL_TIMEOUT_MS = 100


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args(argv)


def model_options() -> pyneat.ModelOptions:
    """yolov5s-face on NV12 frames: letterbox to the 800x800 canvas and scale to 0-1 on the
    device. No box decode: the model outputs its raw heads."""
    options = pyneat.ModelOptions()
    options.preprocess.kind = pyneat.InputKind.Image
    options.preprocess.enable = pyneat.AutoFlag.On
    options.preprocess.color_convert.input_format = pyneat.PreprocessColorFormat.NV12
    options.preprocess.preset = pyneat.NormalizePreset.COCO_YOLO
    return options


def rtsp_options(config: Config, stream: RtspStreamInfo) -> pyneat.RtspDecodedInputOptions:
    """Receive the H.264 stream and decode it to NV12 frames of the stream's size and rate."""
    options = pyneat.RtspDecodedInputOptions()
    options.url = config.string("input.source")
    options.tcp = config.boolean("input.rtsp.tcp")
    options.latency_ms = config.integer("input.rtsp.latency_ms")
    options.fallback_h264_width = stream.width
    options.fallback_h264_height = stream.height
    options.fallback_h264_fps = stream.fps
    options.output_caps.enable = True
    options.output_caps.width = stream.width
    options.output_caps.height = stream.height
    options.output_caps.fps = stream.fps
    return options


def video_graph(config: Config, stream: RtspStreamInfo) -> pyneat.Graph:
    """The "video" branch: decoded frames encoded as H.264 for Insight."""
    encode = pyneat.SimaEncodeOptions()
    encode.fps = stream.fps
    options = pyneat.VideoSenderOptions.from_raw(encode)
    options.host = config.string("output.insight.host")
    options.video_port_base = config.integer("output.insight.video_port_base")
    graph = pyneat.Graph("video")
    graph.connect(pyneat.nodes.input("video"), pyneat.groups.video_sender(options))
    return graph


def model_graph(model: pyneat.Model) -> pyneat.Graph:
    """The "model" branch: decoded frames into the model."""
    graph = pyneat.Graph("model")
    graph.connect(pyneat.nodes.input("model"), model)
    return graph


def detections_graph() -> pyneat.Graph:
    """The "detections" output, holding up to 4 results for the pull loop."""
    graph = pyneat.Graph("detections")
    graph.add(pyneat.nodes.output("detections", pyneat.OutputOptions.every_frame(4)))
    return graph


def graph_options() -> pyneat.GraphOptions:
    """Run the model's postprocessing adapters, such as dequantize, on the EV74 vision
    processor, which leaves the host CPU to the face decode."""
    options = pyneat.GraphOptions()
    options.advanced_execution.postprocess_target = "EV74"
    return options


def live_link() -> pyneat.GraphLinkOptions:
    """Pass on only the newest frame, so a slow branch cannot hold back the stream."""
    link = pyneat.GraphLinkOptions()
    link.policy = pyneat.GraphLinkPolicy.RealtimeLatestByStream
    return link


def metadata_options(config: Config) -> pyneat.MetadataSenderOptions:
    options = pyneat.MetadataSenderOptions()
    options.host = config.string("output.insight.host")
    options.metadata_port_base = config.integer("output.insight.metadata_port_base")
    return options


def run_options() -> pyneat.RunOptions:
    options = pyneat.RunOptions()
    options.preset = pyneat.RunPreset.Realtime
    # One more queued frame than the Realtime preset's 2.
    options.queue_depth = 3
    options.output_memory = pyneat.OutputMemory.ZeroCopy
    return options


def split_heads(sample: pyneat.Sample) -> list:
    """The model's output tensors as numpy views, from a Tensor, TensorSet, or Bundle sample."""
    # Core candidate: tensors_from_sample is not bound in Python, so the example walks the
    # sample itself.
    if sample.kind == pyneat.SampleKind.Tensor:
        return [sample.tensor.to_numpy(copy=False)]
    if sample.kind == pyneat.SampleKind.TensorSet:
        return [tensor.to_numpy(copy=False) for tensor in sample.tensors]
    return [head for field in sample.fields for head in split_heads(field)]


def send_faces(
    run: pyneat.Run,
    metadata_sender: pyneat.MetadataSender,
    config: Config,
    label: str,
    stream: RtspStreamInfo,
    frames: int,
    stop: threading.Event,
) -> int:
    """Send every frame's faces to Insight until `frames` (0: no limit) or a stop signal; return
    the count."""
    min_score = config.number("inference.min_score")
    nms_iou = config.number("inference.nms_iou")
    max_detections = config.integer("inference.max_detections")
    letterbox = yolov5_face.letterbox(stream.width, stream.height)
    processed = 0
    while not stop.is_set() and (frames == 0 or processed < frames):
        sample = run.pull("detections", PULL_TIMEOUT_MS)
        if sample is None:
            if run.last_error() or not run.running():
                raise RuntimeError(f"detections output closed: {run.last_error()}")
            continue
        faces = yolov5_face.decode_faces(split_heads(sample), min_score, nms_iou, max_detections)
        faces = yolov5_face.to_frame(faces, letterbox, stream.width, stream.height)
        # The frame's presentation time lets Insight draw the faces on the frame they came from.
        timestamp_ms = sample.pts_ns // 1_000_000 if sample.pts_ns >= 0 else -1
        # A nonblocking send drops the message while the socket buffer is full; the drops are
        # counted in the sender's stats and reported when the run ends. Any other failure raises.
        metadata_sender.send_metadata(
            "pose-estimation",
            yolov5_face.poses_json(faces, label),
            timestamp_ms,
            str(max(sample.frame_id, 0)),
        )
        processed += 1
    return processed


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = Config.load(args.config)
    configured = RtspStreamInfo(
        config.integer("input.width"), config.integer("input.height"), config.integer("input.fps")
    )
    # The stream's size and rate come from the probe for each value the config leaves at 0.
    stream = probe_rtsp_stream_info(
        config.string("input.source"), config.boolean("input.rtsp.tcp"), configured
    )
    # The label file names the one class, face.
    label = load_labels(Path(config.string("model.labels")))[0]
    model_path = args.config.absolute().parent / config.string("model.path")
    model = pyneat.Model(str(model_path), model_options())

    source = pyneat.groups.rtsp_decoded_input(rtsp_options(config, stream))
    branch = pyneat.graphs.branch("source", ["video", "model"])
    video = video_graph(config, stream)
    detector = model_graph(model)

    graph = pyneat.Graph(graph_options())
    graph.connect(source, branch)
    graph.connect(branch, video, live_link())
    graph.connect(branch, detector, live_link())
    graph.connect(detector, detections_graph())

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
    try:
        processed = send_faces(
            run, metadata_sender, config, label, stream, config.integer("inference.frames"), stop
        )
    finally:
        run.close()

    print(f"processed={processed} video_sender={host}:{video_port}")
    if metadata_sender.stats().send_failures:
        print(f"{metadata_sender.stats().send_failures} metadata messages dropped")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:  # noqa: BLE001 - report the cause in one line, not a traceback
        print(f"[ERR] {error}", file=sys.stderr)
        sys.exit(1)
