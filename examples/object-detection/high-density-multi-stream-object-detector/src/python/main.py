"""Detect objects on many RTSP streams with one shared YOLO26 model.

Every stream's video and its detection boxes go to Insight, one channel per stream.
"""

from __future__ import annotations

import argparse
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

import pyneat

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "common" / "config.yaml"

# The helpers the examples share live in <apps root>/support/python.
SUPPORT_DIR = next(
    (
        path / "support" / "python"
        for path in Path(__file__).resolve().parents
        if (path / "support" / "python" / "sima_examples").is_dir()
    ),
    None,
)
if SUPPORT_DIR is None:
    raise SystemExit(
        "support/python not found above this example; reinstall the Apps package"
    )
sys.path.insert(0, str(SUPPORT_DIR))

from sima_examples import detection_egress
from sima_examples.config import Config
from sima_examples.labels import load_labels
from sima_examples.rtsp_probe import probe_rtsp_stream_info
from sima_examples.stop_signals import stop_on_signals

RTSP_CODECS = {"h264": pyneat.RtspCodec.H264, "h265": pyneat.RtspCodec.H265}
DECODE_TYPES = {"h264": pyneat.SimaDecodeType.H264, "h265": pyneat.SimaDecodeType.H265}
PULL_TIMEOUT_MS = 100


@dataclass(frozen=True)
class AppConfig:
    model_path: Path
    labels_path: Path
    sources: list[str]
    codec: str
    width: int
    height: int
    fps: int  # 0 reads each stream's frame rate with a probe.
    rtsp_tcp: bool
    rtsp_latency_ms: int
    min_score: float
    nms_iou: float
    max_detections: int
    max_inflight_per_stream: int
    max_inflight_total: int
    video_enabled: bool
    insight_host: str
    video_port_base: int
    metadata_port_base: int
    decoder_buffers: int
    queue_depth: int
    internal_queue_depth: int


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--validate-config-only", action="store_true", help="check the config and exit"
    )
    return parser.parse_args(argv)


def load_app_config(path: Path) -> AppConfig:
    config = Config.load(path)
    app_config = AppConfig(
        model_path=path.resolve().parent
        / Path(config.string("model.path")).expanduser(),
        labels_path=Path(config.string("model.labels")).expanduser(),
        sources=config.string_list("input.source"),
        codec=config.string("input.codec"),
        width=config.integer("input.width"),
        height=config.integer("input.height"),
        fps=config.integer("input.fps"),
        rtsp_tcp=config.boolean("input.rtsp.tcp"),
        rtsp_latency_ms=config.integer("input.rtsp.latency_ms"),
        min_score=config.number("inference.min_score"),
        nms_iou=config.number("inference.nms_iou"),
        max_detections=config.integer("inference.max_detections"),
        max_inflight_per_stream=config.integer("inference.max_inflight_per_stream"),
        max_inflight_total=config.integer("inference.max_inflight_total"),
        video_enabled=config.boolean("output.video_enabled"),
        insight_host=config.string("output.insight.host"),
        video_port_base=config.integer("output.insight.video_port_base"),
        metadata_port_base=config.integer("output.insight.metadata_port_base"),
        decoder_buffers=config.integer("app.decoder_buffers"),
        queue_depth=config.integer("app.queue_depth"),
        internal_queue_depth=config.integer("app.internal_queue_depth"),
    )
    config.reject_unread_keys()
    if app_config.codec not in RTSP_CODECS:
        raise ValueError("input.codec must be h264 or h265")
    return app_config


def model_options(cfg: AppConfig) -> pyneat.ModelOptions:
    options = pyneat.ModelOptions()
    options.verbose = pyneat.VerboseOptions.quiet()
    options.preprocess.kind = pyneat.InputKind.Image
    options.preprocess.enable = pyneat.AutoFlag.On
    options.preprocess.color_convert.input_format = pyneat.PreprocessColorFormat.NV12
    options.preprocess.preset = pyneat.NormalizePreset.COCO_YOLO
    options.decode_type = pyneat.BoxDecodeType.YoloV26
    options.score_threshold = cfg.min_score
    options.nms_iou_threshold = cfg.nms_iou
    options.top_k = cfg.max_detections
    return options


def detector_graph(model: pyneat.Model, cfg: AppConfig) -> pyneat.Graph:
    """The model every stream shares: "detector_frame" input -> model -> "detections" output."""
    input_options = model.input_appsrc_options(False)
    input_options.max_bytes = (
        cfg.width * cfg.height * 3 // 2 * input_options.pool_max_buffers
    )
    frames = pyneat.Graph()
    frames.add(pyneat.nodes.input("detector_frame", input_options))

    detections = pyneat.Graph()
    detections.add(
        pyneat.nodes.output(
            "detections", pyneat.OutputOptions.every_frame(cfg.queue_depth)
        )
    )

    model_graph = model.graph()
    graph = pyneat.Graph()
    graph.connect(frames, model_graph)
    graph.connect(model_graph, detections)
    return graph


def rtsp_options(cfg: AppConfig, url: str, fps: int) -> pyneat.RtspEncodedInputOptions:
    options = pyneat.RtspEncodedInputOptions()
    options.url = url
    options.codec = RTSP_CODECS[cfg.codec]
    options.tcp = cfg.rtsp_tcp
    options.latency_ms = cfg.rtsp_latency_ms
    options.auto_caps_from_stream = cfg.fps == 0
    options.source_fps = cfg.fps
    if cfg.codec == "h264":
        options.fallback_h264_width = cfg.width
        options.fallback_h264_height = cfg.height
        options.fallback_h264_fps = fps
    return options


def decoder_graph(cfg: AppConfig, fps: int) -> pyneat.Graph:
    """Hardware decode to NV12 frames at the stream's size, ending at "detector_frame"."""
    decode = pyneat.SimaDecodeOptions()
    decode.type = DECODE_TYPES[cfg.codec]
    decode.decoder_name = "decoder"
    decode.next_element = "CVU"
    decode.dec_width = cfg.width
    decode.dec_height = cfg.height
    decode.dec_fps = fps
    decode.num_buffers = cfg.decoder_buffers
    decode.input_buffers = 2
    decode.decoder_tuning = "throughput-low-latency"
    decode.memory_opt = True
    graph = pyneat.Graph("decoder")
    graph.add(pyneat.nodes.sima_decode(decode))
    graph.add(
        pyneat.nodes.caps_raw(
            "NV12", cfg.width, cfg.height, cfg.fps, pyneat.CapsMemory.Any
        )
    )
    graph.add(pyneat.nodes.output("detector_frame"))
    return graph


def stream_id(index: int) -> str:
    return f"stream{index}"


def detector_link(cfg: AppConfig, index: int) -> pyneat.GraphLinkOptions:
    """Feed the shared model the latest frame of each stream, with bounded frames in flight."""
    link = pyneat.GraphLinkOptions()
    link.policy = pyneat.GraphLinkPolicy.RealtimeLatestByStream
    link.stream_id = stream_id(index)
    link.max_inflight_per_stream = cfg.max_inflight_per_stream
    link.max_inflight_total = cfg.max_inflight_total
    return link


def video_link() -> pyneat.GraphLinkOptions:
    link = pyneat.GraphLinkOptions()
    link.policy = pyneat.GraphLinkPolicy.RealtimeLatestByStream
    return link


def video_options(cfg: AppConfig, index: int) -> pyneat.VideoSenderOptions:
    """Send the encoded stream to Insight unchanged, on video port base + index."""
    options = pyneat.VideoSenderOptions.passthrough(RTSP_CODECS[cfg.codec])
    options.host = cfg.insight_host
    options.channel = index
    options.video_port_base = cfg.video_port_base
    return options


def metadata_options(cfg: AppConfig, index: int) -> pyneat.MetadataSenderOptions:
    options = pyneat.MetadataSenderOptions()
    options.host = cfg.insight_host
    options.channel = index
    options.metadata_port_base = cfg.metadata_port_base
    return options


def graph_options(cfg: AppConfig) -> pyneat.GraphOptions:
    options = pyneat.GraphOptions()
    options.advanced_execution.internal_queue_depth = cfg.internal_queue_depth
    return options


def run_options(cfg: AppConfig) -> pyneat.RunOptions:
    options = pyneat.RunOptions()
    options.preset = pyneat.RunPreset.Realtime
    options.queue_depth = cfg.queue_depth
    options.output_memory = pyneat.OutputMemory.ZeroCopy
    return options


def stream_fps(cfg: AppConfig) -> dict[str, int]:
    """Each source URL's frame rate: the configured one, or a probe when input.fps is 0."""
    if cfg.fps > 0:
        return {url: cfg.fps for url in cfg.sources}
    return {
        url: probe_rtsp_stream_info(url, cfg.rtsp_tcp).fps for url in set(cfg.sources)
    }


def decode_boxes(sample: pyneat.Sample, cfg: AppConfig) -> list:
    """The boxes in a detections sample, whose first tensor is the BBOX payload."""
    if sample.kind == pyneat.SampleKind.TensorSet and sample.tensors:
        bbox = sample.tensors[0]
    elif sample.kind == pyneat.SampleKind.Tensor:
        bbox = sample.tensor
    else:
        raise RuntimeError(f"expected a BBOX tensor, got {sample.kind}")
    result = pyneat.detections.decode_bbox_tensor(
        bbox, cfg.width, cfg.height, cfg.max_detections, strict=True
    )
    return result.boxes


def frame_metadata(sample: pyneat.Sample, index: int) -> detection_egress.FrameMetadata:
    return detection_egress.FrameMetadata(
        stream_index=index,
        stream_id=sample.stream_id,
        frame_id=sample.frame_id,
        pts_ns=sample.pts_ns,
        dts_ns=sample.dts_ns,
        duration_ns=sample.duration_ns,
        input_seq=sample.input_seq,
        orig_input_seq=sample.orig_input_seq,
        rtp_timestamp=(
            detection_egress.rtp_timestamp_from_pts_ns(sample.pts_ns)
            if sample.pts_ns >= 0
            else None
        ),
    )


def send_detections(
    run: pyneat.Run,
    metadata_senders: list[pyneat.MetadataSender],
    labels: list[str],
    cfg: AppConfig,
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
        payload = detection_egress.serialize(
            decode_boxes(sample, cfg),
            labels,
            cfg.width,
            cfg.height,
            frame_metadata(sample, index),
        )
        # A nonblocking send drops the message while the socket buffer is full; the drops are
        # counted in the sender's stats and reported when the run ends. Any other failure raises.
        metadata_senders[index].send_raw_json(payload)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.config.is_file():
        print(f"Error: config file not found: {args.config}", file=sys.stderr)
        return 2
    cfg = load_app_config(args.config)
    if args.validate_config_only:
        print(f"Config validated: {args.config} (streams={len(cfg.sources)})")
        return 0

    labels = load_labels(cfg.labels_path)
    fps_by_url = stream_fps(cfg)
    model = pyneat.Model(str(cfg.model_path), model_options(cfg))
    detector = detector_graph(model, cfg)

    graph = pyneat.Graph(options=graph_options(cfg))
    metadata_senders = []
    for index, url in enumerate(cfg.sources):
        rtsp = pyneat.groups.rtsp_encoded_input(rtsp_options(cfg, url, fps_by_url[url]))
        decoder = decoder_graph(cfg, fps_by_url[url])
        graph.connect(rtsp, decoder)
        graph.connect(decoder, detector, detector_link(cfg, index))
        if cfg.video_enabled:
            graph.connect(
                rtsp,
                pyneat.groups.video_sender(video_options(cfg, index)),
                video_link(),
            )
        metadata_senders.append(pyneat.MetadataSender(metadata_options(cfg, index)))
    print(
        f"streams={len(cfg.sources)} insight={cfg.insight_host} "
        f"video_ports={cfg.video_port_base}+ metadata_ports={cfg.metadata_port_base}+"
    )

    stop = stop_on_signals()
    run = graph.build(run_options(cfg))
    try:
        send_detections(run, metadata_senders, labels, cfg, stop)
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
