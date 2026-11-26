"""Flag surface defects in one RTSP stream with FastFlow and outline them in Insight.

The decoded frames feed the model, and the host turns its anomaly map into defect regions.
Insight receives the video as received and every frame's regions as polygons.
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
from sima_examples.rtsp_probe import RtspStreamInfo, probe_rtsp_stream_info
from sima_examples.stop_signals import stop_on_signals
from utils import anomaly_regions

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
# The mean each package expects, in RGB order on 0..1 pixels. fastflow_demo takes its training
# set's mean; an anomalib package normalizes inside its graph.
NORMALIZE_MEANS = {
    "fastflow_demo": [0.3893437, 0.35421189, 0.36142577],
    "anomalib": [0.0, 0.0, 0.0],
}
PULL_TIMEOUT_MS = 100


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args(argv)


def model_options(config: Config, stream: RtspStreamInfo) -> pyneat.ModelOptions:
    """EV74 converts each NV12 frame, stretches it to the model input, and normalizes it.

    Stretching makes the map cover the whole frame, so a region scales back to the frame by
    its width and height. A square frame, as FastFlow is trained on, is not distorted.
    """
    options = pyneat.ModelOptions()
    options.preprocess.kind = pyneat.InputKind.Image
    options.preprocess.enable = pyneat.AutoFlag.On
    options.preprocess.color_convert.input_format = pyneat.PreprocessColorFormat.NV12
    options.preprocess.input_max_width = stream.width
    options.preprocess.input_max_height = stream.height
    options.preprocess.resize.enable = pyneat.AutoFlag.On
    options.preprocess.resize.mode = pyneat.ResizeMode.Stretch
    options.preprocess.normalize.mean = config.choice("model.id", NORMALIZE_MEANS)
    options.preprocess.normalize.has_explicit_stats = True
    return options


def rtsp_options(config: Config, stream: RtspStreamInfo) -> pyneat.RtspEncodedInputOptions:
    options = pyneat.RtspEncodedInputOptions()
    options.url = config.string("input.source")
    options.codec = config.choice("input.codec", RTSP_CODECS)
    options.tcp = config.boolean("input.rtsp.tcp")
    options.latency_ms = config.integer("input.rtsp.latency_ms")
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


def detector_graph(model: pyneat.Model) -> pyneat.Graph:
    """Decoded frames -> FastFlow -> "anomaly_map" output."""
    graph = pyneat.Graph("detector")
    graph.add(model.graph())
    graph.add(pyneat.nodes.output("anomaly_map"))
    return graph


def video_options(config: Config) -> pyneat.VideoSenderOptions:
    """Send the stream to Insight as received."""
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


def region_options(config: Config, stream: RtspStreamInfo) -> anomaly_regions.RegionOptions:
    return anomaly_regions.RegionOptions(
        width=stream.width,
        height=stream.height,
        threshold=config.number("app.anomaly_threshold"),
        min_region_px=config.integer("app.min_region_px"),
        max_regions=config.integer("inference.max_detections"),
    )


def anomaly_map(sample: pyneat.Sample, map_side: int) -> np.ndarray:
    """The map of an "anomaly_map" sample: per pixel, the probability it shows no good part."""
    tensor = sample.tensor if sample.kind == pyneat.SampleKind.Tensor else sample.tensors[0]
    values = tensor.to_numpy(copy=False).astype(np.float32, copy=False)
    return values.reshape(map_side, map_side)


def send_regions(
    run: pyneat.Run,
    metadata_sender: pyneat.MetadataSender,
    map_side: int,
    options: anomaly_regions.RegionOptions,
    frames: int,
    stop: threading.Event,
) -> tuple[int, int]:
    """Send every frame's defect regions to Insight until `frames` (0: no limit) or a stop
    signal; return the frames processed and the frames with a defect."""
    processed = 0
    flagged = 0
    while not stop.is_set() and (frames == 0 or processed < frames):
        sample = run.pull("anomaly_map", PULL_TIMEOUT_MS)
        if sample is None:
            if run.last_error() or not run.running():
                raise RuntimeError(f"anomaly_map output closed: {run.last_error()}")
            continue
        regions = anomaly_regions.defect_regions(anomaly_map(sample, map_side), options)
        # A frame without a defect sends no segments, which clears the last outline. A
        # nonblocking send drops the message while the socket buffer is full; the drops are
        # counted in the sender's stats and reported when the run ends. Any other failure raises.
        metadata_sender.send_metadata(
            "segmentation",
            anomaly_regions.segments_json(regions),
            sample.pts_ns // 1_000_000 if sample.pts_ns >= 0 else -1,
            str(sample.frame_id) if sample.frame_id >= 0 else "",
        )
        processed += 1
        flagged += bool(regions)
    return processed, flagged


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
    model_path = args.config.absolute().parent / config.string("model.path")
    model = pyneat.Model(str(model_path), model_options(config, stream))
    map_side = int(model.output_specs()[0].shape[1])

    rtsp = pyneat.groups.rtsp_encoded_input(rtsp_options(config, stream))
    decoder = decoder_graph(config, stream)
    video = pyneat.groups.video_sender(video_options(config))
    graph = pyneat.Graph()
    graph.connect(rtsp, decoder)
    # A plain link lets Core tee the encoded stream to the decoder and the video sender.
    graph.connect(rtsp, video)
    graph.connect(decoder, detector_graph(model))

    metadata_sender = pyneat.MetadataSender(metadata_options(config))
    stop = stop_on_signals()
    run = graph.build(run_options())
    host = config.string("output.insight.host")
    video_port = config.integer("output.insight.video_port_base")
    print(
        f"source={config.string('input.source')} codec={config.string('input.codec')} "
        f"stream={stream.width}x{stream.height}@{stream.fps} map={map_side}x{map_side} "
        f"insight={host} video={video_port} "
        f"metadata={config.integer('output.insight.metadata_port_base')}"
    )
    try:
        processed, flagged = send_regions(
            run,
            metadata_sender,
            map_side,
            region_options(config, stream),
            config.integer("inference.frames"),
            stop,
        )
    finally:
        run.close()

    print(f"processed={processed} flagged={flagged} video_sender={host}:{video_port}")
    if metadata_sender.stats().send_failures:
        print(f"{metadata_sender.stats().send_failures} metadata messages dropped")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:  # noqa: BLE001 - report the cause in one line, not a traceback
        print(f"[ERR] {error}", file=sys.stderr)
        sys.exit(1)
