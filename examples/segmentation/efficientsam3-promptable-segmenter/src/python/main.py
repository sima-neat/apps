"""Segment what a text prompt names in one RTSP stream with EfficientSAM3, shown in Insight.

The text encoder turns the prompt into text features once, at startup. The decoded frames go to
Insight's video and to the host, which hands the newest one, with the text features, to the
segmentation model. Insight receives every frame's instance outlines.
"""

from __future__ import annotations

import argparse
import sys
import threading
from collections import deque
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
from utils import clip_tokenizer, efficientsam3

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "common" / "config.yaml"
# The text encoder package sits next to the segmentation model that model.path names.
TEXT_ENCODER_PACKAGE = "efficientsam3_text_encoder_mpk.tar.gz"
RTSP_CODECS = {"h264": pyneat.RtspCodec.H264, "h265": pyneat.RtspCodec.H265}
PULL_TIMEOUT_MS = 100
# The text encoder reads 16 tokens, each one-hot over the CLIP vocabulary, and returns 256
# features per token.
TEXT_TOKENS = 16
VOCAB_SIZE = 49408
TEXT_DIM = 256
# Three frames on the model keep the MLA busy while the host prepares and reads frames (13 FPS;
# 11 FPS with two).
PIPELINE_DEPTH = 3


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args(argv)


def ev74_tensor(array: np.ndarray) -> pyneat.Tensor:
    """A model input in EV74 memory, built from the array with copy=False."""
    return pyneat.Tensor.from_numpy(
        array, copy=False, memory=pyneat.TensorMemory.EV74, layout=pyneat.TensorLayout.HWC
    )


def text_features(text_encoder: pyneat.Model, tokens: np.ndarray) -> np.ndarray:
    """The segmentation model's text input: the encoder's features of each token, then a row
    that marks the prompt's tokens."""
    # The MLA cannot look up token ids, so the text encoder takes them one-hot.
    one_hot = np.zeros((1, TEXT_TOKENS, VOCAB_SIZE), np.float32)
    one_hot[0, np.arange(TEXT_TOKENS), tokens] = 1.0
    encoder = text_encoder.build([ev74_tensor(one_hot)])
    encoder.push([ev74_tensor(one_hot)])
    features = encoder.pull().tensors[0].to_numpy(copy=True)
    encoder.close()
    text = np.zeros((1, TEXT_TOKENS + 1, TEXT_DIM), np.float32)
    text[0, :TEXT_TOKENS] = features.reshape(TEXT_TOKENS, TEXT_DIM)
    text[0, TEXT_TOKENS, :TEXT_TOKENS] = tokens != 0
    return text


def segmenter_run_options() -> pyneat.RunOptions:
    options = pyneat.RunOptions()
    options.queue_depth = PIPELINE_DEPTH
    return options


def rtsp_options(config: Config, stream: RtspStreamInfo) -> pyneat.RtspDecodedInputOptions:
    """Receive the stream and decode it to NV12 frames of the stream's size."""
    options = pyneat.RtspDecodedInputOptions()
    options.url = config.string("input.source")
    options.codec = config.choice("input.codec", RTSP_CODECS)
    options.tcp = config.boolean("input.rtsp.tcp")
    options.latency_ms = config.integer("input.rtsp.latency_ms")
    if options.codec == pyneat.RtspCodec.H264:
        options.fallback_h264_width = stream.width
        options.fallback_h264_height = stream.height
        options.fallback_h264_fps = stream.fps
    else:
        options.dec_width = stream.width
        options.dec_height = stream.height
        options.source_fps = stream.fps
    return options


def video_options(config: Config, stream: RtspStreamInfo) -> pyneat.VideoSenderOptions:
    """The decoded frames, encoded as H.264 for Insight."""
    encode = pyneat.SimaEncodeOptions()
    encode.fps = stream.fps
    options = pyneat.VideoSenderOptions.from_raw(encode)
    options.host = config.string("output.insight.host")
    options.video_port_base = config.integer("output.insight.video_port_base")
    return options


def frames_options() -> pyneat.OutputOptions:
    """Hold up to 8 decoded frames for the host, dropping the oldest when the host falls behind."""
    options = pyneat.OutputOptions.every_frame(8)
    options.drop = True
    return options


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
    return options


def segment_options(
    config: Config, stream: RtspStreamInfo, prompt: str
) -> efficientsam3.SegmentOptions:
    return efficientsam3.SegmentOptions(
        width=stream.width,
        height=stream.height,
        label=prompt,
        min_score=config.number("inference.min_score"),
        max_detections=config.integer("inference.max_detections"),
        mask_threshold=config.number("app.mask_threshold"),
    )


def nv12(tensor: pyneat.Tensor) -> np.ndarray:
    """The frame's NV12 bytes as height * 3 / 2 rows of width bytes."""
    # contiguous() drops the row padding the decoder adds to some frames.
    packed = np.frombuffer(tensor.contiguous().copy_payload_bytes(), np.uint8)
    return packed.reshape(tensor.height() * 3 // 2, tensor.width())


def newest_frame(
    run: pyneat.Run, overlay: efficientsam3.OverlayClock, timeout_ms: int
) -> pyneat.Sample | None:
    """The newest decoded frame, waiting up to `timeout_ms` for the first. Every frame pulled
    joins the overlay, so it is sent a result too."""
    newest = None
    while (sample := run.pull("frames", timeout_ms)) is not None:
        overlay.add_frame(sample.pts_ns // 1_000_000, str(sample.frame_id))
        newest, timeout_ms = sample, 0
    return newest


def segment_frames(
    run: pyneat.Run,
    segmenter: pyneat.ModelRunner,
    text: pyneat.Tensor,
    metadata_sender: pyneat.MetadataSender,
    options: efficientsam3.SegmentOptions,
    frames: int,
    stop: threading.Event,
) -> int:
    """Segment the newest frame whenever the model has room, and send every frame the nearest
    result, until `frames` results (0: no limit) or a stop signal; return the count."""
    # A pushed tensor is built from its array with copy=False, so each frame on the model keeps
    # its own array, and one more is filled meanwhile.
    images = [
        np.empty((efficientsam3.MODEL_SIZE, efficientsam3.MODEL_SIZE, 3), np.float32)
        for _ in range(PIPELINE_DEPTH + 1)
    ]
    overlay = efficientsam3.OverlayClock()
    in_flight: deque[int] = deque()  # The presentation time, in ms, of each frame on the model.
    pushed = processed = 0
    while not stop.is_set() and (frames == 0 or processed < frames):
        if len(in_flight) < PIPELINE_DEPTH:
            sample = newest_frame(run, overlay, 0 if in_flight else PULL_TIMEOUT_MS)
            if sample is not None:
                image = images[pushed % len(images)]
                efficientsam3.preprocess(nv12(sample.tensors[0]), image)
                segmenter.push([ev74_tensor(image), text])
                pushed += 1
                in_flight.append(sample.pts_ns // 1_000_000)
                continue
            if not in_flight:
                if run.last_error() or not run.running():
                    raise RuntimeError(f"frames output closed: {run.last_error()}")
                continue

        result = segmenter.pull()
        detections, masks = (tensor.to_numpy(copy=False) for tensor in result.tensors)
        data = efficientsam3.segments_json(efficientsam3.segments_of(detections, masks, options))
        for message, timestamp_ms, frame_id in overlay.add_result(in_flight.popleft(), data):
            # A nonblocking send drops the message while the socket buffer is full; the drops
            # are counted in the sender's stats and reported when the run ends. Any other
            # failure raises.
            metadata_sender.send_metadata("segmentation", message, timestamp_ms, frame_id)
        processed += 1
    return processed


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = Config.load(args.config)
    # Check the prompt and the stream first: loading the two models takes about a minute.
    prompt = config.string("app.prompt")
    tokens = clip_tokenizer.prompt_tokens(clip_tokenizer.ClipTokenizer(), prompt, TEXT_TOKENS)
    configured = RtspStreamInfo(
        config.integer("input.width"), config.integer("input.height"), config.integer("input.fps")
    )
    # The stream's size and rate come from the probe for each value the config leaves at 0.
    stream = probe_rtsp_stream_info(
        config.string("input.source"), config.boolean("input.rtsp.tcp"), configured
    )
    model_path = args.config.absolute().parent / config.string("model.path")
    text_encoder = pyneat.Model(str(model_path.with_name(TEXT_ENCODER_PACKAGE)))
    text = ev74_tensor(text_features(text_encoder, tokens))
    image = ev74_tensor(
        np.zeros((efficientsam3.MODEL_SIZE, efficientsam3.MODEL_SIZE, 3), np.float32)
    )
    segmenter = pyneat.Model(str(model_path)).build(
        [image, text], run_options=segmenter_run_options()
    )

    source = pyneat.groups.rtsp_decoded_input(rtsp_options(config, stream))
    graph = pyneat.Graph()
    graph.connect(source, pyneat.groups.video_sender(video_options(config, stream)))
    graph.connect(source, pyneat.nodes.output("frames", frames_options()))

    metadata_sender = pyneat.MetadataSender(metadata_options(config))
    stop = stop_on_signals()
    run = graph.build(run_options())
    host = config.string("output.insight.host")
    video_port = config.integer("output.insight.video_port_base")
    print(
        f"source={config.string('input.source')} codec={config.string('input.codec')} "
        f"stream={stream.width}x{stream.height}@{stream.fps} prompt='{prompt}' "
        f"tokens={int((tokens != 0).sum())} insight={host} video={video_port} "
        f"metadata={config.integer('output.insight.metadata_port_base')}",
        flush=True,
    )
    try:
        processed = segment_frames(
            run,
            segmenter,
            text,
            metadata_sender,
            segment_options(config, stream, prompt),
            config.integer("inference.frames"),
            stop,
        )
    finally:
        run.close()
        segmenter.close()

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
