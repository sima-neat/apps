"""Zero-shot classification of one RTSP stream with CLIP ViT-B/16 and Insight output, using pyneat."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np
import pyneat
import yaml

from clip_tokenizer import ClipTokenizer

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "common" / "config.yaml"
PULL_TIMEOUT_MS = 20000
TEXT_TOKENS = 77
EMBED_DIM = 512
VOCAB_SIZE = 49408
# The checkpoint's logit_scale.exp(): CLIP scores a prompt as 100 times its cosine similarity with the image.
LOGIT_SCALE = 100.0
TOP_PROMPTS = 3
MODEL_SIZE = 224
# CLIP's per-channel mean and std, from the model's README. Neat applies (pixel / 255 - mean) / stddev.
MEAN = [0.48145466, 0.4578275, 0.40821073]
STD = [0.26862954, 0.26130258, 0.27577711]
RTSP_CODECS = {"h264": pyneat.RtspCodec.H264, "h265": pyneat.RtspCodec.H265, "mjpeg": pyneat.RtspCodec.MJPEG}
# FFmpeg's codec tags for the stream, as OpenCV reports them.
STREAM_CODECS = {"h264": "h264", "avc1": "h264", "hevc": "h265", "hev1": "h265", "hvc1": "h265", "h265": "h265",
                 "mjpg": "mjpeg", "jpeg": "mjpeg"}
# The result panel at the bottom left of the video, in pixels of a 720-line frame. Other heights scale it.
PANEL_LINES = 720
PANEL_MARGIN = 24
# The bottom margin keeps the panel above the status line the Insight viewer draws over the video.
PANEL_BOTTOM = 56
PANEL_WIDTH = 640
PANEL_PADDING = 16
TEXT_BAR_GAP = 10
ROW_GAP = 14
# Font scale, line thickness and bar height of each row, best prompt first.
PANEL_ROWS = [(0.75, 2, 14), (0.65, 2, 8), (0.65, 2, 8)]
FONT = cv2.FONT_HERSHEY_SIMPLEX
# The panel keeps this share of the luma and of the chroma's distance from neutral gray.
PANEL_DIM = 0.3
BEST_TEXT_LUMA = 235
TEXT_LUMA = 200
TRACK_LUMA = 60
BAR_LUMA = 170
# Amber (RGB 255, 170, 0) in limited-range BT.601 YUV.
ACCENT_YUV = (167, 41, 178)
NEUTRAL_CHROMA = 128


@dataclass(frozen=True)
class Config:
    image_encoder: str
    text_encoder: str
    tokenizer: str
    token_embeddings: str
    prompts_path: str
    rtsp_url: str
    codec: str
    tcp: bool
    latency_ms: int
    frames: int
    profile: bool
    profile_interval: int
    insight_host: str
    video_port: int
    metadata_port: int


@dataclass(frozen=True)
class Stream:
    width: int
    height: int
    fps: int


@dataclass(frozen=True)
class PanelRow:
    font_scale: float
    thickness: int
    baseline_y: int
    label_width: int
    # The x, y, width and height of the bar at 100%.
    bar: tuple[int, int, int, int]


@dataclass(frozen=True)
class PanelLayout:
    # The x, y, width and height of the darkened area.
    box: tuple[int, int, int, int]
    rows: list[PanelRow]


def load_config(path: Path) -> Config:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    model, source, runtime, insight = raw["model"], raw["source"], raw["runtime"], raw["output"]["insight"]
    if runtime["profile_interval"] <= 0:
        raise ValueError("runtime.profile_interval must be > 0")
    if source["codec"] not in RTSP_CODECS:
        raise ValueError("source.codec must be h264, h265 or mjpeg")
    return Config(
        image_encoder=model["image_encoder"],
        text_encoder=model["text_encoder"],
        tokenizer=model["tokenizer"],
        token_embeddings=model["token_embeddings"],
        prompts_path=raw["prompts"]["path"],
        rtsp_url=source["rtsp_url"],
        codec=source["codec"],
        tcp=source["tcp"],
        latency_ms=source["latency_ms"],
        frames=raw["inference"]["frames"],
        profile=runtime["profile"],
        profile_interval=runtime["profile_interval"],
        insight_host=insight["host"],
        video_port=insight["video_port"],
        metadata_port=insight["metadata_port"],
    )


def load_prompts(path: str) -> list[str]:
    if not Path(path).is_file():
        raise ValueError(f"prompts.path {path} does not exist; point it at a text file with one prompt per line")
    prompts = [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    # A softmax over one prompt always gives it 100%.
    if len(prompts) < 2:
        raise ValueError(f"{path} has {len(prompts)} prompts; write at least two, one per line, to compare them")
    return prompts


def prompt_tokens(tokenizer_path: str, prompts: list[str]) -> np.ndarray:
    """The (prompts, 77) token ids, rejecting a prompt the text encoder would not see whole."""
    if not Path(tokenizer_path).is_file():
        raise ValueError(f"model.tokenizer {tokenizer_path} does not exist; point it at text_encoder/tokenizer.json "
                         "in the model package")
    tokenizer = ClipTokenizer(tokenizer_path)
    tokens = []
    for prompt in prompts:
        count = len(tokenizer.tokens(prompt))
        if count > TEXT_TOKENS - 2:
            raise ValueError(f"prompt '{prompt}' is {count} tokens; the text encoder reads at most "
                             f"{TEXT_TOKENS - 2}, so shorten it")
        tokens.append(tokenizer.encode(prompt, TEXT_TOKENS))
    return np.stack(tokens)


def load_token_embeddings(path: str) -> np.ndarray:
    if not Path(path).is_file():
        raise ValueError(f"model.token_embeddings {path} does not exist; point it at "
                         "text_encoder/token_embeddings.npy in the model package")
    # Memory-mapped: the prompts read only their own rows of the 100 MB table.
    try:
        table = np.load(path, mmap_mode="r")
    except ValueError as error:
        raise ValueError(f"model.token_embeddings {path} is not a NumPy .npy file ({error}); "
                         "download the model package again") from error
    if table.shape != (VOCAB_SIZE, EMBED_DIM):
        raise ValueError(f"model.token_embeddings {path} has shape {table.shape}, not ({VOCAB_SIZE}, {EMBED_DIM}); "
                         "point it at CLIP ViT-B/16's token-embedding table")
    if table.dtype != np.dtype("<f4") or not table.flags.c_contiguous:
        raise ValueError(f"model.token_embeddings {path} is not little-endian float32 in C order; "
                         "save it with numpy.save")
    return table


def text_encoder_options() -> pyneat.ModelOptions:
    # The input is the prompt's token embeddings, not an image.
    options = pyneat.ModelOptions()
    options.preprocess.kind = pyneat.InputKind.Tensor
    options.preprocess.resize.enable = pyneat.AutoFlag.Off
    options.preprocess.color_convert.enable = pyneat.AutoFlag.Off
    options.preprocess.normalize.enable = pyneat.AutoFlag.Off
    return options


def end_of_text_row(rows: np.ndarray, ids: np.ndarray) -> np.ndarray:
    """The unit-length text encoder row at the prompt's first end-of-text token, the highest id in the vocabulary."""
    row = rows.reshape(TEXT_TOKENS, EMBED_DIM)[int(np.argmax(ids))]
    return row / np.linalg.norm(row)


def ev74_tensor(rows: np.ndarray) -> pyneat.Tensor:
    # Neat sizes the input as height 1, width 77 and depth 512, so the tensor declares HWC axes.
    return pyneat.Tensor.from_numpy(np.ascontiguousarray(rows, np.float32)[None], copy=True,
                                    memory=pyneat.TensorMemory.EV74, layout=pyneat.TensorLayout.HWC)


def encode_prompts(cfg: Config, tokens: np.ndarray, table: np.ndarray) -> np.ndarray:
    """The (prompts, 512) unit-length text embeddings, one text encoder run per prompt."""
    model = pyneat.Model(cfg.text_encoder, text_encoder_options())
    runner = model.build([ev74_tensor(table[tokens[0]])])
    vectors = []
    for ids in tokens:
        (output,) = runner.run([ev74_tensor(table[ids])], timeout_ms=PULL_TIMEOUT_MS)
        rows = output.to_numpy(copy=True)
        if rows.dtype != np.float32 or rows.size != TEXT_TOKENS * EMBED_DIM:
            raise RuntimeError(f"model.text_encoder {cfg.text_encoder} did not return 77 x 512 float32 values; "
                               "point it at CLIP ViT-B/16's text encoder")
        vectors.append(end_of_text_row(rows, ids))
    runner.close()
    return np.stack(vectors)


def fourcc_codec(fourcc: float) -> str | None:
    """The codec of an OpenCV FOURCC value, or None when it is not one this application decodes."""
    tag = "".join(chr((int(fourcc) >> 8 * i) & 0xFF) for i in range(4)).lower()
    return STREAM_CODECS.get(tag)


def probe_stream(cfg: Config) -> Stream:
    if cfg.tcp:
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
    capture = cv2.VideoCapture(cfg.rtsp_url, cv2.CAP_FFMPEG)
    stream = Stream(int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    round(capture.get(cv2.CAP_PROP_FPS)))
    stream_codec = fourcc_codec(capture.get(cv2.CAP_PROP_FOURCC))
    capture.release()
    if stream.width <= 0 or stream.height <= 0:
        raise RuntimeError(f"could not read the frame size of {cfg.rtsp_url}; check source.rtsp_url and that the "
                           "stream is running")
    if stream.fps <= 0:
        raise RuntimeError(f"{cfg.rtsp_url} does not report its frame rate; use a stream that reports its frame rate")
    # The decoder accepts only the configured codec. Another codec would fail only after the models load.
    if stream_codec is not None and stream_codec != cfg.codec:
        raise RuntimeError(f"{cfg.rtsp_url} is {stream_codec}, but source.codec is {cfg.codec}; "
                           f"set source.codec to {stream_codec}")
    return stream


def rtsp_options(cfg: Config, stream: Stream) -> pyneat.RtspDecodedInputOptions:
    options = pyneat.RtspDecodedInputOptions()
    options.url = cfg.rtsp_url
    options.tcp = cfg.tcp
    options.latency_ms = cfg.latency_ms
    options.codec = RTSP_CODECS[cfg.codec]
    options.source_fps = stream.fps
    if cfg.codec == "h264":
        options.fallback_h264_width, options.fallback_h264_height = stream.width, stream.height
    else:
        options.dec_width, options.dec_height = stream.width, stream.height
    caps = options.output_caps
    caps.enable = True
    caps.width, caps.height, caps.fps = stream.width, stream.height, stream.fps
    return options


def image_encoder_options(stream: Stream) -> pyneat.ModelOptions:
    # Neat stretches the decoded NV12 frame to the model's 224x224 input and applies CLIP's mean and std on the board.
    options = pyneat.ModelOptions()
    options.preprocess.kind = pyneat.InputKind.Image
    options.preprocess.enable = pyneat.AutoFlag.On
    options.preprocess.color_convert.input_format = pyneat.PreprocessColorFormat.NV12
    options.preprocess.input_max_width = stream.width
    options.preprocess.input_max_height = stream.height
    options.preprocess.resize.enable = pyneat.AutoFlag.On
    options.preprocess.resize.mode = pyneat.ResizeMode.Stretch
    # Without a size Neat infers it from the model and prints a planner warning at every start.
    options.preprocess.resize.width = options.preprocess.resize.height = MODEL_SIZE
    options.preprocess.normalize.enable = pyneat.AutoFlag.On
    options.preprocess.normalize.mean = MEAN
    options.preprocess.normalize.stddev = STD
    options.preprocess.normalize.has_explicit_stats = True
    return options


def video_input_options(stream: Stream) -> pyneat.InputOptions:
    # The application pushes each NV12 frame after drawing the result panel into it.
    options = pyneat.InputOptions()
    options.payload_type = pyneat.PayloadType.Image
    options.format = pyneat.Format.NV12
    options.width, options.height, options.depth = stream.width, stream.height, 1
    options.fps_n, options.fps_d = stream.fps, 1
    return options


def video_options(cfg: Config, fps: int) -> pyneat.VideoSenderOptions:
    encode = pyneat.SimaEncodeOptions()
    encode.fps = fps
    options = pyneat.VideoSenderOptions.from_raw(encode)
    options.host = cfg.insight_host
    options.video_port_base = cfg.video_port
    return options


def metadata_options(cfg: Config) -> pyneat.MetadataSenderOptions:
    options = pyneat.MetadataSenderOptions()
    options.host = cfg.insight_host
    options.metadata_port_base = cfg.metadata_port
    return options


def run_options() -> pyneat.RunOptions:
    options = pyneat.RunOptions()
    options.preset = pyneat.RunPreset.Realtime
    options.queue_depth = 3
    options.overflow_policy = pyneat.OverflowPolicy.KeepLatest
    options.output_memory = pyneat.OutputMemory.ZeroCopy
    return options


def build_graphs(cfg: Config, image_encoder: pyneat.Model, stream: Stream) -> tuple[pyneat.Graph, pyneat.Graph]:
    """The classifier graph, RTSP source -> decoded frames -> {"frame", image encoder -> "embeddings"} joined per
    frame into "result", and the video graph, pushed frames -> Insight video sender."""
    branch = pyneat.graphs.branch("source", ["frame", "model"])
    frame = pyneat.Graph("frame")
    frame.add(pyneat.nodes.output("frame", pyneat.OutputOptions.every_frame(4)))
    model = pyneat.Graph("model")
    model.connect(pyneat.nodes.input("model"), image_encoder)
    embeddings = pyneat.Graph("embeddings")
    embeddings.add(pyneat.nodes.output("embeddings", pyneat.OutputOptions.every_frame(4)))
    result = pyneat.graphs.combine(["frame", "embeddings"], "result", pyneat.CombinePolicy.ByFrame)

    graph = pyneat.Graph()
    graph.connect(pyneat.groups.rtsp_decoded_input(rtsp_options(cfg, stream)), branch)
    graph.connect(branch, frame)
    graph.connect(branch, model)
    graph.connect(model, embeddings)
    graph.connect(frame, result)
    graph.connect(embeddings, result)

    video = pyneat.Graph("video")
    video.connect(pyneat.nodes.input(video_input_options(stream)),
                  pyneat.groups.video_sender(video_options(cfg, stream.fps)))
    return graph, video


def sample_tensors(sample: pyneat.Sample) -> list[pyneat.Tensor]:
    """The sample's tensors: a Sample keeps a single tensor in .tensor and several in .tensors."""
    return [sample.tensor] if sample.kind == pyneat.SampleKind.Tensor else list(sample.tensors)


def image_embedding(sample: pyneat.Sample) -> np.ndarray:
    """The image encoder's 512-value embedding, found by its size: some packages also return 1000 ImageNet logits."""
    sizes = []
    for tensor in sample_tensors(sample):
        size = math.prod(tensor.shape)
        if size == EMBED_DIM:
            return tensor.to_numpy(copy=False).reshape(-1)
        sizes.append(size)
    raise RuntimeError(f"model.image_encoder returned outputs of sizes {sizes}, none of {EMBED_DIM} values; "
                       "point it at CLIP ViT-B/16's image encoder")


def field(sample: pyneat.Sample, label: str) -> pyneat.Sample:
    """The field of a joined sample that came from the named output."""
    for part in sample.fields:
        if part.stream_label == label:
            return part
    raise RuntimeError(f"the joined sample has no {label} field")


def copy_frame(frame: pyneat.Sample, nv12: np.ndarray) -> None:
    """Copies the decoded NV12 frame of `frame` into nv12, a (height * 3 / 2, width) array, leaving the decoder's
    buffer as it is: the decoder may still read it as a reference frame."""
    height, width = nv12.shape[0] * 2 // 3, nv12.shape[1]
    tensors = sample_tensors(frame)
    tensor = tensors[0] if tensors else None
    # Each plane's offset and row stride, the stride being the width when the plane has none.
    planes = {} if tensor is None else {
        plane.role: (plane.byte_offset, plane.strides_bytes[0] if plane.strides_bytes else width)
        for plane in tensor.planes}
    if (tensor is None or not tensor.is_nv12() or (tensor.width(), tensor.height()) != (width, height)
            or pyneat.PlaneRole.Y not in planes or pyneat.PlaneRole.UV not in planes):
        raise RuntimeError(f"the decoded frame is not {width}x{height} NV12")
    # NumPy reads only dense tensors, so this Python copy of the tensor describes its buffer as flat bytes. Reading
    # the buffer in place took 0.3 ms per 1280x720 frame on the DevKit, copy_payload_bytes() 3.3 ms.
    tensor.planes, tensor.shape, tensor.strides_bytes, tensor.axis_semantics = [], [tensor.storage.size_bytes], [1], []
    buffer = tensor.to_numpy(copy=False)
    for role, out in ((pyneat.PlaneRole.Y, nv12[:height]), (pyneat.PlaneRole.UV, nv12[height:])):
        offset, stride = planes[role]
        np.copyto(out, np.ndarray(out.shape, np.uint8, buffer=buffer, offset=offset, strides=(stride, 1)))


def nv12_sample(nv12: np.ndarray, source: pyneat.Sample) -> pyneat.Sample:
    """A sample of the NV12 frame with the source frame's timestamp and id, which Insight matches the metadata to."""
    height, width = nv12.shape[0] * 2 // 3, nv12.shape[1]
    tensor = pyneat.Tensor.from_numpy(nv12, copy=False, image_format=pyneat.PixelFormat.NV12,
                                      memory=pyneat.TensorMemory.CPU)
    planes = []
    for role, rows, offset in ((pyneat.PlaneRole.Y, height, 0), (pyneat.PlaneRole.UV, height // 2, height * width)):
        plane = pyneat.Plane()
        plane.role, plane.shape, plane.strides_bytes, plane.byte_offset = role, [rows, width], [width, 1], offset
        planes.append(plane)
    tensor.shape, tensor.strides_bytes, tensor.planes = [height, width], [], planes
    tensor.axis_semantics = [pyneat.TensorAxisSemantic.H, pyneat.TensorAxisSemantic.W]
    sample = pyneat.make_tensor_sample("video", tensor)
    sample.pts_ns, sample.frame_id = source.pts_ns, source.frame_id
    return sample


def top_prompts(image: np.ndarray, prompt_vectors: np.ndarray, prompts: list[str]) -> list[dict]:
    """The best prompts by softmax probability over all prompts, with their cosine similarity."""
    similarity = (prompt_vectors @ image) / np.sqrt(image @ image)
    logits = LOGIT_SCALE * similarity
    scores = np.exp((logits - logits.max()).astype(np.float64))
    probability = scores / scores.sum()
    # A stable sort keeps prompts with equal scores in file order, as C++'s topk_with_softmax does.
    best = np.argsort(-logits, kind="stable")[:TOP_PROMPTS]
    return [{"label": prompts[i], "confidence": round(float(probability[i]), 4),
             "similarity": round(float(similarity[i]), 4)} for i in best]


def top_classes_json(top: list[dict]) -> str:
    """The classification metadata Insight draws."""
    return json.dumps({"top_classes": top}, separators=(",", ":"))


def even(value: float) -> int:
    """The nearest even pixel count: NV12 has one chroma sample per 2x2 pixels."""
    return int(value / 2 + 0.5) * 2


def text_width(text: str, font_scale: float, thickness: int) -> int:
    return cv2.getTextSize(text, FONT, font_scale, thickness)[0][0]


def panel_layout(width: int, height: int, rows: int) -> PanelLayout:
    """Where the result panel of `rows` prompts goes in a width x height frame."""
    scale = height / PANEL_LINES
    margin, padding = even(PANEL_MARGIN * scale), even(PANEL_PADDING * scale)
    panel_width = min(even(PANEL_WIDTH * scale), width - 2 * margin)
    inner_x, inner_width = margin + padding, panel_width - 2 * padding
    placed, y = [], padding
    for font_scale, thickness, bar_height in PANEL_ROWS[:rows]:
        font_scale, thickness = font_scale * scale, max(1, int(thickness * scale + 0.5))
        text_height = cv2.getTextSize("Ag", FONT, font_scale, thickness)[0][1]
        bar = (inner_x, even(y + text_height + TEXT_BAR_GAP * scale), inner_width, max(2, even(bar_height * scale)))
        placed.append(PanelRow(font_scale, thickness, y + text_height,
                               inner_width - text_width(" 100%", font_scale, thickness), bar))
        y = bar[1] + bar[3] + even(ROW_GAP * scale)
    panel_height = y - even(ROW_GAP * scale) + padding
    top = height - even(PANEL_BOTTOM * scale) - panel_height
    return PanelLayout((margin, top, panel_width, panel_height),
                       [replace(row, baseline_y=row.baseline_y + top, bar=(row.bar[0], row.bar[1] + top, *row.bar[2:]))
                        for row in placed])


def fit_text(text: str, font_scale: float, thickness: int, max_width: int) -> str:
    """The text in ASCII, which is all the panel font draws, cut with "..." when it is wider than max_width."""
    text = text.encode("ascii", "replace").decode("ascii")
    if text_width(text, font_scale, thickness) <= max_width:
        return text
    while text and text_width(text + "...", font_scale, thickness) > max_width:
        text = text[:-1]
    return text.rstrip() + "..."


def text_image(texts: dict, text: str, font_scale: float, thickness: int, luma: int,
               max_width: int = 0) -> tuple[np.ndarray, int]:
    """The text, cut to max_width when that is above 0, drawn on black, and the row of its baseline. Each text,
    style and width is fitted and rendered once into `texts`: drawing antialiased text into the frame took up to
    1.8 ms per line on the DevKit."""
    key = (text, font_scale, thickness, luma, max_width)
    if key not in texts:
        if max_width > 0:
            text = fit_text(text, font_scale, thickness, max_width)
        (width, height), descent = cv2.getTextSize(text, FONT, font_scale, thickness)
        image = np.zeros((height + descent + 2 * thickness, width + 2 * thickness), np.uint8)
        cv2.putText(image, text, (thickness, thickness + height), FONT, font_scale, luma, thickness, cv2.LINE_AA)
        texts[key] = (image, thickness + height)
    return texts[key]


def draw_result_panel(nv12: np.ndarray, top: list[dict], layout: PanelLayout, texts: dict) -> None:
    """Draws the top prompts into an NV12 frame in place: a darkened box with each prompt's text, bar and
    percentage, best first. `texts` keeps the rendered texts from frame to frame."""
    height = nv12.shape[0] * 2 // 3
    luma, chroma = nv12[:height], nv12[height:].reshape(height // 2, -1, 2)
    x, y, w, h = layout.box
    box_luma, box_chroma = luma[y:y + h, x:x + w], chroma[y // 2:(y + h) // 2, x // 2:(x + w) // 2]
    cv2.convertScaleAbs(box_luma, box_luma, PANEL_DIM, 0)
    cv2.convertScaleAbs(box_chroma, box_chroma, PANEL_DIM, NEUTRAL_CHROMA * (1 - PANEL_DIM))
    for i, (row, prompt) in enumerate(zip(layout.rows, top)):
        bar_x, bar_y, bar_width, bar_height = row.bar
        fill = even(bar_width * prompt["confidence"])
        cv2.rectangle(luma, row.bar, TRACK_LUMA, cv2.FILLED)
        if fill > 0:
            cv2.rectangle(luma, (bar_x, bar_y, fill, bar_height), ACCENT_YUV[0] if i == 0 else BAR_LUMA, cv2.FILLED)
            if i == 0:
                cv2.rectangle(chroma, (bar_x // 2, bar_y // 2, fill // 2, bar_height // 2), ACCENT_YUV[1:], cv2.FILLED)
        # The label starts at the bar's left end, the percentage ends at its right end.
        percent = f"{int(prompt['confidence'] * 100 + 0.5)}%"
        for text, max_width, align_right in ((prompt["label"], row.label_width, False), (percent, 0, True)):
            image, baseline = text_image(texts, text, row.font_scale, row.thickness,
                                         BEST_TEXT_LUMA if i == 0 else TEXT_LUMA, max_width)
            left = bar_x + bar_width - image.shape[1] + row.thickness if align_right else bar_x - row.thickness
            area = luma[row.baseline_y - baseline:row.baseline_y - baseline + image.shape[0],
                        left:left + image.shape[1]]
            # The text is brighter than the darkened box, so the brighter pixel of the two keeps its antialiased edge.
            cv2.max(area, image, area)


def run(cfg: Config) -> None:
    # Check the prompts, the table and the stream first: they fail in a second, the models take longer to load.
    prompts = load_prompts(cfg.prompts_path)
    tokens = prompt_tokens(cfg.tokenizer, prompts)
    table = load_token_embeddings(cfg.token_embeddings)
    stream = probe_stream(cfg)

    started = time.perf_counter()
    prompt_vectors = encode_prompts(cfg, tokens, table)
    encoded = time.perf_counter()
    image_encoder = pyneat.Model(cfg.image_encoder, image_encoder_options(stream))
    graph, video = build_graphs(cfg, image_encoder, stream)
    # The sender opens before the Runs build, so a bad host stops the application before the stream starts.
    try:
        metadata = pyneat.MetadataSender(metadata_options(cfg))
    except RuntimeError as error:
        raise RuntimeError(f"could not open the Insight metadata sender: {error}; check output.insight.host") from error
    graph_run, video_run = graph.build(run_options()), video.build(pyneat.RunOptions())
    print(f"rtsp={cfg.rtsp_url} stream={stream.width}x{stream.height}@{stream.fps} codec={cfg.codec} "
          f"prompts={len(prompts)} text_encoding_s={encoded - started:.1f} "
          f"graph_build_s={time.perf_counter() - encoded:.1f} insight={cfg.insight_host} video={cfg.video_port} "
          f"metadata={cfg.metadata_port} channel=0", flush=True)

    layout = panel_layout(stream.width, stream.height, min(TOP_PROMPTS, len(prompts)))
    texts = {}
    processed = skipped_frames = window_start_skipped = 0
    window_start, ranking_ms, panel_ms, last_frame_id = time.perf_counter(), 0.0, 0.0, None
    try:
        while cfg.frames <= 0 or processed < cfg.frames:
            sample = graph_run.pull("result", PULL_TIMEOUT_MS)
            if sample is None:
                error = graph_run.last_error()
                raise RuntimeError(f"no frame classified in {PULL_TIMEOUT_MS} ms; check that {cfg.rtsp_url} is "
                                   "streaming" + (f": {error}" if error else ""))
            ranking_start = time.perf_counter()
            top = top_prompts(image_embedding(field(sample, "embeddings")), prompt_vectors, prompts)
            ranking_ms += (time.perf_counter() - ranking_start) * 1000
            panel_start = time.perf_counter()
            # A new frame each time: the video Run reads the pushed frame later, on its own thread.
            nv12 = np.empty((stream.height * 3 // 2, stream.width), np.uint8)
            copy_frame(field(sample, "frame"), nv12)
            draw_result_panel(nv12, top, layout, texts)
            if not video_run.push([nv12_sample(nv12, sample)]):
                raise RuntimeError("the Insight video sender stopped accepting frames")
            panel_ms += (time.perf_counter() - panel_start) * 1000
            timestamp_ms = sample.pts_ns // 1_000_000 if sample.pts_ns >= 0 else -1
            metadata.send_metadata("classification", top_classes_json(top), timestamp_ms, str(sample.frame_id))

            processed += 1
            # A frame the decoder produced but the image encoder did not classify leaves a gap in the frame ids.
            if last_frame_id is not None:
                skipped_frames += max(0, sample.frame_id - last_frame_id - 1)
            last_frame_id = sample.frame_id
            if cfg.profile and processed % cfg.profile_interval == 0:
                elapsed_s = time.perf_counter() - window_start
                print(f"[profile] frames={cfg.profile_interval} classified_fps={cfg.profile_interval / elapsed_s:.1f} "
                      f"ranking_ms={ranking_ms / cfg.profile_interval:.3f} "
                      f"panel_ms={panel_ms / cfg.profile_interval:.3f} "
                      f"skipped_frames={skipped_frames - window_start_skipped} "
                      f"top='{top[0]['label']}' confidence={top[0]['confidence']:.2f}", flush=True)
                window_start, ranking_ms, panel_ms = time.perf_counter(), 0.0, 0.0
                window_start_skipped = skipped_frames
    finally:
        graph_run.close()
        video_run.close()
        print(f"processed={processed} skipped_frames={skipped_frames} "
              f"video_sender={cfg.insight_host}:{cfg.video_port}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CLIP zero-shot classification of one RTSP stream against text prompts, video and top prompts "
                    "to Insight")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    cfg = load_config(parser.parse_args().config)
    try:
        run(cfg)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
