"""Flag surface defects with PatchCore and outline them, in an RTSP stream or a folder of images.

The model turns each frame into patch features, and the host compares every patch with a memory
bank of known-good patches that --calibrate builds. Insight receives the stream as received and
each anomalous frame's defect regions as polygons; image_dir writes the outlined images instead.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import threading
from pathlib import Path

import cv2
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

from sima_examples import anomaly_regions
from sima_examples.config import Config
from sima_examples.rtsp_probe import RtspStreamInfo, probe_rtsp_stream_info
from sima_examples.stop_signals import stop_on_signals
from utils import memory_bank

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
INPUT_TYPES = ("rtsp", "image_dir")
IMAGE_SUFFIXES = (".bmp", ".jpeg", ".jpg", ".png")
BACKBONE = "wide_resnet50_2"
TORCHVISION_WEIGHTS = "IMAGENET1K_V1"
MODEL_INPUT_SIDE = 224  # The model's input, and the side of the map regions are found on.
PATCH_GRID = (28, 28)
NUM_NEIGHBORS = 9  # Neighborhood size of the image score's reweighting (paper Eq. 7-8).
SEED = 0
MAP_SIGMA = 4.0  # Smoothing of the upsampled score map, in map pixels.
MIN_REGION_PX = 1  # The smoothing already removes single-patch speckle.
THRESHOLD_EVERY = 5  # --calibrate keeps every 5th nominal image out of the bank.
PULL_TIMEOUT_MS = 100
OUTLINE_BGR = (0, 0, 255)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Build the memory bank from app.nominal_images_dir, then exit.",
    )
    return parser.parse_args(argv)


def image_model_options() -> pyneat.ModelOptions:
    """Host-decoded RGB images, already at the model's input size, normalized as ImageNet."""
    options = pyneat.ModelOptions()
    options.preprocess.kind = pyneat.InputKind.Image
    options.preprocess.color_convert.input_format = pyneat.PreprocessColorFormat.RGB
    options.preprocess.preset = pyneat.NormalizePreset.ImageNet
    return options


def stream_model_options(stream: RtspStreamInfo) -> pyneat.ModelOptions:
    """EV74 converts each NV12 frame, stretches it to the model input as image_input does on the
    host, and normalizes it as ImageNet, so the bank's images and the stream match."""
    options = pyneat.ModelOptions()
    options.preprocess.kind = pyneat.InputKind.Image
    options.preprocess.enable = pyneat.AutoFlag.On
    options.preprocess.color_convert.input_format = pyneat.PreprocessColorFormat.NV12
    options.preprocess.input_max_width = stream.width
    options.preprocess.input_max_height = stream.height
    options.preprocess.resize.enable = pyneat.AutoFlag.On
    options.preprocess.resize.mode = pyneat.ResizeMode.Stretch
    options.preprocess.preset = pyneat.NormalizePreset.ImageNet
    return options


def image_input(bgr: np.ndarray) -> pyneat.Tensor:
    side = (MODEL_INPUT_SIDE, MODEL_INPUT_SIDE)
    rgb = cv2.cvtColor(cv2.resize(bgr, side, interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
    return pyneat.Tensor.from_numpy(rgb, copy=True, image_format=pyneat.PixelFormat.RGB)


def image_paths(folder: Path) -> list[Path]:
    paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not paths:
        raise RuntimeError(f"no images in {folder}")
    return paths


def read_image(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"cannot read image {path}")
    return bgr


def patch_features(model: pyneat.Model, bgr: np.ndarray) -> np.ndarray:
    # A view of the output holds its zero-copy loan, and the run stalls once six are held.
    return memory_bank.extract_hwc(model.run([image_input(bgr)])[0].to_numpy(copy=True))


def calibrate(config: Config, model_path: Path) -> None:
    """Build the memory bank from the nominal images and measure both thresholds on the images
    it keeps out, then write memory_bank.npy and bank_meta.json."""
    nominal = image_paths(Path(config.string("app.nominal_images_dir")))
    if len(nominal) < THRESHOLD_EVERY:
        raise RuntimeError(f"app.nominal_images_dir needs at least {THRESHOLD_EVERY} images")
    held_out = nominal[THRESHOLD_EVERY - 1 :: THRESHOLD_EVERY]
    bank_images = [path for path in nominal if path not in held_out]
    model = pyneat.Model(str(model_path), image_model_options())
    coreset_ratio = config.number("app.coreset_ratio")
    bank = memory_bank.MemoryBank.build(
        [patch_features(model, read_image(path)) for path in bank_images], coreset_ratio, SEED
    )

    image_scores: list[float] = []
    patch_scores: list[float] = []
    for path in held_out:
        scored = bank.score(patch_features(model, read_image(path)), NUM_NEIGHBORS)
        image_scores.append(scored.image_score)
        patch_scores.extend(scored.score_map.ravel().tolist())
    percentile = config.number("app.threshold_percentile")
    margin = config.number("app.threshold_margin")
    threshold = memory_bank.percentile_threshold(image_scores, percentile) * margin
    patch_threshold = memory_bank.percentile_threshold(patch_scores, percentile) * margin

    bank_dir = Path(config.string("app.memory_bank_dir"))
    bank_dir.mkdir(parents=True, exist_ok=True)
    bank.save(bank_dir / "memory_bank.npy")
    meta = memory_bank.build_bank_meta(
        model_path=model_path,
        bank_path=bank_dir / "memory_bank.npy",
        backbone=BACKBONE,
        torchvision_weights=TORCHVISION_WEIGHTS,
        embed_dim=memory_bank.EMBED_DIM,
        patch_grid=PATCH_GRID,
        coreset_ratio=coreset_ratio,
        seed=SEED,
        num_nominal_images=len(bank_images),
        bank_size=bank.size,
        num_neighbors=NUM_NEIGHBORS,
        gaussian_sigma=MAP_SIGMA,
        threshold=threshold,
        threshold_percentile=percentile,
        threshold_num_images=len(held_out),
        patch_threshold=patch_threshold,
    )
    memory_bank.save_bank_meta(bank_dir / "bank_meta.json", meta)
    print(
        f"bank={bank.size} patches from {len(bank_images)} images "
        f"threshold={threshold:.4f} patch_threshold={patch_threshold:.4f} "
        f"({len(held_out)} held-out images) saved={bank_dir}",
        flush=True,
    )


def load_bank(config: Config, model_path: Path) -> tuple[memory_bank.MemoryBank, dict]:
    """The calibrated bank and its metadata, checked against the configured model."""
    bank_dir = Path(config.string("app.memory_bank_dir"))
    bank_path, meta_path = bank_dir / "memory_bank.npy", bank_dir / "bank_meta.json"
    if not bank_path.exists() or not meta_path.exists():
        raise RuntimeError(f"no memory bank in {bank_dir}; run with --calibrate first")
    meta = memory_bank.load_bank_meta(meta_path)
    memory_bank.verify_bank_matches_model(meta, model_path)
    memory_bank.verify_bank_hash(meta, bank_path)
    if "value" not in meta.get("patch_threshold", {}):
        raise RuntimeError(f"{meta_path} has no patch_threshold; run with --calibrate again")
    return memory_bank.MemoryBank.load(bank_path), meta


def region_options(config: Config, meta: dict, width: int, height: int) -> anomaly_regions.RegionOptions:
    return anomaly_regions.RegionOptions(
        width=width,
        height=height,
        threshold=float(meta["patch_threshold"]["value"]),
        min_region_px=MIN_REGION_PX,
        max_regions=config.integer("inference.max_detections"),
    )


def defect_regions(
    bank: memory_bank.MemoryBank,
    features: np.ndarray,
    threshold: float,
    options: anomaly_regions.RegionOptions,
) -> tuple[float, list[anomaly_regions.Region]]:
    """The frame's anomaly score, and its defect regions when the score is above threshold."""
    scored = bank.score(features, NUM_NEIGHBORS)
    if scored.image_score <= threshold:
        return scored.image_score, []
    side_map = memory_bank.anomaly_map(scored.score_map, MODEL_INPUT_SIDE, MAP_SIGMA)
    # Insight reads a segment's confidence as 0 to 1, so the raw distance is scaled to it.
    regions = [
        dataclasses.replace(region, score=memory_bank.confidence(region.score, options.threshold))
        for region in anomaly_regions.defect_regions(side_map, options)
    ]
    return scored.image_score, regions


def score_images(config: Config, model_path: Path, bank: memory_bank.MemoryBank, meta: dict) -> None:
    """Score every image of input.source and write it to output.dir with its regions outlined."""
    output_dir = Path(config.string("output.dir"))
    output_dir.mkdir(parents=True, exist_ok=True)
    threshold = float(meta["threshold"]["value"])
    model = pyneat.Model(str(model_path), image_model_options())
    flagged = 0
    paths = image_paths(Path(config.string("input.source")))
    for path in paths:
        bgr = read_image(path)
        options = region_options(config, meta, bgr.shape[1], bgr.shape[0])
        score, regions = defect_regions(bank, patch_features(model, bgr), threshold, options)
        for region in regions:
            cv2.polylines(bgr, [np.array(region.polygon, np.int32)], True, OUTLINE_BGR, 2)
        if not cv2.imwrite(str(output_dir / path.name), bgr):
            raise RuntimeError(f"cannot write {output_dir / path.name}")
        verdict = "ANOMALOUS" if score > threshold else "normal"
        print(
            f"{path.name}: {verdict} score={score:.2f} threshold={threshold:.2f} "
            f"regions={len(regions)}",
            flush=True,
        )
        flagged += score > threshold
    print(f"processed={len(paths)} flagged={flagged} output={output_dir}", flush=True)


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


def features_graph(model: pyneat.Model) -> pyneat.Graph:
    """Decoded frames -> PatchCore feature extractor -> "features" output."""
    graph = pyneat.Graph("extractor")
    graph.add(model.graph())
    graph.add(pyneat.nodes.output("features"))
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


def sample_features(sample: pyneat.Sample) -> np.ndarray:
    tensor = sample.tensor if sample.kind == pyneat.SampleKind.Tensor else sample.tensors[0]
    return memory_bank.extract_hwc(tensor.to_numpy(copy=False))


def send_regions(
    run: pyneat.Run,
    metadata_sender: pyneat.MetadataSender,
    bank: memory_bank.MemoryBank,
    threshold: float,
    options: anomaly_regions.RegionOptions,
    frames: int,
    stop: threading.Event,
) -> tuple[int, int]:
    """Send every frame's defect regions to Insight until `frames` (0: no limit) or a stop
    signal; return the frames processed and the anomalous frames."""
    processed = 0
    flagged = 0
    while not stop.is_set() and (frames == 0 or processed < frames):
        sample = run.pull("features", PULL_TIMEOUT_MS)
        if sample is None:
            if run.last_error() or not run.running():
                raise RuntimeError(f"patch_features output closed: {run.last_error()}")
            continue
        _, regions = defect_regions(bank, sample_features(sample), threshold, options)
        # A normal frame sends no segments, which clears the last outline. A nonblocking send
        # drops the message while the socket buffer is full; the drops are counted in the
        # sender's stats and reported when the run ends. Any other failure raises.
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
    model_path = args.config.absolute().parent / config.string("model.path")
    if args.calibrate:
        calibrate(config, model_path)
        return 0
    bank, meta = load_bank(config, model_path)
    if config.choice("input.type", {name: name for name in INPUT_TYPES}) == "image_dir":
        score_images(config, model_path, bank, meta)
        return 0

    stream = probe_rtsp_stream_info(
        config.string("input.source"),
        config.boolean("input.rtsp.tcp"),
        RtspStreamInfo(
            config.integer("input.width"),
            config.integer("input.height"),
            config.integer("input.fps"),
        ),
    )
    model = pyneat.Model(str(model_path), stream_model_options(stream))

    rtsp = pyneat.groups.rtsp_encoded_input(rtsp_options(config, stream))
    decoder = decoder_graph(config, stream)
    video = pyneat.groups.video_sender(video_options(config))
    graph = pyneat.Graph()
    graph.connect(rtsp, decoder)
    # A plain link lets Core tee the encoded stream to the decoder and the video sender.
    graph.connect(rtsp, video)
    graph.connect(decoder, features_graph(model))

    metadata_sender = pyneat.MetadataSender(metadata_options(config))
    stop = stop_on_signals()
    run = graph.build(run_options())
    host = config.string("output.insight.host")
    video_port = config.integer("output.insight.video_port_base")
    print(
        f"source={config.string('input.source')} codec={config.string('input.codec')} "
        f"stream={stream.width}x{stream.height}@{stream.fps} bank={bank.size} "
        f"insight={host} video={video_port} "
        f"metadata={config.integer('output.insight.metadata_port_base')}",
        flush=True,
    )
    try:
        processed, flagged = send_regions(
            run,
            metadata_sender,
            bank,
            float(meta["threshold"]["value"]),
            region_options(config, meta, stream.width, stream.height),
            config.integer("inference.frames"),
            stop,
        )
    finally:
        run.close()

    print(f"processed={processed} flagged={flagged} video_sender={host}:{video_port}", flush=True)
    if metadata_sender.stats().send_failures:
        print(f"{metadata_sender.stats().send_failures} metadata messages dropped", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:  # noqa: BLE001 - report the cause in one line, not a traceback
        print(f"[ERR] {error}", file=sys.stderr)
        sys.exit(1)
