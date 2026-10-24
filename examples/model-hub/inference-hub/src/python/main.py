"""Run a selected model on one live source and show the video and results in Neat Insight."""

from __future__ import annotations

import argparse
import importlib
import json
import signal
import sys
import threading
import time
from pathlib import Path

import cv2
import yaml

import pyneat

PYTHON_DIR = Path(__file__).resolve().parent
MODELS_DIR = PYTHON_DIR / "hub_models"
LABELS_DIR = PYTHON_DIR.parent / "common" / "labels"
CODECS = {"h264", "h265", "mjpeg"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args(argv)


def load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must hold a mapping")
    return data


def load_config(path: Path) -> dict:
    cfg = load_yaml(path)
    model = cfg["model"]
    available = sorted(p.parent.name for p in MODELS_DIR.glob("*/config.yaml"))
    if model["name"] not in available:
        raise ValueError(f"model.name must be one of {available}, got {model['name']!r}")
    model_cfg = load_yaml(MODELS_DIR / model["name"] / "config.yaml")
    if str(model["variant"]) not in model_cfg["variants"]:
        raise ValueError(
            f"model.variant for {model['name']} must be one of "
            f"{list(model_cfg['variants'])}, got {model['variant']!r}"
        )
    source = cfg["source"]
    if source["type"] not in {"rtsp", "camera"}:
        raise ValueError("source.type must be rtsp or camera")
    if source["type"] == "rtsp" and source["rtsp"]["codec"] not in CODECS:
        raise ValueError(f"source.rtsp.codec must be one of {sorted(CODECS)}")
    cfg["model_cfg"] = model_cfg
    return cfg


def install_commands(model_id: str, build: str, models_dir: str) -> str:
    package = f"{models_dir}/{model_id}/{build}/{model_id}_{build}_default.zip"
    return (
        f"sima-cli models download --stg -b develop --id {model_id} --variant {build} -o {models_dir}\n"
        f"unzip -o {package} -d {models_dir}"
    )


def find_packs(cfg: dict) -> dict[str, str]:
    """Map each component of the selected variant to its installed Model Registry MPK."""
    model_cfg = cfg["model_cfg"]
    models_dir = cfg["model"]["dir"]
    model_id = model_cfg["variants"][str(cfg["model"]["variant"])]
    packs = {}
    for component in model_cfg["components"]:
        found = sorted(Path(models_dir, model_id, "components", component).glob("*_mpk.tar.gz"))
        if not found:
            raise FileNotFoundError(
                f"{model_id} is not installed in {models_dir}. Install it with:\n"
                + install_commands(model_id, model_cfg["build"], models_dir)
            )
        packs[component] = str(found[0])
    return packs


def load_labels(name: str) -> list[str]:
    return [line.strip() for line in (LABELS_DIR / name).read_text(encoding="utf-8").splitlines()]


def probe_rtsp(url: str) -> tuple[int, int, int]:
    capture = cv2.VideoCapture(url)
    try:
        if not capture.isOpened():
            raise RuntimeError(f"cannot open RTSP source {url}")
        return (
            int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            round(capture.get(cv2.CAP_PROP_FPS)),
        )
    finally:
        capture.release()


def source_geometry(source: dict) -> tuple[int, int, int]:
    if source["type"] == "camera":
        camera = source["camera"]
        return camera["width"], camera["height"], camera["fps"]
    rtsp = source["rtsp"]
    width, height, fps = probe_rtsp(rtsp["url"])
    fps = rtsp.get("fps") or fps
    if width <= 0 or height <= 0 or fps <= 0:
        raise RuntimeError("could not read the RTSP size and FPS; set source.rtsp.fps")
    return width, height, fps


def rtsp_options(rtsp: dict, width: int, height: int, fps: int):
    opt = pyneat.RtspEncodedInputOptions()
    opt.url = rtsp["url"]
    opt.codec = {"h264": pyneat.RtspCodec.H264, "h265": pyneat.RtspCodec.H265,
                 "mjpeg": pyneat.RtspCodec.MJPEG}[rtsp["codec"]]
    opt.latency_ms = rtsp["latency_ms"]
    opt.tcp = rtsp["tcp"]
    if rtsp["tcp"]:
        opt.buffer_mode = "none"
    opt.source_fps = fps
    if rtsp["codec"] == "h264":
        opt.fallback_h264_width = width
        opt.fallback_h264_height = height
    return opt


def decoder_options(codec: str, width: int, height: int, fps: int):
    opt = pyneat.SimaDecodeOptions()
    opt.type = {"h264": pyneat.SimaDecodeType.H264, "h265": pyneat.SimaDecodeType.H265,
                "mjpeg": pyneat.SimaDecodeType.MJPEG}[codec]
    opt.out_format = pyneat.Format.NV12
    opt.raw_output = True
    opt.dec_width = width
    opt.dec_height = height
    opt.dec_fps = fps
    if codec == "mjpeg":
        # Keep free decode surfaces while inference and the preview hold frames.
        opt.num_buffers = 32
    return opt


def camera_options(camera: dict):
    opt = pyneat.CameraInputOptions()
    if camera["name"]:
        opt.camera_name = camera["name"]
    opt.width = camera["width"]
    opt.height = camera["height"]
    opt.framerate_num = camera["fps"]
    opt.framerate_den = 1
    opt.format = "NV12"
    return opt


def video_options(cfg: dict, width: int, height: int, fps: int):
    """Forward H.264/H.265 as received; encode decoded MJPEG and camera frames to H.264."""
    source = cfg["source"]
    if source["type"] == "rtsp" and source["rtsp"]["codec"] != "mjpeg":
        codec = pyneat.RtspCodec.H264 if source["rtsp"]["codec"] == "h264" else pyneat.RtspCodec.H265
        opt = pyneat.VideoSenderOptions.passthrough(codec)
    else:
        opt = pyneat.VideoSenderOptions.h264_rtp_udp_from_raw(width, height, fps)
    opt.host = cfg["insight"]["host"]
    opt.video_port_base = cfg["insight"]["video_port"]
    opt.channel = 0
    opt.async_ = False
    return opt


def latest_queue(depth: int):
    """Drops the oldest frame instead of slowing the source when inference lags."""
    opt = pyneat.QueueOptions()
    opt.max_buffers = depth
    opt.overflow_policy = pyneat.OverflowPolicy.KeepLatest
    return opt


def preview_link():
    """Bounds raw preview frames so encoding cannot hold source buffers."""
    link = pyneat.GraphLinkOptions()
    link.policy = pyneat.GraphLinkPolicy.RealtimeLatestByStream
    link.max_inflight_per_stream = 2
    link.max_inflight_total = 2
    return link


def run_options():
    opt = pyneat.RunOptions()
    opt.preset = pyneat.RunPreset.Realtime
    opt.output_memory = pyneat.OutputMemory.ZeroCopy
    opt.advanced.prepare_output_cpu_visible = True
    return opt


def metadata_options(insight: dict):
    opt = pyneat.MetadataSenderOptions()
    opt.host = insight["host"]
    opt.metadata_port_base = insight["metadata_port"]
    opt.channel = 0
    return opt


def build_graph(cfg: dict, neat_model, width: int, height: int, fps: int):
    """Decode the source once; send it to Insight and its newest frames to the model."""
    video = pyneat.Graph("video")
    video.add(pyneat.groups.video_sender(video_options(cfg, width, height, fps)))

    inference = pyneat.Graph("inference")
    inference.add(pyneat.nodes.queue(latest_queue(4)))
    inference.add(neat_model.graph())
    result = pyneat.Graph("result")
    result.add(pyneat.nodes.output("result", pyneat.OutputOptions.every_frame(4)))
    inference.add(result)

    source = cfg["source"]
    graph = pyneat.Graph("inference_hub")
    if source["type"] == "camera":
        camera = pyneat.Graph("camera")
        camera.add(pyneat.nodes.camera_input(camera_options(source["camera"])))
        graph.connect(camera, video, preview_link())
        graph.connect(camera, inference)
        return graph

    rtsp = source["rtsp"]
    encoded = pyneat.Graph("rtsp")
    encoded.add(pyneat.groups.rtsp_encoded_input(rtsp_options(rtsp, width, height, fps)))
    decoder = pyneat.Graph("decoder")
    decoder.add(pyneat.nodes.sima_decode(decoder_options(rtsp["codec"], width, height, fps)))
    graph.connect(encoded, decoder)
    if rtsp["codec"] == "mjpeg":
        graph.connect(decoder, video, preview_link())
    else:
        # A plain source link lets Core tee the encoded stream to the decoder and the sender.
        graph.connect(encoded, video)
    graph.connect(decoder, inference)
    return graph


def run(cfg: dict) -> int:
    model_cfg = cfg["model_cfg"]
    packs = find_packs(cfg)
    width, height, fps = source_geometry(cfg["source"])
    module = importlib.import_module(f"hub_models.{cfg['model']['name']}.model")
    model = module.Model(model_cfg, packs, load_labels(model_cfg["labels"]), width, height)

    graph = build_graph(cfg, model.neat_model, width, height, fps)
    source_run = graph.build(run_options())
    metadata_sender = pyneat.MetadataSender(metadata_options(cfg["insight"]))
    print(
        f"{cfg['model']['name']} {cfg['model']['variant']} on {cfg['source']['type']} "
        f"{width}x{height}@{fps} -> Insight {cfg['insight']['host']} "
        f"video={cfg['insight']['video_port']} metadata={metadata_sender.metadata_port()}",
        flush=True,
    )

    stop = threading.Event()
    previous = {sig: signal.signal(sig, lambda *_: stop.set()) for sig in (signal.SIGINT, signal.SIGTERM)}
    frames = cfg.get("inference", {}).get("frames", 0)
    published = 0
    started = time.monotonic()
    results = model.results(source_run, stop)
    try:
        for sample in results:
            timestamp_ms = sample.pts_ns // 1_000_000 if sample.pts_ns >= 0 else -1
            frame_id = str(sample.frame_id) if sample.frame_id >= 0 else ""
            for metadata_type, data in model.postprocess(sample):
                metadata_sender.send_metadata(
                    metadata_type, json.dumps(data, separators=(",", ":")), timestamp_ms, frame_id
                )
            published += 1
            if frames and published >= frames:
                stop.set()
    finally:
        stop.set()
        results.close()  # Runs the model's own cleanup, such as RF-DETR's bridge thread.
        source_run.stop()
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    elapsed = time.monotonic() - started
    print(f"published={published} results_fps={published / elapsed if elapsed else 0.0:.1f}", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return run(load_config(parse_args(argv).config))
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
