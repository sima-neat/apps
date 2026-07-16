from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import yaml


@dataclass
class Config:
    model_path: str = ""
    clip_image_path: str = ""
    clip_text_path: str = ""
    clip_consts_path: str = ""
    rtsp_urls: List[str] = field(default_factory=list)
    text: str = ""
    insight_host: str = ""
    infer_size: int = 640
    score_threshold: float = 0.7
    nms_iou: float = 0.9
    max_detections: int = 300
    max_crops: int = 0
    min_score: float = 0.65
    max_box_frac: float = 0.8
    latency_ms: int = 200
    tcp: bool = True
    timeout_ms: int = 20000
    queue_depth: int = 8
    frames: int = 0
    video_port_base: int = 9000
    metadata_port_base: int = 9100
    bitrate_kbps: int = 1000


def _get(raw, dotted, default):
    node = raw
    for key in dotted.split("."):
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return default if node is None else node


def load_config(path):
    if not Path(path).is_file():
        raise RuntimeError("config file not found: " + str(path))
    raw = yaml.safe_load(Path(path).read_text()) or {}

    cfg = Config(
        model_path=str(_get(raw, "model.path", "")),
        clip_image_path=str(_get(raw, "clip.image_encoder_path", "")),
        clip_text_path=str(_get(raw, "clip.text_encoder_path", "")),
        clip_consts_path=str(_get(raw, "clip.text_host_consts", "")),
        rtsp_urls=[str(url) for url in _get(raw, "source.rtsp_urls", []) if url],
        text=str(_get(raw, "prompt.text", "")),
        insight_host=str(_get(raw, "output.insight.host", "")),
        infer_size=int(_get(raw, "runtime.infer_size", 640)),
        score_threshold=float(_get(raw, "decode.score_threshold", 0.7)),
        nms_iou=float(_get(raw, "decode.nms_iou", 0.9)),
        max_detections=int(_get(raw, "decode.max_detections", 300)),
        max_crops=int(_get(raw, "clip.max_crops", 0)),
        min_score=float(_get(raw, "clip.min_score", 0.65)),
        max_box_frac=float(_get(raw, "clip.max_box_frac", 0.8)),
        latency_ms=int(_get(raw, "source.latency_ms", 200)),
        tcp=bool(_get(raw, "source.tcp", True)),
        queue_depth=int(_get(raw, "runtime.queue_depth", 8)),
        frames=int(_get(raw, "runtime.frames", 0)),
        video_port_base=int(_get(raw, "output.insight.video_port_base", 9000)),
        metadata_port_base=int(_get(raw, "output.insight.metadata_port_base", 9100)),
        bitrate_kbps=int(_get(raw, "output.insight.bitrate_kbps", 1000)),
    )

    required = [
        (cfg.model_path, "model.path must be set"),
        (cfg.clip_image_path, "clip.image_encoder_path must be set"),
        (cfg.clip_text_path, "clip.text_encoder_path must be set"),
        (cfg.clip_consts_path, "clip.text_host_consts must be set"),
        (cfg.rtsp_urls, "source.rtsp_urls must be a non-empty list of RTSP URLs"),
        (cfg.text, "prompt.text must be set"),
        (cfg.insight_host, "output.insight.host must be set"),
    ]
    for ok, message in required:
        if not ok:
            raise RuntimeError(message)
    return cfg
