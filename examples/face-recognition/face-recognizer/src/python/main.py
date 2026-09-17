"""Face recognizer — Python implementation using pyneat.

Real-time SCRFD 2.5G face detection + ArcFace W600K R50 recognition on SiMa.ai Modalix.
Publishes H.264 video + object-detection metadata to an Insight viewer.

Pipeline:
  RTSP decode (NV12) --> branch --> video_sender (H264 RTP/UDP -> Insight)
                               \\--> SCRFD model --> [detect, align, ArcFace, match] --> metadata

Gallery:
  Build with:  python3 main.py --enroll --video clip.mp4 --name "Alice" [--gallery gallery.bin]
  Recognize:   python3 main.py --config path/to/config.yaml

Configuration is read from src/common/config.yaml; all CLI flags override individual keys.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import yaml

_DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "common" / "config.yaml"

# ── lazy imports (not available at build time) ────────────────────────────────
cv2 = None
np = None
pyneat = None


def _load_runtime_deps() -> None:
    """Import cv2, numpy and pyneat once at runtime."""
    import glob
    global cv2, np, pyneat
    if pyneat is not None:
        return
    for p in glob.glob("/usr/lib/python3*/dist-packages"):
        if p not in sys.path:
            sys.path.insert(0, p)
    # Force TCP for OpenCV's FFmpeg RTSP client.  It defaults to UDP, which drops
    # packets on a 720p high-fps stream — especially as a second subscriber
    # alongside the pyneat rtspsrc (which already uses TCP) — producing
    # "corrupted macroblock" decode errors and degraded alignment crops.
    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
    import cv2 as _cv2
    import numpy as _np
    import pyneat as _pyneat
    cv2 = _cv2
    np = _np
    pyneat = _pyneat


def _time_ms() -> float:
    return time.perf_counter() * 1000.0


# ── gallery binary format (v3) ────────────────────────────────────────────────
_GALLERY_MAGIC   = b"FRGAL1\n\x00"
_EMBEDDING_DIM   = 512
_ARCFACE_W       = 112
_ARCFACE_H       = 112

# Name of the combine() output carrying {frame, detections} for the same frame.
_JOINED_OUTPUT   = "recognizer_output"

# InsightFace canonical 112×112 landmark positions (left_eye, right_eye, nose, mouth L/R).
_ARCFACE_TEMPLATE = None  # initialized after numpy is loaded


def _arcface_template() -> "np.ndarray":
    global _ARCFACE_TEMPLATE
    if _ARCFACE_TEMPLATE is None:
        _ARCFACE_TEMPLATE = np.array(
            [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
             [41.5493, 92.3655], [70.7299, 92.2041]],
            dtype=np.float32,
        )
    return _ARCFACE_TEMPLATE


@dataclass
class GalleryEntry:
    name: str
    embedding: "np.ndarray"   # L2-normalized, shape (512,)
    raw_mean:  "np.ndarray"   # unnormalized mean, shape (512,)
    sample_count: int = 1


def _l2_normalize(v: "np.ndarray") -> "np.ndarray":
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-6 else v


def load_gallery(path: str) -> List[GalleryEntry]:
    """Read a gallery.bin written by the C++ GalleryBuilder (format v1–v3)."""
    entries: List[GalleryEntry] = []
    with open(path, "rb") as f:
        magic = f.read(8)
        if magic != _GALLERY_MAGIC:
            raise ValueError(f"load_gallery: invalid magic in {path}")
        (version,) = struct.unpack("<I", f.read(4))
        if version < 1 or version > 3:
            raise ValueError(f"load_gallery: unsupported version {version}")
        (n,) = struct.unpack("<I", f.read(4))
        for _ in range(n):
            (nl,) = struct.unpack("<H", f.read(2))
            name = f.read(nl).decode("utf-8")
            emb = np.frombuffer(f.read(_EMBEDDING_DIM * 4), dtype=np.float32).copy()
            sample_count = 1
            if version >= 2:
                (sample_count,) = struct.unpack("<I", f.read(4))
            raw_mean = emb.copy()
            if version >= 3:
                raw_mean = np.frombuffer(f.read(_EMBEDDING_DIM * 4), dtype=np.float32).copy()
            entries.append(GalleryEntry(name, emb, raw_mean, sample_count))
    return entries


def save_gallery(entries: List[GalleryEntry], path: str) -> None:
    """Write a v3 gallery.bin compatible with the C++ loader."""
    with open(path + ".tmp", "wb") as f:
        f.write(_GALLERY_MAGIC)
        f.write(struct.pack("<I", 3))          # version 3
        f.write(struct.pack("<I", len(entries)))
        for e in entries:
            nb = e.name.encode("utf-8")
            f.write(struct.pack("<H", len(nb)))
            f.write(nb)
            f.write(e.embedding.astype(np.float32).tobytes())
            f.write(struct.pack("<I", e.sample_count))
            f.write(e.raw_mean.astype(np.float32).tobytes())
        f.flush()
    import os
    os.replace(path + ".tmp", path)


# ── GalleryBuilder ─────────────────────────────────────────────────────────────

class GalleryBuilder:
    """Weighted-sum gallery accumulator — mirrors C++ GalleryBuilder."""

    def __init__(self) -> None:
        self._ws: dict[str, "np.ndarray"] = {}
        self._cnt: dict[str, int] = {}

    def add(self, name: str, raw_emb: "np.ndarray", count: int = 1) -> None:
        if name in self._ws:
            self._ws[name] += raw_emb * count
            self._cnt[name] += count
        else:
            self._ws[name] = raw_emb * float(count)
            self._cnt[name] = count

    def finish(self) -> List[GalleryEntry]:
        entries = []
        for name, ws in self._ws.items():
            cnt = self._cnt[name]
            raw_mean = ws / float(cnt)
            embedding = _l2_normalize(raw_mean.copy())
            entries.append(GalleryEntry(name, embedding, raw_mean, cnt))
        return entries


# ── face matching ──────────────────────────────────────────────────────────────

def match_embedding(
    emb: "np.ndarray",
    gallery: List[GalleryEntry],
    threshold: float,
    margin: float,
    unknown_label: str = "Unknown",
) -> Tuple[str, float]:
    if not gallery:
        return unknown_label, -1.0
    scores = [float(np.dot(emb, e.embedding)) for e in gallery]
    best_i = int(np.argmax(scores))
    best = scores[best_i]
    if best < threshold:
        return unknown_label, best
    if len(scores) > 1:
        second = sorted(scores, reverse=True)[1]
        if best - second < margin:
            return unknown_label, best
    return gallery[best_i].name, best


# ── SCRFD decode ───────────────────────────────────────────────────────────────

_SCRFD_STRIDES = (8, 16, 32)


def _sigmoid(x: "np.ndarray") -> "np.ndarray":
    return 1.0 / (1.0 + np.exp(-np.clip(x, -88.0, 88.0)))


def _make_grid(stride: int, fh: int, fw: int, num_anchors: int) -> "np.ndarray":
    """Anchor centre grid, shape (fh*fw*num_anchors, 2) in pixel coords."""
    ys = np.arange(fh, dtype=np.float32) * stride
    xs = np.arange(fw, dtype=np.float32) * stride
    grid_x, grid_y = np.meshgrid(xs, ys)
    cx = np.repeat(grid_x.reshape(-1), num_anchors)
    cy = np.repeat(grid_y.reshape(-1), num_anchors)
    return np.stack([cx, cy], axis=1)  # (fh*fw*na, 2)


def _nms_xyxy(boxes: "np.ndarray", scores: "np.ndarray", iou_thr: float, keep_top_k: int) -> List[int]:
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    kept = []
    while order.size > 0:
        i = int(order[0])
        kept.append(i)
        if keep_top_k > 0 and len(kept) >= keep_top_k:
            break
        ix1 = np.maximum(x1[i], x1[order[1:]])
        iy1 = np.maximum(y1[i], y1[order[1:]])
        ix2 = np.minimum(x2[i], x2[order[1:]])
        iy2 = np.minimum(y2[i], y2[order[1:]])
        iw = np.maximum(0.0, ix2 - ix1 + 1)
        ih = np.maximum(0.0, iy2 - iy1 + 1)
        inter = iw * ih
        iou = inter / (area[i] + area[order[1:]] - inter)
        order = order[1:][iou <= iou_thr]
    return kept


def decode_scrfd(
    tensors: list,
    conf_threshold: float,
    nms_iou: float,
    top_k: int,
    keep_top_k: int,
    infer_w: int,
    infer_h: int,
    num_anchors: int,
    pad_left: int,
    pad_top: int,
    scale: float,
) -> List[dict]:
    """
    Decode 9 NHWC tensors [cls×3, box×3, kps×3] at strides 8/16/32.
    Returns list of dicts: x1,y1,x2,y2,score,landmarks(shape 5×2).
    """
    cls_t = tensors[0:3]
    box_t = tensors[3:6]
    kps_t = tensors[6:9]

    all_boxes:  List["np.ndarray"] = []
    all_scores: List["np.ndarray"] = []
    all_lms:    List["np.ndarray"] = []

    for si, stride in enumerate(_SCRFD_STRIDES):
        fh = math.ceil(infer_h / stride)
        fw = math.ceil(infer_w / stride)
        grid = _make_grid(stride, fh, fw, num_anchors)  # (N, 2)

        cls_raw = np.asarray(cls_t[si]).reshape(-1)
        box_raw = np.asarray(box_t[si]).reshape(-1, 4)
        kps_raw = np.asarray(kps_t[si]).reshape(-1, 10)

        scores = _sigmoid(cls_raw)
        mask = scores > conf_threshold
        if not np.any(mask):
            continue

        anc = grid[mask]         # (M, 2): cx, cy
        sc  = scores[mask]
        bx  = box_raw[mask] * stride
        kp  = kps_raw[mask] * stride

        x1 = (anc[:, 0] - bx[:, 0] - pad_left) / scale
        y1 = (anc[:, 1] - bx[:, 1] - pad_top)  / scale
        x2 = (anc[:, 0] + bx[:, 2] - pad_left) / scale
        y2 = (anc[:, 1] + bx[:, 3] - pad_top)  / scale

        # Interleaved landmarks: [x0,y0, x1,y1, ... x4,y4]
        lm = np.empty((mask.sum(), 5, 2), dtype=np.float32)
        lm[:, :, 0] = (anc[:, 0:1] + kp[:, 0::2] - pad_left) / scale
        lm[:, :, 1] = (anc[:, 1:2] + kp[:, 1::2] - pad_top)  / scale

        all_boxes.append(np.stack([x1, y1, x2, y2], axis=1))
        all_scores.append(sc)
        all_lms.append(lm)

    if not all_boxes:
        return []

    boxes  = np.concatenate(all_boxes,  axis=0)
    scores = np.concatenate(all_scores, axis=0)
    lms    = np.concatenate(all_lms,    axis=0)

    order = scores.argsort()[::-1]
    if top_k > 0:
        order = order[:top_k]
    boxes  = boxes[order]
    scores = scores[order]
    lms    = lms[order]

    keep = _nms_xyxy(boxes, scores, nms_iou, keep_top_k)
    return [
        {
            "x1": float(boxes[i, 0]), "y1": float(boxes[i, 1]),
            "x2": float(boxes[i, 2]), "y2": float(boxes[i, 3]),
            "score": float(scores[i]),
            "landmarks": lms[i],  # shape (5, 2)
        }
        for i in keep
    ]


# ── face alignment ─────────────────────────────────────────────────────────────

def _similarity_transform_lsq(src: "np.ndarray", dst: "np.ndarray"):
    """Least-squares similarity transform (rotation+uniform scale+translation).

    Fits all five landmarks exactly, matching similarity_transform_lsq() in
    src/cpp/align.cpp.  A robust estimator such as estimateAffinePartial2D with
    LMEDS/RANSAC would discard landmarks as outliers, and with only five points
    that measurably degrades the crop and lowers ArcFace similarity.
    """
    n = len(src)
    x,  y  = src[:, 0], src[:, 1]
    dx, dy = dst[:, 0], dst[:, 1]
    ss   = float(np.sum(x * x + y * y))
    sx,  sy  = float(np.sum(x)),  float(np.sum(y))
    sdx, sdy = float(np.sum(dx)), float(np.sum(dy))
    sxdx = float(np.sum(x * dx + y * dy))
    sxdy = float(np.sum(x * dy - y * dx))

    inv_n = 1.0 / n
    S = ss - (sx * sx + sy * sy) * inv_n
    if abs(S) < 1e-12:
        return None
    a = (sxdx - (sx * sdx + sy * sdy) * inv_n) / S
    b = (sxdy - (sx * sdy - sy * sdx) * inv_n) / S
    c = (sdx - a * sx + b * sy) * inv_n
    d = (sdy - b * sx - a * sy) * inv_n
    return np.array([[a, -b, c],
                     [b,  a, d]], dtype=np.float64)


def align_face(bgr: "np.ndarray", landmarks: "np.ndarray") -> "np.ndarray":
    """
    Similarity-transform align a detected face to the 112×112 ArcFace template.
    landmarks: shape (5, 2) in original image pixel coords (x, y).
    Returns: 112×112 BGR uint8 aligned crop.
    """
    M = _similarity_transform_lsq(landmarks.astype(np.float64), _arcface_template())
    if M is None:
        # Fallback: direct bounding-box crop without alignment.
        return np.zeros((_ARCFACE_H, _ARCFACE_W, 3), dtype=np.uint8)
    return cv2.warpAffine(bgr, M, (_ARCFACE_W, _ARCFACE_H),
                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def preprocess_arcface(crop: "np.ndarray") -> "np.ndarray":
    """BGR uint8 112×112 → RGB float32 112×112 in [-1, 1]."""
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    return (rgb.astype(np.float32) / 127.5) - 1.0


# ── tensor helpers ─────────────────────────────────────────────────────────────

def _collect_tensors(sample) -> list:
    """Recursively collect all leaf Tensors from a pyneat Sample."""
    out = []
    if sample.kind == pyneat.SampleKind.Tensor and sample.tensor is not None:
        out.append(sample.tensor)
    elif sample.kind == pyneat.SampleKind.TensorSet:
        out.extend(sample.tensors)
    for f in sample.fields:
        out.extend(_collect_tensors(f))
    return out


def _tensor_to_np(t) -> "np.ndarray":
    return np.asarray(t.to_numpy(copy=False))


def _tensor_from_f32(arr: "np.ndarray"):
    """Wrap a contiguous float32 HWC numpy array as a pyneat EV74 tensor."""
    return pyneat.Tensor.from_numpy(
        np.ascontiguousarray(arr, dtype=np.float32),
        copy=True,
        layout=pyneat.TensorLayout.HWC,
        memory=pyneat.TensorMemory.EV74,
    )


def _find_field(sample, label: str):
    """Locate a named sub-sample inside a combined (Bundle) sample."""
    if getattr(sample, "stream_label", "") == label:
        return sample
    for f in getattr(sample, "fields", []):
        found = _find_field(f, label)
        if found is not None:
            return found
    return None


def _joined_field(sample, label: str, bundle_index: int):
    """Field `label` from a combine() output, falling back to positional index."""
    field = _find_field(sample, label)
    if field is not None:
        return field
    fields = list(getattr(sample, "fields", []))
    if getattr(sample, "kind", None) == pyneat.SampleKind.Bundle and len(fields) > bundle_index:
        return fields[bundle_index]
    raise RuntimeError(f"joined output missing '{label}' field")


def _first_tensor(sample):
    tensors = _collect_tensors(sample)
    if not tensors:
        raise RuntimeError("sample field carries no tensor")
    return tensors[0]


def _tensor_dim(t, name: str) -> int:
    v = getattr(t, name)
    return int(v() if callable(v) else v)


def _tensor_to_bgr(t) -> "np.ndarray":
    """Convert a decoded frame tensor (NV12/I420) from the graph to BGR."""
    w = _tensor_dim(t, "width")
    h = _tensor_dim(t, "height")
    need = w * h * 3 // 2
    if t.is_nv12():
        code = cv2.COLOR_YUV2BGR_NV12
    elif t.is_i420():
        code = cv2.COLOR_YUV2BGR_I420
    else:
        raise RuntimeError("decoded frame tensor is neither NV12 nor I420")
    payload = np.frombuffer(t.copy_payload_bytes(), dtype=np.uint8)
    if payload.size < need:
        raise RuntimeError(f"frame payload too small: {payload.size} < {need}")
    return np.ascontiguousarray(cv2.cvtColor(payload[:need].reshape((h * 3 // 2, w)), code))


# ── config ─────────────────────────────────────────────────────────────────────

@dataclass
class AppConfig:
    scrfd_model:   str = ""
    arcface_model: str = ""
    gallery_path:  str = "gallery.bin"
    input_uri:     str = ""
    insight_host:  str = ""
    insight_video_port:    int   = 9000
    insight_metadata_port: int   = 9100
    conf_threshold: float = 0.50
    nms_iou:        float = 0.40
    top_k:          int   = 5000
    keep_top_k:     int   = 5
    num_anchors:    int   = 2
    infer_w:        int   = 640
    infer_h:        int   = 640
    match_threshold: float = 0.55
    match_margin:    float = 0.12
    unknown_label:  str = "Unknown"
    recog_interval: int   = 8
    timeout_ms:     int   = 20000
    max_frames:     int   = 0


def _load_config(config_path: Path) -> AppConfig:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    pkg_root = config_path.resolve().parent.parent.parent  # face-recognizer/

    def resolve(p: str) -> str:
        if not p or Path(p).is_absolute() or "://" in p:
            return p
        return str((pkg_root / p).resolve())

    cfg = AppConfig()
    cfg.scrfd_model   = resolve(raw.get("scrfd", {}).get("model", ""))
    cfg.arcface_model = resolve(raw.get("arcface", {}).get("model", ""))
    cfg.gallery_path  = resolve(raw.get("gallery", {}).get("path", "gallery.bin"))
    cfg.input_uri     = raw.get("input", {}).get("uri", "")
    insight           = raw.get("output", {}).get("insight", {})
    cfg.insight_host          = insight.get("host", "")
    cfg.insight_video_port    = int(insight.get("video_port", 9000))
    cfg.insight_metadata_port = int(insight.get("metadata_port", 9100))
    scrfd = raw.get("scrfd", {})
    cfg.conf_threshold = float(scrfd.get("conf_threshold", cfg.conf_threshold))
    cfg.nms_iou        = float(scrfd.get("nms_iou", cfg.nms_iou))
    cfg.top_k          = int(scrfd.get("top_k", cfg.top_k))
    cfg.keep_top_k     = int(scrfd.get("keep_top_k", cfg.keep_top_k))
    cfg.num_anchors    = int(scrfd.get("num_anchors", cfg.num_anchors))
    match = raw.get("match", {})
    cfg.match_threshold = float(match.get("threshold", cfg.match_threshold))
    cfg.match_margin    = float(match.get("margin", cfg.match_margin))
    cfg.unknown_label   = match.get("unknown_label", cfg.unknown_label)
    runtime = raw.get("runtime", {})
    cfg.recog_interval = int(runtime.get("recog_interval", cfg.recog_interval))
    cfg.timeout_ms     = int(runtime.get("timeout_ms", cfg.timeout_ms))
    return cfg


# ── SCRFD pyneat pipeline build ────────────────────────────────────────────────

def _probe_rtsp(url: str) -> Tuple[int, int, int]:
    """Read width/height/fps from the stream caps before the graph is built."""
    # This short-lived probe joins the stream mid-GOP, so FFmpeg emits
    # "Missing reference picture" chatter while it waits for a keyframe.  Only
    # caps are read here, never pixels, so silence it; genuine failures still
    # surface as the explicit errors below.
    key  = "OPENCV_FFMPEG_CAPTURE_OPTIONS"
    prev = os.environ.get(key)
    os.environ[key] = (prev + "|" if prev else "") + "loglevel;quiet"
    try:
        cap = cv2.VideoCapture(url)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open RTSP source for probing: {url}")
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = round(cap.get(cv2.CAP_PROP_FPS) or 0)
        cap.release()
    finally:
        if prev is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = prev
    if w <= 0 or h <= 0:
        raise RuntimeError("RTSP probe: could not read frame dimensions")
    if fps <= 0:
        raise RuntimeError("RTSP probe: could not read FPS")
    return w, h, fps


def _build_scrfd_run(cfg: AppConfig, frame_w: int, frame_h: int, fps: int):
    """Build SCRFD pipeline: RTSP → branch → [VideoSender, SCRFD → output]."""
    opt = pyneat.ModelOptions()
    opt.preprocess.kind = pyneat.InputKind.Image
    opt.preprocess.enable = pyneat.AutoFlag.On
    opt.preprocess.color_convert.input_format = pyneat.PreprocessColorFormat.NV12
    opt.preprocess.input_max_width  = frame_w
    opt.preprocess.input_max_height = frame_h
    opt.preprocess.input_max_depth  = 3
    opt.preprocess.resize.enable    = pyneat.AutoFlag.On
    opt.preprocess.resize.mode      = pyneat.ResizeMode.Letterbox
    opt.preprocess.resize.pad_value = 0
    opt.preprocess.normalize.enable = pyneat.AutoFlag.On
    opt.preprocess.normalize.mean   = (0.0, 0.0, 0.0)
    opt.preprocess.normalize.stddev = (1.0, 1.0, 1.0)
    scrfd_model = pyneat.Model(cfg.scrfd_model, opt)

    rtsp_opt = pyneat.RtspDecodedInputOptions()
    rtsp_opt.url              = cfg.input_uri
    rtsp_opt.latency_ms       = 200
    rtsp_opt.tcp              = True
    rtsp_opt.payload_type     = 96
    rtsp_opt.insert_queue     = True
    rtsp_opt.decoder_name     = "decoder"
    rtsp_opt.decoder_raw_output = True
    rtsp_opt.auto_caps_from_stream = True
    rtsp_opt.fallback_h264_width   = frame_w
    rtsp_opt.fallback_h264_height  = frame_h
    rtsp_opt.fallback_h264_fps     = fps
    rtsp_opt.output_caps.enable    = True
    rtsp_opt.output_caps.format    = pyneat.Format.NV12
    rtsp_opt.output_caps.width     = frame_w
    rtsp_opt.output_caps.height    = frame_h
    rtsp_opt.output_caps.fps       = fps
    rtsp_opt.output_caps.memory    = pyneat.CapsMemory.Any

    source = pyneat.groups.rtsp_decoded_input(rtsp_opt)

    model_graph = pyneat.Graph("model")
    model_graph.connect(pyneat.nodes.input("model"), scrfd_model)

    det_graph = pyneat.Graph("detections")
    det_graph.add(pyneat.nodes.output("detections", pyneat.OutputOptions.every_frame(4)))

    # Tap the decoded frame off the same branch as the model input and join it to
    # the detections by frame.  This gives face alignment the exact frame SCRFD
    # saw, instead of an independently decoded one from a second RTSP client.
    frame_graph = pyneat.Graph("frame")
    frame_graph.add(pyneat.nodes.output("frame", pyneat.OutputOptions.every_frame(4)))
    joined = pyneat.graphs.combine(
        ["frame", "detections"], _JOINED_OUTPUT, pyneat.CombinePolicy.ByFrame
    )

    g_opts = pyneat.GraphOptions()
    g_opts.advanced_execution.postprocess_target = "EV74"
    graph = pyneat.Graph(g_opts)

    video_opt = None
    branch_names = ["video", "model", "frame"] if cfg.insight_host else ["model", "frame"]
    branch = pyneat.graphs.branch("source", branch_names)
    graph.connect(source, branch)

    if cfg.insight_host:
        _vopt = pyneat.VideoSenderOptions.h264_rtp_udp_from_raw(frame_w, frame_h, fps)
        _vopt.host            = cfg.insight_host
        _vopt.channel         = 0
        _vopt.video_port_base = cfg.insight_video_port
        _vopt.encoder.bitrate_kbps = 4000
        video_graph = pyneat.Graph("video")
        video_graph.connect(pyneat.nodes.input("video"), pyneat.groups.video_sender(_vopt))
        graph.connect(branch, video_graph)
        video_opt = _vopt

    graph.connect(branch, model_graph)
    graph.connect(branch, frame_graph)
    graph.connect(model_graph, det_graph)
    graph.connect(frame_graph, joined)
    graph.connect(det_graph, joined)

    run_opts = pyneat.RunOptions()
    run_opts.preset          = pyneat.RunPreset.Realtime
    run_opts.queue_depth     = 3
    run_opts.overflow_policy = pyneat.OverflowPolicy.KeepLatest
    run_opts.output_memory   = pyneat.OutputMemory.ZeroCopy
    run = graph.build(run_opts)

    return run, video_opt


def _build_arcface_run(cfg: AppConfig, dummy_crop: "np.ndarray"):
    """Build ArcFace pipeline: tensor input → ArcFace → output."""
    opt = pyneat.ModelOptions()
    opt.preprocess.kind             = pyneat.InputKind.Tensor
    opt.preprocess.input_max_width  = _ARCFACE_W
    opt.preprocess.input_max_height = _ARCFACE_H
    opt.preprocess.input_max_depth  = 3
    model = pyneat.Model(cfg.arcface_model, opt)

    graph = pyneat.Graph()
    graph.add(pyneat.nodes.input(model.input_appsrc_options(True)))
    graph.add(model)
    graph.add(pyneat.nodes.output())

    run_opts = pyneat.RunOptions()
    run_opts.output_memory = pyneat.OutputMemory.Owned
    dummy = _tensor_from_f32(dummy_crop)
    return graph.build([dummy], run_opts)


# ── enrollment ─────────────────────────────────────────────────────────────────

def run_enrollment(
    cfg: AppConfig,
    video_path: str,
    name: str,
    gallery_out: str,
    sample_every: int,
    min_score: float,
) -> int:
    """Enroll faces from a video into gallery_out (appends to existing gallery)."""
    builder = GalleryBuilder()

    if Path(gallery_out).exists():
        existing = load_gallery(gallery_out)
        for e in existing:
            builder.add(e.name, e.raw_mean, e.sample_count)
        print(f"[GALLERY] Loaded {len(existing)} existing identity(ies) from {gallery_out}")

    dummy_crop = np.zeros((_ARCFACE_H, _ARCFACE_W, 3), dtype=np.float32)
    scrfd_opt = pyneat.ModelOptions()
    scrfd_opt.preprocess.kind             = pyneat.InputKind.Tensor
    scrfd_opt.preprocess.input_max_width  = cfg.infer_w
    scrfd_opt.preprocess.input_max_height = cfg.infer_h
    scrfd_opt.preprocess.input_max_depth  = 3
    scrfd_model = pyneat.Model(cfg.scrfd_model, scrfd_opt)

    scrfd_graph = pyneat.Graph()
    scrfd_graph.add(pyneat.nodes.input(scrfd_model.input_appsrc_options(True)))
    scrfd_graph.add(scrfd_model)
    scrfd_graph.add(pyneat.nodes.output())
    dummy_frame = np.zeros((cfg.infer_h, cfg.infer_w, 3), dtype=np.float32)
    scrfd_run = scrfd_graph.build([_tensor_from_f32(dummy_frame)])
    arc_run   = _build_arcface_run(cfg, dummy_crop)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    print(f"[VIDEO] {video_path} — {total} frames @ {fps:.1f} fps, sample_every={sample_every}")

    enrolled = skipped = frame_idx = 0
    while True:
        ret, bgr = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % sample_every != 0:
            continue

        # Preprocess for SCRFD
        scale = min(cfg.infer_w / bgr.shape[1], cfg.infer_h / bgr.shape[0])
        nw = round(bgr.shape[1] * scale)
        nh = round(bgr.shape[0] * scale)
        pad_l = (cfg.infer_w - nw) // 2
        pad_t = (cfg.infer_h - nh) // 2
        resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((cfg.infer_h, cfg.infer_w, 3), dtype=np.float32)
        rgb_resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        canvas[pad_t:pad_t+nh, pad_l:pad_l+nw] = rgb_resized

        if not scrfd_run.push([_tensor_from_f32(canvas)]):
            raise RuntimeError("SCRFD push failed: " + scrfd_run.last_error())
        sample = scrfd_run.pull(timeout_ms=cfg.timeout_ms)
        if sample is None:
            raise RuntimeError(f"SCRFD timeout at frame {frame_idx}")

        tensors = [_tensor_to_np(t) for t in _collect_tensors(sample)]
        dets = decode_scrfd(tensors, cfg.conf_threshold, cfg.nms_iou,
                            cfg.top_k, cfg.keep_top_k, cfg.infer_w, cfg.infer_h,
                            cfg.num_anchors, pad_l, pad_t, scale)
        if not dets:
            skipped += 1
            continue

        best = max(dets, key=lambda d: d["score"])
        if best["score"] < min_score:
            print(f"  [skip] frame {frame_idx} score={best['score']:.3f} < {min_score}")
            skipped += 1
            continue

        crop = align_face(bgr, best["landmarks"])
        f32  = preprocess_arcface(crop)

        if not arc_run.push([_tensor_from_f32(f32)]):
            raise RuntimeError("ArcFace push failed: " + arc_run.last_error())
        arc_sample = arc_run.pull(timeout_ms=cfg.timeout_ms)
        if arc_sample is None:
            raise RuntimeError(
                f"ArcFace timeout at frame {frame_idx} — pipeline stalled, "
                "cannot continue enrollment safely"
            )

        emb_np = _tensor_to_np(_collect_tensors(arc_sample)[0]).flatten().astype(np.float32)
        emb_np = _l2_normalize(emb_np)
        builder.add(name, emb_np)
        enrolled += 1
        print(f"  [{name}] frame {frame_idx} → enrolled (score={best['score']:.3f}, total={enrolled})")

    cap.release()
    scrfd_run.close()
    arc_run.close()

    if enrolled == 0:
        print("ERROR: no faces enrolled — check video contains visible faces", file=sys.stderr)
        return 2

    entries = builder.finish()
    save_gallery(entries, gallery_out)
    print(f"\nEnrollment complete: {enrolled} enrolled, {skipped} skipped")
    print(f"Gallery saved: {gallery_out} ({len(entries)} identity(ies))")
    return 0


# ── recognition pipeline ───────────────────────────────────────────────────────

def _letterbox_params(orig_w: int, orig_h: int, infer_w: int, infer_h: int):
    scale = min(infer_w / orig_w, infer_h / orig_h)
    nw = round(orig_w * scale)
    nh = round(orig_h * scale)
    pad_l = (infer_w - nw) // 2
    pad_t = (infer_h - nh) // 2
    return scale, pad_l, pad_t


def run_recognition(cfg: AppConfig, gallery: List[GalleryEntry], max_frames: int,
                    test_mode: bool = False) -> int:
    """Main recognition loop: RTSP → SCRFD → ArcFace → MetadataSender → Insight."""
    frame_w, frame_h, fps = _probe_rtsp(cfg.input_uri)
    scale, pad_l, pad_t = _letterbox_params(frame_w, frame_h, cfg.infer_w, cfg.infer_h)

    # Build pipelines
    scrfd_run, video_opt = _build_scrfd_run(cfg, frame_w, frame_h, fps)

    dummy_crop = np.zeros((_ARCFACE_H, _ARCFACE_W, 3), dtype=np.float32)
    arc_run = _build_arcface_run(cfg, dummy_crop)

    metadata_sender = None
    if cfg.insight_host:
        meta_opt = pyneat.MetadataSenderOptions()
        meta_opt.host               = cfg.insight_host
        meta_opt.channel            = 0
        meta_opt.metadata_port_base = cfg.insight_metadata_port
        metadata_sender = pyneat.MetadataSender(meta_opt)

    if cfg.insight_host:
        print(
            f"rtsp={cfg.input_uri} stream={frame_w}x{frame_h}@{fps} "
            f"insight={cfg.insight_host} video={video_opt.video_port} "
            f"metadata={metadata_sender.metadata_port()} channel=0"
        )
    else:
        print(f"rtsp={cfg.input_uri} stream={frame_w}x{frame_h}@{fps} headless")

    cached_labels:    List[str]              = []
    cached_sims:      List[float]            = []
    cached_centroids: List[Tuple[float, float]] = []  # box centres from last recog
    last_recog_frame = -cfg.recog_interval  # force recog on frame 0

    frame_count = 0
    processed   = 0
    start_ms    = _time_ms()

    while max_frames <= 0 or frame_count < max_frames:
        # Pull the joined {frame, detections} sample — both come from the same
        # decoded frame, so landmarks always match the image they are cropped from.
        sample = scrfd_run.pull(_JOINED_OUTPUT, cfg.timeout_ms)
        if sample is None:
            err = str(scrfd_run.last_error() or "")
            if err:
                raise RuntimeError(f"SCRFD pull failed: {err}")
            if not scrfd_run.running():
                break
            print("[warn] SCRFD pull timeout", file=sys.stderr, flush=True)
            continue

        ts_ms    = int(sample.pts_ns // 1_000_000) if sample.pts_ns >= 0 else -1
        frame_id = getattr(sample, "frame_id", frame_count)

        det_field   = _joined_field(sample, "detections", 1)
        frame_field = _joined_field(sample, "frame", 0)

        tensors = [_tensor_to_np(t) for t in _collect_tensors(det_field)]
        dets = decode_scrfd(
            tensors, cfg.conf_threshold, cfg.nms_iou,
            cfg.top_k, cfg.keep_top_k, cfg.infer_w, cfg.infer_h,
            cfg.num_anchors, pad_l, pad_t, scale,
        )

        # Run recognition at recog_interval, or immediately when detections reorder.
        # Detection reorder: SCRFD sorts by score; if two faces' scores cross between
        # recog runs, index 0 and 1 swap, assigning the wrong cached label to each face.
        # Detect this by checking whether each current box centre is still closest to
        # the same-index cached position, and force a fresh recognition if not.
        def _reordered(cur_dets, centroids):
            for i, d in enumerate(cur_dets):
                cx = (d["x1"] + d["x2"]) / 2
                cy = (d["y1"] + d["y2"]) / 2
                d2_same = (cx - centroids[i][0]) ** 2 + (cy - centroids[i][1]) ** 2
                # Displacement check: if the face moved more than one face-width it is
                # almost certainly a different person, even when len(centroids) == 1
                # (where the any() below would produce an empty sequence and miss it).
                face_w = d["x2"] - d["x1"]
                if d2_same > face_w * face_w:
                    return True
                if any((cx - centroids[j][0]) ** 2 + (cy - centroids[j][1]) ** 2 < d2_same
                       for j in range(len(centroids)) if j != i):
                    return True
            return False

        need_recog = bool(dets) and (
            (frame_count - last_recog_frame) >= cfg.recog_interval
            or len(dets) != len(cached_labels)
            or (cached_centroids and _reordered(dets, cached_centroids))
        )
        if need_recog:
            # NV12→BGR only on frames that actually re-embed (≈1 in recog_interval).
            bgr = _tensor_to_bgr(_first_tensor(frame_field))
            cached_labels    = []
            cached_sims      = []
            cached_centroids = []
            for det in dets:
                crop = align_face(bgr, det["landmarks"])
                f32  = preprocess_arcface(crop)
                if not arc_run.push([_tensor_from_f32(f32)]):
                    raise RuntimeError("ArcFace push failed: " + arc_run.last_error())
                arc_sample = arc_run.pull(timeout_ms=cfg.timeout_ms)
                if arc_sample is None:
                    raise RuntimeError("ArcFace timeout during recognition — pipeline stalled")
                emb = _tensor_to_np(_collect_tensors(arc_sample)[0]).flatten().astype(np.float32)
                emb = _l2_normalize(emb)
                label, sim = match_embedding(
                    emb, gallery,
                    cfg.match_threshold, cfg.match_margin, cfg.unknown_label,
                )
                cached_labels.append(label)
                cached_sims.append(sim)
                cached_centroids.append(((det["x1"] + det["x2"]) / 2,
                                         (det["y1"] + det["y2"]) / 2))
            last_recog_frame = frame_count
        elif not dets or len(dets) != len(cached_labels):
            cached_labels    = []
            cached_sims      = []
            cached_centroids = []

        # Per-frame console output in test mode (mirrors C++ --test output)
        if test_mode and dets:
            for i, det in enumerate(dets):
                label = cached_labels[i] if i < len(cached_labels) else cfg.unknown_label
                sim   = cached_sims[i]   if i < len(cached_sims)   else -1.0
                print(f"  face[{i}] → {label}  similarity={sim:.4f}")

        # Publish metadata to Insight
        if cfg.insight_host and metadata_sender:
            fw = float(frame_w)
            fh = float(frame_h)
            objects = []
            for i, det in enumerate(dets):
                label = cached_labels[i] if i < len(cached_labels) else cfg.unknown_label
                sim   = cached_sims[i]   if i < len(cached_sims)   else -1.0
                cx1 = max(0.0, min(det["x1"], fw))
                cy1 = max(0.0, min(det["y1"], fh))
                cx2 = max(cx1, min(det["x2"], fw))
                cy2 = max(cy1, min(det["y2"], fh))
                objects.append({
                    "id":         f"face_{i + 1}",
                    "label":      label,
                    "confidence": float(det["score"]),
                    "similarity": float(sim),
                    "bbox":       [cx1, cy1, cx2 - cx1, cy2 - cy1],
                })
            data_json = json.dumps({"objects": objects}, separators=(",", ":"))
            metadata_sender.send_metadata(
                "object-detection", data_json, ts_ms, str(frame_id)
            )

        frame_count += 1
        processed   += 1

    elapsed_s = (_time_ms() - start_ms) / 1000.0
    fps_out = processed / elapsed_s if elapsed_s > 0 else 0.0
    print(f"processed={processed} fps={fps_out:.1f}")

    scrfd_run.close()
    arc_run.close()
    return 0


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse_args(argv):
    p = argparse.ArgumentParser(
        description="Face recognizer — Python (SCRFD + ArcFace on SiMa.ai Modalix)"
    )
    p.add_argument("--config",       type=Path,  default=_DEFAULT_CONFIG)
    p.add_argument("--input",        type=str,   default=None)
    p.add_argument("--gallery",      type=str,   default=None)
    p.add_argument("--scrfd-model",  type=str,   default=None)
    p.add_argument("--arcface-model",type=str,   default=None)
    p.add_argument("--max-frames",   type=int,   default=0)
    p.add_argument("--test",         action="store_true")
    p.add_argument("--enroll",       action="store_true")
    p.add_argument("--video",        type=str,   default=None)
    p.add_argument("--name",         type=str,   default=None)
    p.add_argument("--sample-every", type=int,   default=5)
    p.add_argument("--min-score",    type=float, default=0.75)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)

    try:
        cfg = _load_config(args.config)
    except Exception as exc:
        print(f"[ERR] Config error: {exc}", file=sys.stderr)
        return 1

    # CLI overrides
    if args.input:        cfg.input_uri     = args.input
    if args.gallery:      cfg.gallery_path  = args.gallery
    if args.scrfd_model:  cfg.scrfd_model   = args.scrfd_model
    if args.arcface_model:cfg.arcface_model = args.arcface_model
    if args.max_frames:   cfg.max_frames    = args.max_frames

    try:
        _load_runtime_deps()
    except Exception as exc:
        print(f"[ERR] Failed to load runtime dependencies: {exc}", file=sys.stderr)
        return 1

    if args.enroll:
        if not args.video:
            print("[ERR] --enroll requires --video <path>", file=sys.stderr)
            return 1
        if not args.name:
            print("[ERR] --enroll requires --name <identity>", file=sys.stderr)
            return 1
        gallery_out = args.gallery or cfg.gallery_path
        try:
            return run_enrollment(
                cfg, args.video, args.name, gallery_out,
                args.sample_every, args.min_score,
            )
        except Exception as exc:
            print(f"[ERR] Enrollment failed: {exc}", file=sys.stderr)
            return 2

    # Recognition mode
    if not cfg.input_uri:
        print("[ERR] input.uri not set — provide in config.yaml or via --input", file=sys.stderr)
        return 1
    if not cfg.scrfd_model:
        print("[ERR] scrfd.model not set", file=sys.stderr)
        return 1
    if not cfg.arcface_model:
        print("[ERR] arcface.model not set", file=sys.stderr)
        return 1

    gallery: List[GalleryEntry] = []
    gallery_path = Path(cfg.gallery_path) if cfg.gallery_path else None
    if gallery_path and gallery_path.exists():
        try:
            gallery = load_gallery(cfg.gallery_path)
            print(f"[GALLERY] Loaded {len(gallery)} identity(ies) from {cfg.gallery_path}")
        except Exception as exc:
            print(f"[ERR] Gallery exists but cannot be read: {exc}", file=sys.stderr)
            return 1
    else:
        print("[WARN] No gallery loaded — all faces will be labelled Unknown", file=sys.stderr)

    try:
        return run_recognition(cfg, gallery, cfg.max_frames, test_mode=args.test)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"[ERR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
