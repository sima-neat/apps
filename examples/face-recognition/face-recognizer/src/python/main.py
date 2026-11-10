"""Face recognizer: SCRFD finds the faces in a frame, ArcFace turns each face into an
embedding, and the embedding is matched against a gallery of enrolled people.

Recognition reads an RTSP stream, a USB camera or an H.264 video file and reports each
face's box and name to Insight. --enroll builds the gallery from images or a video:

  Enroll:     python3 main.py --enroll --images gallery_images/
              python3 main.py --enroll --video clip.mp4 --name "Alice"
  Recognize:  python3 main.py --config path/to/config.yaml

Settings come from src/common/config.yaml; command-line flags override them.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import time
from dataclasses import dataclass
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
# SCRFD 2.5G: 640x640 input, two anchors per cell.
_SCRFD_W         = 640
_SCRFD_H         = 640
_SCRFD_ANCHORS   = 2

# Name of the combine() output carrying {frame, detections} for the same frame.
_JOINED_OUTPUT   = "face_output"

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
    pad_left: int,
    pad_top: int,
    scale: float,
) -> List[dict]:
    """
    Decode 9 NHWC tensors [cls×3, box×3, kps×3] at strides 8/16/32.
    Returns list of dicts: x1,y1,x2,y2,score,landmarks(shape 5×2), highest score first.
    """
    cls_t = tensors[0:3]
    box_t = tensors[3:6]
    kps_t = tensors[6:9]

    all_boxes:  List["np.ndarray"] = []
    all_scores: List["np.ndarray"] = []
    all_lms:    List["np.ndarray"] = []

    for si, stride in enumerate(_SCRFD_STRIDES):
        fh = _SCRFD_H // stride
        fw = _SCRFD_W // stride
        grid = _make_grid(stride, fh, fw, _SCRFD_ANCHORS)  # (N, 2)

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
    """Convert a decoded frame tensor (NV12/I420, or NV16 from a USB camera) to BGR."""
    w = _tensor_dim(t, "width")
    h = _tensor_dim(t, "height")
    if t.image_format() == pyneat.PixelFormat.NV16:
        # A 4:2:2 USB camera frame. The payload is the whole mapped buffer, so the Y and UV
        # planes are read at their own offsets and strides; each row keeps its own chroma by
        # interleaving Y and UV into packed 4:2:2 (Y0 U0 Y1 V0).
        payload = np.frombuffer(t.copy_payload_bytes(), dtype=np.uint8)
        planes = {plane.role: plane for plane in t.planes}

        def plane_rows(role):
            plane = planes[role]
            stride = plane.strides_bytes[0]
            end = plane.byte_offset + stride * h
            if end > payload.size:
                raise RuntimeError(f"NV16 payload smaller than its {role} plane")
            return payload[plane.byte_offset:end].reshape((h, stride))[:, :w]

        yuy2 = np.dstack([plane_rows(pyneat.PlaneRole.Y), plane_rows(pyneat.PlaneRole.UV)])
        return np.ascontiguousarray(cv2.cvtColor(yuy2, cv2.COLOR_YUV2BGR_YUY2))
    need = w * h * 3 // 2
    if t.is_nv12():
        code = cv2.COLOR_YUV2BGR_NV12
    elif t.is_i420():
        code = cv2.COLOR_YUV2BGR_I420
    else:
        raise RuntimeError("decoded frame tensor is neither NV12 nor I420")
    # The decoder may pad planes (e.g. Y padded to 768 rows for 720p); repack tightly.
    payload = np.frombuffer(t.contiguous().copy_payload_bytes(), dtype=np.uint8)
    if payload.size < need:
        raise RuntimeError(f"frame payload too small: {payload.size} < {need}")
    return np.ascontiguousarray(cv2.cvtColor(payload[:need].reshape((h * 3 // 2, w)), code))


# ── config ─────────────────────────────────────────────────────────────────────

@dataclass
class AppConfig:
    scrfd_model:   str = ""
    arcface_model: str = ""
    gallery_path:  str = "gallery.bin"
    input_uri:     str = ""   # RTSP URL or H.264 video file
    usb_camera:    str = ""   # a usb_camera_ids() ID; used when input_uri is empty
    usb_width:     int = 1280
    usb_height:    int = 720
    usb_fps:       int = 30
    insight_host:  str = ""   # empty = no Insight output
    insight_video_port:    int = 9000
    insight_metadata_port: int = 9100
    conf_threshold:  float = 0.65
    nms_iou:         float = 0.40
    top_k:           int   = 5000
    keep_top_k:      int   = 100
    match_threshold: float = 0.55
    match_margin:    float = 0.12
    unknown_label:   str   = "Unknown"
    timeout_ms:      int   = 20000
    max_frames:      int   = 0   # 0 = unlimited


def _load_config(config_path: Path) -> AppConfig:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    # Relative paths in config.yaml are relative to the example directory.
    example_dir = config_path.resolve().parent.parent.parent

    def resolve(p: str) -> str:
        if not p or Path(p).is_absolute() or "://" in p:
            return p
        return str((example_dir / p).resolve())

    cfg = AppConfig()
    scrfd   = raw.get("scrfd", {}) or {}
    usb     = (raw.get("input", {}) or {}).get("usb", {}) or {}
    insight = (raw.get("output", {}) or {}).get("insight", {}) or {}
    match   = raw.get("match", {}) or {}
    cfg.scrfd_model   = resolve(scrfd.get("model", ""))
    cfg.arcface_model = resolve((raw.get("arcface", {}) or {}).get("model", ""))
    cfg.gallery_path  = resolve((raw.get("gallery", {}) or {}).get("path", cfg.gallery_path))
    cfg.input_uri     = resolve((raw.get("input", {}) or {}).get("uri", "") or "")
    cfg.usb_camera    = str(usb.get("camera", "") or "")
    cfg.usb_width     = int(usb.get("width", cfg.usb_width))
    cfg.usb_height    = int(usb.get("height", cfg.usb_height))
    cfg.usb_fps       = int(usb.get("fps", cfg.usb_fps))
    cfg.insight_host          = insight.get("host", "") or ""
    cfg.insight_video_port    = int(insight.get("video_port", cfg.insight_video_port))
    cfg.insight_metadata_port = int(insight.get("metadata_port", cfg.insight_metadata_port))
    cfg.conf_threshold  = float(scrfd.get("conf_threshold", cfg.conf_threshold))
    cfg.nms_iou         = float(scrfd.get("nms_iou", cfg.nms_iou))
    cfg.top_k           = int(scrfd.get("top_k", cfg.top_k))
    cfg.keep_top_k      = int(scrfd.get("keep_top_k", cfg.keep_top_k))
    cfg.match_threshold = float(match.get("threshold", cfg.match_threshold))
    cfg.match_margin    = float(match.get("margin", cfg.match_margin))
    cfg.unknown_label   = match.get("unknown_label", cfg.unknown_label)
    cfg.timeout_ms      = int((raw.get("runtime", {}) or {}).get("timeout_ms", cfg.timeout_ms))
    return cfg


# ── models ─────────────────────────────────────────────────────────────────────

def _build_tensor_run(model_path: str, width: int, height: int):
    """A model fed one float tensor at a time: ArcFace (a 112x112 aligned face) and, for
    the still images of enrollment, SCRFD (a 640x640 letterboxed image)."""
    opt = pyneat.ModelOptions()
    opt.preprocess.kind             = pyneat.InputKind.Tensor
    opt.preprocess.input_max_width  = width
    opt.preprocess.input_max_height = height
    opt.preprocess.input_max_depth  = 3
    model = pyneat.Model(model_path, opt)

    graph = pyneat.Graph()
    graph.add(pyneat.nodes.input(model.input_appsrc_options(True)))
    graph.add(model)
    graph.add(pyneat.nodes.output())

    run_opts = pyneat.RunOptions()
    run_opts.output_memory = pyneat.OutputMemory.Owned
    seed = np.zeros((height, width, 3), dtype=np.float32)
    return graph.build([_tensor_from_f32(seed)], run_opts)


def _run_tensor(run, array: "np.ndarray", timeout_ms: int, what: str) -> list:
    if not run.push([_tensor_from_f32(array)]):
        raise RuntimeError(f"{what} push failed: {run.last_error()}")
    sample = run.pull(timeout_ms=timeout_ms)
    # A result still in flight would come back for the next input, so a timeout is fatal.
    if sample is None:
        raise RuntimeError(f"{what} timed out")
    return [_tensor_to_np(t) for t in _collect_tensors(sample)]


def embed_face(arcface_run, bgr: "np.ndarray", landmarks: "np.ndarray",
               timeout_ms: int) -> "np.ndarray":
    """The one path from a detected face to its embedding, shared by enrollment and
    recognition: align to the ArcFace template, run ArcFace, L2-normalize."""
    crop = align_face(bgr, landmarks)
    out = _run_tensor(arcface_run, preprocess_arcface(crop), timeout_ms, "ArcFace")
    return _l2_normalize(out[0].flatten().astype(np.float32))


def _letterbox(frame_w: int, frame_h: int) -> Tuple[float, int, int]:
    """How a frame is letterboxed into the SCRFD input: scale, left pad, top pad."""
    scale = min(_SCRFD_W / frame_w, _SCRFD_H / frame_h)
    pad_l = (_SCRFD_W - round(frame_w * scale)) // 2
    pad_t = (_SCRFD_H - round(frame_h * scale)) // 2
    return scale, pad_l, pad_t


def preprocess_scrfd(bgr: "np.ndarray") -> "np.ndarray":
    """Letterbox a BGR image to the SCRFD input (black bars, RGB, [0,1] float32), the same
    preprocessing the EV74 runs for decoded video frames. Used for still images."""
    scale, pad_l, pad_t = _letterbox(bgr.shape[1], bgr.shape[0])
    nw, nh = round(bgr.shape[1] * scale), round(bgr.shape[0] * scale)
    resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((_SCRFD_H, _SCRFD_W, 3), dtype=np.float32)
    canvas[pad_t:pad_t + nh, pad_l:pad_l + nw] = (
        cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    )
    return canvas


# ── enrollment ─────────────────────────────────────────────────────────────────

_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")


def run_enrollment(cfg: AppConfig, images_dir: Optional[str], video_path: Optional[str],
                   name: Optional[str], sample_every: int, min_score: float,
                   max_per_person: int) -> int:
    """Add people to the gallery from a folder of images (one subfolder per person) or
    from a video of one person."""
    if not images_dir and not video_path:
        raise RuntimeError("--enroll needs --images <dir> or --video <file> --name <person>")
    if video_path and not name:
        raise RuntimeError("--video needs --name <person>")
    if name and not video_path:
        raise RuntimeError("--name applies to --video; --images names people by folder")
    if images_dir and not Path(images_dir).is_dir():
        raise RuntimeError(f"not a directory: {images_dir}")
    if sample_every <= 0:
        raise RuntimeError("--sample-every must be positive")

    print("[BUILD] Building SCRFD and ArcFace pipelines...")
    scrfd_run   = _build_tensor_run(cfg.scrfd_model, _SCRFD_W, _SCRFD_H)
    arcface_run = _build_tensor_run(cfg.arcface_model, _ARCFACE_W, _ARCFACE_H)

    # New samples are added to the people already in the gallery. An existing gallery
    # that cannot be read stops enrollment rather than being replaced.
    builder = GalleryBuilder()
    if Path(cfg.gallery_path).exists():
        for entry in load_gallery(cfg.gallery_path):
            builder.add(entry.name, entry.raw_mean, entry.sample_count)

    counts = {"enrolled": 0, "skipped": 0}

    def enroll_image(person: str, bgr: "np.ndarray", what: str) -> None:
        """Enroll the highest-scoring face of one image under `person`."""
        scale, pad_l, pad_t = _letterbox(bgr.shape[1], bgr.shape[0])
        tensors = _run_tensor(scrfd_run, preprocess_scrfd(bgr), cfg.timeout_ms, "SCRFD")
        faces = decode_scrfd(tensors, cfg.conf_threshold, cfg.nms_iou, cfg.top_k,
                             cfg.keep_top_k, pad_l, pad_t, scale)
        if not faces or faces[0]["score"] < min_score:
            print(f"  [skip] {what}: no face scoring at least {min_score}")
            counts["skipped"] += 1
            return
        builder.add(person, embed_face(arcface_run, bgr, faces[0]["landmarks"], cfg.timeout_ms))
        counts["enrolled"] += 1
        print(f"  [{person}] {what} (score {faces[0]['score']:.6f})")

    if images_dir:
        for person_dir in sorted(p for p in Path(images_dir).iterdir() if p.is_dir()):
            images = sorted(p for p in person_dir.iterdir()
                            if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES)
            if max_per_person > 0:
                images = images[:max_per_person]
            for image in images:
                bgr = cv2.imread(str(image), cv2.IMREAD_COLOR)
                if bgr is None:
                    print(f"  [skip] cannot read {image.name}")
                    counts["skipped"] += 1
                    continue
                enroll_image(person_dir.name, bgr, image.name)
    if video_path:
        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            raise RuntimeError(f"cannot open video: {video_path}")
        frame = 0
        while True:
            ok, bgr = video.read()
            if not ok:
                break
            frame += 1
            if frame % sample_every == 0:
                enroll_image(name, bgr, f"frame {frame}")
        video.release()
    scrfd_run.close()
    arcface_run.close()
    if counts["enrolled"] == 0:
        raise RuntimeError("no faces were enrolled; check that the input shows clear, "
                           "forward-facing faces")

    entries = builder.finish()
    save_gallery(entries, cfg.gallery_path)  # writes a temporary file, then renames it
    print(f"Enrolled {counts['enrolled']} face(s), skipped {counts['skipped']}; "
          f"{len(entries)} people in {cfg.gallery_path}")
    return 0


# ── recognition ────────────────────────────────────────────────────────────────

def _is_rtsp(uri: str) -> bool:
    return uri.startswith(("rtsp://", "rtsps://"))


def _probe_input(uri: str) -> Tuple[int, int, int]:
    """Read width/height/fps from the stream or file before the graph is built."""
    if not _is_rtsp(uri) and not Path(uri).is_file():
        raise RuntimeError(f"input video not found: {uri}")
    # This short-lived probe joins the stream mid-GOP, so FFmpeg emits "Missing reference
    # picture" chatter while it waits for a keyframe. Only caps are read here, so silence it.
    key  = "OPENCV_FFMPEG_CAPTURE_OPTIONS"
    prev = os.environ.get(key)
    os.environ[key] = (prev + "|" if prev else "") + "loglevel;quiet"
    try:
        cap = cv2.VideoCapture(uri)
        if not cap.isOpened():
            raise RuntimeError(f"cannot open input: {uri}")
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
        raise RuntimeError(f"could not read the input frame size from {uri}")
    if fps <= 0:
        raise RuntimeError(f"could not read the input frame rate from {uri}")
    return w, h, fps


def _usb_camera_options(cfg: AppConfig):
    opt = pyneat.UsbCameraInputOptions()
    opt.camera_name   = cfg.usb_camera
    opt.width         = cfg.usb_width
    opt.height        = cfg.usb_height
    opt.framerate_num = cfg.usb_fps
    return opt


def _scrfd_frame_options(frame_w: int, frame_h: int, nv16_frames: bool):
    """SCRFD fed decoded video frames: the EV74 letterboxes, converts and normalizes."""
    opt = pyneat.ModelOptions()
    opt.preprocess.kind = pyneat.InputKind.Image
    opt.preprocess.enable = pyneat.AutoFlag.On
    opt.preprocess.color_convert.input_format = (
        pyneat.PreprocessColorFormat.NV16 if nv16_frames else pyneat.PreprocessColorFormat.NV12
    )
    opt.preprocess.input_max_width  = frame_w
    opt.preprocess.input_max_height = frame_h
    opt.preprocess.input_max_depth  = 3
    # Both settings must match how SCRFD was calibrated; neither is inferred from the model
    # package. Left on Auto, normalization stays off (a 255x input-scale error) and the
    # letterbox bars are grey (114) instead of black.
    opt.preprocess.resize.enable    = pyneat.AutoFlag.On
    opt.preprocess.resize.mode      = pyneat.ResizeMode.Letterbox
    opt.preprocess.resize.pad_value = 0
    opt.preprocess.normalize.enable = pyneat.AutoFlag.On
    opt.preprocess.normalize.mean   = (0.0, 0.0, 0.0)
    opt.preprocess.normalize.stddev = (1.0, 1.0, 1.0)
    return opt


def _build_recognition_run(cfg: AppConfig, frame_w: int, frame_h: int, fps: int, usb_info=None):
    """One graph for every source:

      RTSP / USB camera ─┬─→ VideoSender (the source's own stream → Insight)
                         └─→ SimaDecode ─┬─ model ─→ SCRFD ─→ detections ─┐
      video file (decodes itself) ───────┴─ frame ────────────────────────┴─ ByFrame join → app

    Insight receives the stream as the source sent it, with no re-encode. The join pairs
    every frame with the detections found in it, so each face is cropped from its own frame.
    """
    is_usb  = usb_info is not None
    is_live = is_usb or _is_rtsp(cfg.input_uri)
    nv16_frames = is_usb and usb_info.decoded_format == pyneat.Format.NV16
    scrfd_model = pyneat.Model(cfg.scrfd_model,
                               _scrfd_frame_options(frame_w, frame_h, nv16_frames))

    g_opts = pyneat.GraphOptions()
    g_opts.advanced_execution.postprocess_target = "EV74"
    graph = pyneat.Graph(g_opts)
    decoded = pyneat.graphs.branch("decoded", ["model", "frame"])

    if not is_live:
        file_opt = pyneat.VideoInputGroupOptions()
        file_opt.path = cfg.input_uri
        graph.connect(pyneat.groups.video_input(file_opt), decoded)
    else:
        source = pyneat.Graph("source")
        decode_opt = pyneat.SimaDecodeOptions()
        decode_opt.raw_output = True
        decode_opt.dec_width  = frame_w
        decode_opt.dec_height = frame_h
        decode_opt.dec_fps    = fps
        if is_usb:
            source.add(pyneat.nodes.usb_camera_input(_usb_camera_options(cfg), usb_info))
            decode_opt.type       = pyneat.SimaDecodeType.MJPEG
            decode_opt.out_format = usb_info.decoded_format
        else:
            rtsp_opt = pyneat.RtspEncodedInputOptions()
            rtsp_opt.url                  = cfg.input_uri
            rtsp_opt.latency_ms           = 50
            rtsp_opt.source_fps           = fps
            rtsp_opt.fallback_h264_width  = frame_w
            rtsp_opt.fallback_h264_height = frame_h
            source.add(pyneat.groups.rtsp_encoded_input(rtsp_opt))
            decode_opt.type       = pyneat.SimaDecodeType.H264
            decode_opt.out_format = pyneat.Format.NV12
        decoder = pyneat.Graph("decoder")
        decoder.add(pyneat.nodes.sima_decode(decode_opt))
        graph.connect(source, decoder)
        graph.connect(decoder, decoded)

        if cfg.insight_host:
            video_opt = pyneat.VideoSenderOptions.passthrough(
                pyneat.RtspCodec.MJPEG if is_usb else pyneat.RtspCodec.H264
            )
            video_opt.host            = cfg.insight_host
            video_opt.video_port_base = cfg.insight_video_port
            video_opt.channel         = 0
            video = pyneat.Graph("video")
            video.add(pyneat.groups.video_sender(video_opt))
            # The viewer gets the newest frame; a slow model never holds the video back.
            latest = pyneat.GraphLinkOptions()
            latest.policy = pyneat.GraphLinkPolicy.RealtimeLatestByStream
            graph.connect(source, video, latest)

    model_graph = pyneat.Graph("model")
    model_graph.connect(pyneat.nodes.input("model"), scrfd_model)
    detections_graph = pyneat.Graph("detections")
    detections_graph.add(pyneat.nodes.output("detections", pyneat.OutputOptions.every_frame(4)))
    frame_graph = pyneat.Graph("frame")
    frame_graph.add(pyneat.nodes.output("frame", pyneat.OutputOptions.every_frame(4)))
    joined = pyneat.graphs.combine(["frame", "detections"], _JOINED_OUTPUT,
                                   pyneat.CombinePolicy.ByFrame)
    graph.connect(decoded, model_graph)
    graph.connect(decoded, frame_graph)
    graph.connect(model_graph, detections_graph)
    graph.connect(frame_graph, joined)
    graph.connect(detections_graph, joined)

    # A live source keeps its newest frame; a file is processed frame by frame.
    run_opts = pyneat.RunOptions()
    if is_live:
        run_opts.preset = pyneat.RunPreset.Realtime
    run_opts.queue_depth     = 3
    run_opts.overflow_policy = (pyneat.OverflowPolicy.KeepLatest if is_live
                                else pyneat.OverflowPolicy.Block)
    run_opts.output_memory   = pyneat.OutputMemory.ZeroCopy
    return graph.build(run_opts)


def run_recognition(cfg: AppConfig, gallery: List[GalleryEntry], test_mode: bool = False) -> int:
    """Detect, align, embed and match every face of every frame; report to Insight."""
    usb_info = None
    if cfg.usb_camera:
        # Checks the camera once: connected, free, mode supported, one frame decodes.
        usb_info = pyneat.usb_camera_info(_usb_camera_options(cfg))
        frame_w, frame_h, fps = cfg.usb_width, cfg.usb_height, cfg.usb_fps
        source_name = "USB camera"
    else:
        frame_w, frame_h, fps = _probe_input(cfg.input_uri)
        source_name = "RTSP" if _is_rtsp(cfg.input_uri) else "video file"
    print(f"[INPUT] {source_name} {frame_w}x{frame_h} @ {fps} FPS")
    scale, pad_l, pad_t = _letterbox(frame_w, frame_h)

    # ArcFace first, so the live graph does not queue frames while it loads.
    print("[BUILD] Building ArcFace pipeline...")
    arcface_run = _build_tensor_run(cfg.arcface_model, _ARCFACE_W, _ARCFACE_H)
    print("[BUILD] Building the source and SCRFD graph...")
    run = _build_recognition_run(cfg, frame_w, frame_h, fps, usb_info)

    metadata_sender = None
    if cfg.insight_host:
        meta_opt = pyneat.MetadataSenderOptions()
        meta_opt.host               = cfg.insight_host
        meta_opt.channel            = 0
        meta_opt.metadata_port_base = cfg.insight_metadata_port
        metadata_sender = pyneat.MetadataSender(meta_opt)
        print(f"[stream] video -> udp://{cfg.insight_host}:{cfg.insight_video_port}, "
              f"metadata -> udp://{cfg.insight_host}:{metadata_sender.metadata_port()} "
              "(Insight channel 0)")

    frame_count = 0
    first_frame_ms = None
    while cfg.max_frames <= 0 or frame_count < cfg.max_frames:
        sample = run.pull(_JOINED_OUTPUT, cfg.timeout_ms)
        if sample is None:
            err = str(run.last_error() or "")
            if err:
                raise RuntimeError(f"pull failed: {err}")
            if not run.running():
                break  # end of stream
            print("[warn] pull timeout", file=sys.stderr, flush=True)
            continue
        if first_frame_ms is None:
            first_frame_ms = _time_ms()

        # Detect, then align, embed and match every face of this frame.
        tensors = [_tensor_to_np(t)
                   for t in _collect_tensors(_joined_field(sample, "detections", 1))]
        faces = decode_scrfd(tensors, cfg.conf_threshold, cfg.nms_iou, cfg.top_k,
                             cfg.keep_top_k, pad_l, pad_t, scale)
        matches: List[Tuple[str, float]] = []
        if faces:
            bgr = _tensor_to_bgr(_first_tensor(_joined_field(sample, "frame", 0)))
            for face in faces:
                embedding = embed_face(arcface_run, bgr, face["landmarks"], cfg.timeout_ms)
                matches.append(match_embedding(embedding, gallery, cfg.match_threshold,
                                               cfg.match_margin, cfg.unknown_label))

        if metadata_sender:
            objects = []
            for i, (face, (label, similarity)) in enumerate(zip(faces, matches)):
                x1 = max(0.0, min(face["x1"], float(frame_w)))
                y1 = max(0.0, min(face["y1"], float(frame_h)))
                x2 = max(x1, min(face["x2"], float(frame_w)))
                y2 = max(y1, min(face["y2"], float(frame_h)))
                objects.append({
                    "id":         f"face_{i + 1}",
                    "label":      label,
                    "confidence": float(face["score"]),
                    "similarity": float(similarity),
                    "bbox":       [x1, y1, x2 - x1, y2 - y1],
                })
            # Insight pairs this message with the video frame that carries the same timestamp.
            timestamp_ms = int(sample.pts_ns // 1_000_000) if sample.pts_ns >= 0 else -1
            frame_id = sample.frame_id if sample.frame_id >= 0 else frame_count
            metadata_sender.send_metadata(
                "object-detection", json.dumps({"objects": objects}, separators=(",", ":")),
                timestamp_ms, str(frame_id),
            )

        frame_count += 1
        if test_mode or frame_count % 100 == 0:
            if not test_mode:
                print(f"[{frame_count}] faces={len(matches)}")
            for i, (label, similarity) in enumerate(matches):
                print(f"  face[{i}] → {label}  similarity={similarity:.4f}")

    elapsed_s = (_time_ms() - first_frame_ms) / 1000.0 if first_frame_ms is not None else 0.0
    run.close()
    arcface_run.close()
    print(f"processed={frame_count} fps={frame_count / max(elapsed_s, 0.001):.1f}")
    return 0


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse_args(argv):
    p = argparse.ArgumentParser(
        description="Face recognizer: SCRFD detection + ArcFace recognition on SiMa.ai Modalix"
    )
    p.add_argument("--config",        type=Path, default=_DEFAULT_CONFIG)
    p.add_argument("--input",
                   help="RTSP URL or H.264 video file; replaces the config input")
    p.add_argument("--gallery")
    p.add_argument("--scrfd-model")
    p.add_argument("--arcface-model")
    p.add_argument("--max-frames",    type=int, default=0,
                   help="stop after N frames (0 = unlimited)")
    p.add_argument("--test",          action="store_true",
                   help="print every frame's result; disables Insight output")
    p.add_argument("--enroll",        action="store_true", help="add people to the gallery")
    p.add_argument("--images",        help="enrollment folder, one subfolder per person")
    p.add_argument("--video",         help="enrollment video of one person; needs --name")
    p.add_argument("--name",          help="the person shown in --video")
    p.add_argument("--sample-every",  type=int, default=5)
    p.add_argument("--min-score",     type=float, default=0.75)
    p.add_argument("--max-per-person", type=int, default=0)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    try:
        # config.yaml is loaded first; command-line flags then override it.
        cfg = _load_config(args.config)
        if args.input:
            cfg.input_uri  = args.input
            cfg.usb_camera = ""
        cfg.gallery_path  = args.gallery or cfg.gallery_path
        cfg.scrfd_model   = args.scrfd_model or cfg.scrfd_model
        cfg.arcface_model = args.arcface_model or cfg.arcface_model
        cfg.max_frames    = args.max_frames
        if args.test:
            cfg.insight_host = ""
        for label, path in (("SCRFD", cfg.scrfd_model), ("ArcFace", cfg.arcface_model)):
            if not Path(path).is_file():
                raise RuntimeError(f"{label} model not found: {path}")
        _load_runtime_deps()

        if args.enroll:
            return run_enrollment(cfg, args.images, args.video, args.name,
                                  args.sample_every, args.min_score, args.max_per_person)

        if not cfg.input_uri and not cfg.usb_camera:
            raise RuntimeError("no input: set input.uri or input.usb.camera in config.yaml, "
                               "or pass --input <rtsp-url|video.mp4>")
        if cfg.input_uri and cfg.usb_camera:
            raise RuntimeError("set only one input: input.uri or input.usb.camera")
        if cfg.usb_camera:
            if cfg.usb_width <= 0 or cfg.usb_height <= 0 or cfg.usb_fps <= 0:
                raise RuntimeError("input.usb.width, height and fps must be positive")
            # RTP/JPEG (RFC 2435) carries 8-pixel blocks and at most 2040 pixels per side.
            sides = (cfg.usb_width, cfg.usb_height)
            if cfg.insight_host and not all(s % 8 == 0 and s <= 2040 for s in sides):
                raise RuntimeError("input.usb.width and height must be multiples of 8 and at "
                                   "most 2040 to send the camera's JPEG to Insight")
        elif cfg.insight_host and not _is_rtsp(cfg.input_uri):
            raise RuntimeError("Insight output needs an RTSP or USB camera input; a video "
                               "file runs offline (clear output.insight.host)")

        gallery: List[GalleryEntry] = []
        if Path(cfg.gallery_path).exists():
            gallery = load_gallery(cfg.gallery_path)
            print(f"Gallery: {len(gallery)} people from {cfg.gallery_path}")
        else:
            print(f"Warning: no gallery at {cfg.gallery_path}; every face will be labelled "
                  f"{cfg.unknown_label}", file=sys.stderr)
        return run_recognition(cfg, gallery, test_mode=args.test)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"[ERR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
