"""Local model setup, decoding and numerical helpers for this example."""
from __future__ import annotations

from ament_index_python.packages import get_package_share_directory
import argparse

from dataclasses import dataclass

import glob

import json

import os

from pathlib import Path

import struct

import sys

import time

import yaml

DEFAULT_CONFIG = Path(get_package_share_directory('neat_face_recognizer')) / "common" / "config.yaml"

PULL_TIMEOUT_MS = 50

OUTPUT_NAME = "recognizer_output"

KIND_NAMES = {bool: "true or false", int: "an integer", float: "numeric", str: "a string"}

GALLERY_MAGIC = b"FRGAL1\n\x00"

EMBEDDING_DIM = 512

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")

SCRFD_SIZE = 640

SCRFD_STRIDES = (8, 16, 32)

SCRFD_ANCHORS = 2

SCRFD_MAX_CANDIDATES = 5000

ARCFACE_SIZE = 112

ARCFACE_TEMPLATE = (
    (38.2946, 51.6963),
    (73.5318, 51.5014),
    (56.0252, 71.7366),
    (41.5493, 92.3655),
    (70.7299, 92.2041),
)

cv2 = None

np = None

pyneat = None

@dataclass(frozen=True)
class AppConfig:
    scrfd_model: str
    arcface_model: str
    gallery_path: str
    source_type: str
    rtsp_url: str
    rtsp_tcp: bool
    rtsp_latency_ms: int
    rtsp_fps: int
    usb_camera: str
    usb_width: int
    usb_height: int
    usb_fps: int
    video_path: str
    frames: int
    min_score: float
    nms_iou: float
    max_faces: int
    match_threshold: float
    match_margin: float
    unknown_label: str
    enroll_images_dir: str
    enroll_video_path: str
    enroll_name: str
    enroll_sample_every: int
    enroll_min_score: float
    enroll_max_per_person: int
    insight_host: str
    video_port: int
    metadata_port: int

    @property
    def live(self) -> bool:
        return self.source_type != "video_file"

@dataclass
class Source:
    """The frame geometry of the opened source, known before the graph is built."""

    width: int
    height: int
    fps: int
    usb: object | None = None

@dataclass
class Person:
    """One enrolled person: the mean of the embeddings enrolled under their name."""

    name: str
    mean: object
    samples: int = 1

@dataclass
class Face:
    """One detected face in source-frame pixels. `box` is x1, y1, x2, y2; `landmarks` holds
    five x, y rows: left eye, right eye, nose tip, left mouth corner, right mouth corner."""

    box: object
    score: float
    landmarks: object

def load_runtime_dependencies() -> None:
    global cv2, np, pyneat
    for path in glob.glob("/usr/lib/python3*/dist-packages"):
        if path not in sys.path:
            sys.path.insert(0, path)

    import cv2 as cv2_module
    import numpy as np_module
    import pyneat as pyneat_module

    cv2 = cv2_module
    np = np_module
    pyneat = pyneat_module

def setting(raw: dict, key: str, default, kind: type):
    """Reads the dotted `key` from the parsed YAML, or `default` when it is absent."""
    value = raw
    for part in key.split("."):
        value = value.get(part) if isinstance(value, dict) else None
    if value is None:
        return default
    if kind is float and isinstance(value, int) and not isinstance(value, bool):
        value = float(value)
    if not isinstance(value, kind) or (kind is int and isinstance(value, bool)):
        raise ValueError(f"{key} must be {KIND_NAMES[kind]}")
    return value

def parse_source_type(value: str) -> str:
    if value in {"rtsp", "usb", "video_file"}:
        return value
    raise ValueError("source.type must be rtsp, usb or video_file")

def validate_config(cfg: AppConfig, enroll: bool) -> None:
    checks = [
        (cfg.scrfd_model, "model.scrfd must be set"),
        (cfg.arcface_model, "model.arcface must be set"),
        (cfg.gallery_path, "gallery.path must be set"),
        (
            0.0 < cfg.min_score < 1.0,
            "inference.min_score must be greater than 0 and less than 1",
        ),
        (0.0 <= cfg.nms_iou <= 1.0, "inference.nms_iou must be between 0 and 1"),
        (cfg.max_faces > 0, "inference.max_faces must be > 0"),
    ]
    if enroll:
        checks += [
            (
                cfg.enroll_images_dir or cfg.enroll_video_path,
                "enrollment.images_dir or enrollment.video_path must be set",
            ),
            (
                not cfg.enroll_video_path or cfg.enroll_name,
                "enrollment.name must be set with enrollment.video_path",
            ),
            (cfg.enroll_sample_every > 0, "enrollment.sample_every must be > 0"),
            (
                0.0 <= cfg.enroll_min_score <= 1.0,
                "enrollment.min_score must be between 0 and 1",
            ),
            (cfg.enroll_max_per_person >= 0, "enrollment.max_per_person must be >= 0"),
        ]
    else:
        checks += [
            (cfg.frames >= 0, "inference.frames must be >= 0"),
            (0.0 <= cfg.match_threshold <= 1.0, "match.threshold must be between 0 and 1"),
            (0.0 <= cfg.match_margin <= 1.0, "match.margin must be between 0 and 1"),
        ]
        if cfg.source_type == "rtsp":
            checks += [
                (cfg.rtsp_url, "source.rtsp.url must be set when source.type=rtsp"),
                (cfg.rtsp_latency_ms >= 0, "source.rtsp.latency_ms must be >= 0"),
                (cfg.rtsp_fps >= 0, "source.rtsp.fps must be >= 0"),
            ]
        elif cfg.source_type == "usb":
            sides = (cfg.usb_width, cfg.usb_height)
            checks += [
                (cfg.usb_camera, "source.usb.camera must be set when source.type=usb"),
                (
                    min(sides) > 0 and cfg.usb_fps > 0,
                    "source.usb.width, height and fps must be > 0",
                ),
                # Insight receives the camera's JPEG as RTP/JPEG, which carries 8-pixel
                # blocks and at most 2040 pixels per side.
                (
                    all(side % 8 == 0 and side <= 2040 for side in sides),
                    "source.usb.width and height must be multiples of 8 and at most 2040",
                ),
            ]
        else:
            checks.append(
                (
                    cfg.video_path,
                    "source.video_file.path must be set when source.type=video_file",
                )
            )
        if cfg.live:
            checks += [
                (cfg.insight_host, "output.insight.host must be set for source.type=rtsp/usb"),
                (cfg.video_port > 0, "output.insight.video_port must be > 0"),
                (cfg.metadata_port > 0, "output.insight.metadata_port must be > 0"),
            ]
    for passed, message in checks:
        if not passed:
            raise ValueError(message)

def load_app_config(config_path: Path, enroll: bool = False) -> AppConfig:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("config root must be a mapping")

    cfg = AppConfig(
        scrfd_model=setting(raw, "model.scrfd", "", str),
        arcface_model=setting(raw, "model.arcface", "", str),
        gallery_path=setting(raw, "gallery.path", "", str),
        source_type=parse_source_type(setting(raw, "source.type", "rtsp", str)),
        rtsp_url=setting(raw, "source.rtsp.url", "", str),
        rtsp_tcp=setting(raw, "source.rtsp.tcp", True, bool),
        rtsp_latency_ms=setting(raw, "source.rtsp.latency_ms", 100, int),
        rtsp_fps=setting(raw, "source.rtsp.fps", 0, int),
        usb_camera=setting(raw, "source.usb.camera", "", str),
        usb_width=setting(raw, "source.usb.width", 1280, int),
        usb_height=setting(raw, "source.usb.height", 720, int),
        usb_fps=setting(raw, "source.usb.fps", 30, int),
        video_path=setting(raw, "source.video_file.path", "", str),
        frames=setting(raw, "inference.frames", 0, int),
        min_score=setting(raw, "inference.min_score", 0.65, float),
        nms_iou=setting(raw, "inference.nms_iou", 0.40, float),
        max_faces=setting(raw, "inference.max_faces", 100, int),
        match_threshold=setting(raw, "match.threshold", 0.55, float),
        match_margin=setting(raw, "match.margin", 0.12, float),
        unknown_label=setting(raw, "match.unknown_label", "Unknown", str),
        enroll_images_dir=setting(raw, "enrollment.images_dir", "", str),
        enroll_video_path=setting(raw, "enrollment.video_path", "", str),
        enroll_name=setting(raw, "enrollment.name", "", str),
        enroll_sample_every=setting(raw, "enrollment.sample_every", 5, int),
        enroll_min_score=setting(raw, "enrollment.min_score", 0.75, float),
        enroll_max_per_person=setting(raw, "enrollment.max_per_person", 0, int),
        insight_host=setting(raw, "output.insight.host", "", str),
        video_port=setting(raw, "output.insight.video_port", 9000, int),
        metadata_port=setting(raw, "output.insight.metadata_port", 9100, int),
    )
    validate_config(cfg, enroll)
    return cfg

def load_gallery(path: str) -> list[Person]:
    """Reads a gallery. The layout, little-endian: the 8-byte magic, a uint32 person count,
    then per person a uint16 name length, the name, a uint32 sample count and 512 float32
    values."""
    data = Path(path).read_bytes()
    if data[: len(GALLERY_MAGIC)] != GALLERY_MAGIC:
        raise RuntimeError(f"not a face-recognizer gallery: {path}")
    people = []
    try:
        offset = len(GALLERY_MAGIC)
        (count,) = struct.unpack_from("<I", data, offset)
        offset += 4
        for _ in range(count):
            (name_length,) = struct.unpack_from("<H", data, offset)
            offset += 2
            name = data[offset : offset + name_length].decode("utf-8")
            offset += name_length
            (samples,) = struct.unpack_from("<I", data, offset)
            offset += 4
            mean = np.frombuffer(data, dtype="<f4", count=EMBEDDING_DIM, offset=offset).copy()
            offset += EMBEDDING_DIM * 4
            people.append(Person(name, mean, samples))
    except (struct.error, ValueError) as exc:
        raise RuntimeError(f"gallery is truncated: {path}") from exc
    return people

def match_embedding(embedding, people: list[Person], cfg: AppConfig) -> tuple[str, float]:
    """Returns the enrolled person most similar to `embedding` and that cosine similarity.
    The name is the unknown label when the similarity is below the threshold, or when the
    runner-up is closer than the margin."""
    if not people:
        return cfg.unknown_label, 0.0
    similarities = sorted(
        (
            (float(np.dot(embedding, person.mean) / np.linalg.norm(person.mean)), person.name)
            for person in people
        ),
        reverse=True,
    )
    best, name = similarities[0]
    runner_up = similarities[1][0] if len(similarities) > 1 else -1.0
    if best < cfg.match_threshold or best - runner_up < cfg.match_margin:
        name = cfg.unknown_label
    return name, best

def letterbox_for(width: int, height: int) -> tuple[float, int, int]:
    """How a frame is letterboxed into the SCRFD input: scale, left padding, top padding."""
    scale = min(SCRFD_SIZE / width, SCRFD_SIZE / height)
    pad_left = (SCRFD_SIZE - round(width * scale)) // 2
    pad_top = (SCRFD_SIZE - round(height * scale)) // 2
    return scale, pad_left, pad_top

def overlap(a, b) -> float:
    """Intersection over union of two x1, y1, x2, y2 boxes."""
    width = max(0.0, min(a[2], b[2]) - max(a[0], b[0]) + 1.0)
    height = max(0.0, min(a[3], b[3]) - max(a[1], b[1]) + 1.0)
    intersection = width * height
    area_a = (a[2] - a[0] + 1.0) * (a[3] - a[1] + 1.0)
    area_b = (b[2] - b[0] + 1.0) * (b[3] - b[1] + 1.0)
    return intersection / (area_a + area_b - intersection)

def decode_scrfd(tensors: list, cfg: AppConfig, letterbox: tuple[float, int, int]) -> list[Face]:
    """Decodes the nine SCRFD output tensors (scores, boxes, landmarks at strides 8, 16 and
    32) into faces in source-frame pixels, highest score first."""
    if len(tensors) != 9:
        raise RuntimeError(f"expected 9 SCRFD output tensors, got {len(tensors)}")
    scale, pad_left, pad_top = letterbox
    padding = np.array([pad_left, pad_top], dtype=np.float32)
    # sigmoid(logit) >= t is logit >= log(t / (1 - t)), so most anchors skip the exponential.
    logit_threshold = np.log(cfg.min_score / (1.0 - cfg.min_score))

    boxes, scores, landmarks = [], [], []
    for index, stride in enumerate(SCRFD_STRIDES):
        cells = SCRFD_SIZE // stride
        logits = np.asarray(tensors[index], dtype=np.float32).reshape(-1)
        keep = np.flatnonzero(logits >= logit_threshold)
        if keep.size == 0:
            continue
        # Both anchors of a cell share its top-left corner as their centre.
        rows, columns = np.divmod(keep // SCRFD_ANCHORS, cells)
        centres = np.stack([columns, rows], axis=1).astype(np.float32) * stride
        # Boxes and landmarks are distances from the anchor centre, in units of the stride.
        sides = np.asarray(tensors[3 + index], dtype=np.float32).reshape(-1, 4)[keep] * stride
        points = np.asarray(tensors[6 + index], dtype=np.float32).reshape(-1, 5, 2)[keep] * stride
        # Model-input pixels back to source-frame pixels.
        boxes.append(
            np.hstack([centres - sides[:, :2] - padding, centres + sides[:, 2:] - padding]) / scale
        )
        landmarks.append((centres[:, None, :] + points - padding) / scale)
        scores.append(1.0 / (1.0 + np.exp(-logits[keep])))
    if not boxes:
        return []

    boxes = np.concatenate(boxes)
    scores = np.concatenate(scores)
    landmarks = np.concatenate(landmarks)
    faces: list[Face] = []
    for index in np.argsort(-scores)[:SCRFD_MAX_CANDIDATES]:
        if any(overlap(boxes[index], face.box) > cfg.nms_iou for face in faces):
            continue
        faces.append(Face(boxes[index], float(scores[index]), landmarks[index]))
        if len(faces) == cfg.max_faces:
            break
    return faces

def similarity_transform(landmarks):
    """Least-squares similarity transform (rotation, uniform scale, translation) that maps
    the landmarks onto the template: x' = a*x - b*y + c, y' = b*x + a*y + d."""
    source = np.asarray(landmarks, dtype=np.float64)
    target = np.asarray(ARCFACE_TEMPLATE, dtype=np.float64)
    source_mean, target_mean = source.mean(axis=0), target.mean(axis=0)
    x, y = (source - source_mean).T
    u, v = (target - target_mean).T
    spread = float(np.sum(x * x + y * y))
    if spread <= 0.0:
        raise RuntimeError("face landmarks are degenerate")
    a = float(np.sum(x * u + y * v)) / spread
    b = float(np.sum(x * v - y * u)) / spread
    c = target_mean[0] - a * source_mean[0] + b * source_mean[1]
    d = target_mean[1] - b * source_mean[0] - a * source_mean[1]
    return np.array([[a, -b, c], [b, a, d]], dtype=np.float64)

def align_face(bgr, landmarks):
    """Rotates, scales and shifts the face so its five landmarks land on the positions
    ArcFace was trained with, and returns the 112x112 BGR crop."""
    return cv2.warpAffine(
        bgr,
        similarity_transform(landmarks),
        (ARCFACE_SIZE, ARCFACE_SIZE),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )

def sample_tensors(sample) -> list:
    if sample.kind == pyneat.SampleKind.Tensor and sample.tensor is not None:
        return [sample.tensor]
    if sample.kind == pyneat.SampleKind.TensorSet:
        return list(sample.tensors)
    return [tensor for field in sample.fields for tensor in sample_tensors(field)]

def float_tensor(image):
    return pyneat.Tensor.from_numpy(
        np.ascontiguousarray(image, dtype=np.float32),
        copy=True,
        layout=pyneat.TensorLayout.HWC,
        memory=pyneat.TensorMemory.EV74,
    )

def build_tensor_run(model_path: str, size: int):
    """ArcFace, and SCRFD for the still images of enrollment, take one float tensor per call."""
    opt = pyneat.ModelOptions()
    opt.preprocess.kind = pyneat.InputKind.Tensor
    opt.preprocess.input_max_width = size
    opt.preprocess.input_max_height = size
    opt.preprocess.input_max_depth = 3
    model = pyneat.Model(model_path, opt)

    graph = pyneat.Graph()
    graph.add(pyneat.nodes.input(model.input_appsrc_options(True)))
    graph.add(model)
    graph.add(pyneat.nodes.output())

    run_options = pyneat.RunOptions()
    run_options.output_memory = pyneat.OutputMemory.Owned
    seed = np.zeros((size, size, 3), dtype=np.float32)
    return graph.build([float_tensor(seed)], run_options)

def infer(run, image, model_name: str) -> list:
    if not run.push([float_tensor(image)]):
        raise RuntimeError(f"{model_name} push failed: {run.last_error()}")
    sample = run.pull()
    if sample is None:
        raise RuntimeError(f"{model_name} returned no output")
    return [np.asarray(tensor.to_numpy(copy=False)) for tensor in sample_tensors(sample)]

def embed_face(arcface, bgr, landmarks):
    """Enrollment and recognition both turn a detected face into its embedding here, so a
    face enrolled from a photo and the same face seen live are comparable."""
    crop = cv2.cvtColor(align_face(bgr, landmarks), cv2.COLOR_BGR2RGB)
    output = infer(arcface, crop.astype(np.float32) / 127.5 - 1.0, "ArcFace")
    if not output:
        raise RuntimeError("ArcFace returned no tensors")
    embedding = output[0].reshape(-1).astype(np.float32)
    norm = float(np.linalg.norm(embedding))
    return embedding / norm if norm > 0.0 else embedding

def build_scrfd(cfg: AppConfig, source: Source):
    """SCRFD fed decoded video frames; the EV74 letterboxes, converts and normalizes each one."""
    nv16 = source.usb is not None and source.usb.decoded_format == pyneat.Format.NV16
    opt = pyneat.ModelOptions()
    opt.preprocess.kind = pyneat.InputKind.Image
    opt.preprocess.enable = pyneat.AutoFlag.On
    opt.preprocess.color_convert.input_format = (
        pyneat.PreprocessColorFormat.NV16 if nv16 else pyneat.PreprocessColorFormat.NV12
    )
    opt.preprocess.input_max_width = source.width
    opt.preprocess.input_max_height = source.height
    opt.preprocess.input_max_depth = 3
    # SCRFD expects [0, 1] input and black letterbox bars. Neither is taken from the model
    # package: left on Auto, normalization stays off and the bars are grey.
    opt.preprocess.resize.enable = pyneat.AutoFlag.On
    opt.preprocess.resize.mode = pyneat.ResizeMode.Letterbox
    opt.preprocess.resize.pad_value = 0
    opt.preprocess.normalize.enable = pyneat.AutoFlag.On
    opt.preprocess.normalize.mean = (0.0, 0.0, 0.0)
    opt.preprocess.normalize.stddev = (1.0, 1.0, 1.0)
    return pyneat.Model(cfg.scrfd_model, opt)

def run_options(cfg: AppConfig):
    """A live source keeps its newest frame; a video file is processed frame by frame."""
    opt = pyneat.RunOptions()
    if cfg.live:
        opt.preset = pyneat.RunPreset.Realtime
    opt.queue_depth = 3
    opt.overflow_policy = (
        pyneat.OverflowPolicy.KeepLatest if cfg.live else pyneat.OverflowPolicy.Block
    )
    opt.output_memory = pyneat.OutputMemory.ZeroCopy
    return opt

def insight_object(index: int, face: Face, name: str, similarity: float, source: Source) -> dict:
    x1 = min(max(float(face.box[0]), 0.0), float(source.width))
    y1 = min(max(float(face.box[1]), 0.0), float(source.height))
    x2 = min(max(float(face.box[2]), x1), float(source.width))
    y2 = min(max(float(face.box[3]), y1), float(source.height))
    return {
        "id": f"face_{index}",
        "label": name,
        "confidence": face.score,
        "similarity": similarity,
        "bbox": [x1, y1, x2 - x1, y2 - y1],
    }
