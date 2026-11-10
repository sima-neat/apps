"""Face recognizer: SCRFD detection and ArcFace recognition of enrolled people, using pyneat."""

from __future__ import annotations

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

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "common" / "config.yaml"
PULL_TIMEOUT_MS = 50
# pyneat does not report the end of a video file; its pulls only stop returning frames. The
# file has ended when no frame follows the last one for this long.
VIDEO_END_SECONDS = 2.0
OUTPUT_NAME = "recognizer_output"
# Matches the C++ config reader's error wording.
KIND_NAMES = {bool: "true or false", int: "an integer", float: "numeric", str: "a string"}

GALLERY_MAGIC = b"FRGAL1\n\x00"
EMBEDDING_DIM = 512
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")

# scrfd_2.5g_bnkps: 640x640 input, three strides, two anchors per cell.
SCRFD_SIZE = 640
SCRFD_STRIDES = (8, 16, 32)
SCRFD_ANCHORS = 2
# Highest-scoring candidates kept for overlap removal.
SCRFD_MAX_CANDIDATES = 5000

ARCFACE_SIZE = 112
# Landmark positions in the 112x112 ArcFace input, from InsightFace: left eye, right eye,
# nose tip, left mouth corner, right mouth corner.
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


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SCRFD and ArcFace face recognizer")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--enroll", action="store_true")
    parser.add_argument("--validate-config-only", action="store_true")
    return parser.parse_args(argv)


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


def save_gallery(people: list[Person], path: str) -> None:
    """Writes a temporary file and renames it over `path`, so a failed save leaves the
    previous gallery in place."""
    data = bytearray(GALLERY_MAGIC)
    data += struct.pack("<I", len(people))
    for person in people:
        name = person.name.encode("utf-8")
        data += struct.pack("<H", len(name)) + name
        data += struct.pack("<I", person.samples)
        data += person.mean.astype("<f4").tobytes()
    temporary = Path(f"{path}.tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


def add_to_gallery(people: list[Person], name: str, embedding) -> None:
    """Averages `embedding` into the person called `name`, adding them on first use."""
    for person in people:
        if person.name == name:
            person.mean = (person.mean * person.samples + embedding) / (person.samples + 1)
            person.samples += 1
            return
    people.append(Person(name, embedding))


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


def preprocess_scrfd(bgr):
    """Letterboxes a BGR image to the SCRFD input (black bars, RGB, [0, 1] float32), the
    same preprocessing the EV74 runs for decoded video frames. Used for still images."""
    scale, pad_left, pad_top = letterbox_for(bgr.shape[1], bgr.shape[0])
    width, height = round(bgr.shape[1] * scale), round(bgr.shape[0] * scale)
    scaled = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((SCRFD_SIZE, SCRFD_SIZE, 3), dtype=np.float32)
    canvas[pad_top : pad_top + height, pad_left : pad_left + width] = (
        cv2.cvtColor(scaled, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    )
    return canvas


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


def run_enrollment(cfg: AppConfig) -> None:
    scrfd = build_tensor_run(cfg.scrfd_model, SCRFD_SIZE)
    arcface = build_tensor_run(cfg.arcface_model, ARCFACE_SIZE)

    # New faces are averaged into the people already enrolled.
    people = load_gallery(cfg.gallery_path) if Path(cfg.gallery_path).exists() else []
    enrolled = 0
    skipped = 0

    def enroll_image(name: str, bgr, image_name: str) -> bool:
        """Enrolls the most confident face of one image, so each image should show one person."""
        letterbox = letterbox_for(bgr.shape[1], bgr.shape[0])
        faces = decode_scrfd(infer(scrfd, preprocess_scrfd(bgr), "SCRFD"), cfg, letterbox)
        if not faces or faces[0].score < cfg.enroll_min_score:
            print(f"[skip] {image_name}: no face scoring at least {cfg.enroll_min_score}")
            return False
        add_to_gallery(people, name, embed_face(arcface, bgr, faces[0].landmarks))
        print(f"[enroll] {name}: {image_name} score={faces[0].score:.3f}")
        return True

    if cfg.enroll_images_dir:
        images_dir = Path(cfg.enroll_images_dir)
        if not images_dir.is_dir():
            raise RuntimeError(f"enrollment.images_dir is not a directory: {images_dir}")
        for person_dir in sorted(path for path in images_dir.iterdir() if path.is_dir()):
            images = sorted(
                path for path in person_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
            )
            if cfg.enroll_max_per_person > 0:
                images = images[: cfg.enroll_max_per_person]
            for image_path in images:
                bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if bgr is None:
                    print(f"[skip] {image_path.name}: cannot be read")
                    skipped += 1
                elif enroll_image(person_dir.name, bgr, image_path.name):
                    enrolled += 1
                else:
                    skipped += 1
    if cfg.enroll_video_path:
        video = cv2.VideoCapture(cfg.enroll_video_path)
        if not video.isOpened():
            raise RuntimeError(f"failed to open video: {cfg.enroll_video_path}")
        frame = 0
        while True:
            read, bgr = video.read()
            if not read:
                break
            frame += 1
            if frame % cfg.enroll_sample_every == 0:
                if enroll_image(cfg.enroll_name, bgr, f"frame {frame}"):
                    enrolled += 1
                else:
                    skipped += 1
        video.release()
    scrfd.close()
    arcface.close()
    if enrolled == 0:
        raise RuntimeError("no face was enrolled")

    Path(cfg.gallery_path).parent.mkdir(parents=True, exist_ok=True)
    save_gallery(people, cfg.gallery_path)
    print(
        f"enrolled={enrolled} skipped={skipped} people={len(people)} gallery={cfg.gallery_path}"
    )


def usb_camera_options(cfg: AppConfig):
    opt = pyneat.UsbCameraInputOptions()
    opt.camera_name = cfg.usb_camera
    opt.width = cfg.usb_width
    opt.height = cfg.usb_height
    opt.framerate_num = cfg.usb_fps
    return opt


def probe_video(path: str) -> tuple[int, int, int]:
    """Returns the width, height and frame rate of an RTSP stream or a video file."""
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise RuntimeError(f"failed to open video: {path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = int(round(capture.get(cv2.CAP_PROP_FPS) or 0))
    capture.release()
    return width, height, fps


def open_source(cfg: AppConfig) -> Source:
    if cfg.source_type == "usb":
        # Checks the camera once: connected, free, mode supported, one frame decodes.
        usb = pyneat.usb_camera_info(usb_camera_options(cfg))
        return Source(cfg.usb_width, cfg.usb_height, cfg.usb_fps, usb)
    if cfg.source_type == "rtsp":
        if cfg.rtsp_tcp:
            # OpenCV's RTSP client uses UDP unless told otherwise.
            os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
        width, height, fps = probe_video(cfg.rtsp_url)
        fps = cfg.rtsp_fps or fps
    else:
        width, height, fps = probe_video(cfg.video_path)
    if width <= 0 or height <= 0 or fps <= 0:
        raise RuntimeError("failed to read the source frame size and rate")
    return Source(width, height, fps)


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


def video_options(cfg: AppConfig):
    opt = pyneat.VideoSenderOptions.passthrough(
        pyneat.RtspCodec.MJPEG if cfg.source_type == "usb" else pyneat.RtspCodec.H264
    )
    opt.host = cfg.insight_host
    opt.channel = 0
    opt.video_port_base = cfg.video_port
    return opt


def encoded_source(cfg: AppConfig, source: Source):
    graph = pyneat.Graph("source")
    if cfg.source_type == "usb":
        graph.add(pyneat.nodes.usb_camera_input(usb_camera_options(cfg), source.usb))
        return graph
    opt = pyneat.RtspEncodedInputOptions()
    opt.url = cfg.rtsp_url
    opt.tcp = cfg.rtsp_tcp
    opt.latency_ms = cfg.rtsp_latency_ms
    opt.source_fps = source.fps
    opt.fallback_h264_width = source.width
    opt.fallback_h264_height = source.height
    graph.add(pyneat.groups.rtsp_encoded_input(opt))
    return graph


def decoder_options(cfg: AppConfig, source: Source):
    opt = pyneat.SimaDecodeOptions()
    if cfg.source_type == "usb":
        opt.type = pyneat.SimaDecodeType.MJPEG
        opt.out_format = source.usb.decoded_format
    else:
        opt.type = pyneat.SimaDecodeType.H264
        opt.out_format = pyneat.Format.NV12
    opt.raw_output = True
    opt.dec_width = source.width
    opt.dec_height = source.height
    opt.dec_fps = source.fps
    return opt


def build_graph(cfg: AppConfig, source: Source, scrfd):
    """A live source is sent to Insight as it arrives and decoded once for the model. Each
    decoded frame goes to SCRFD and to a frame tap, and the two are joined by frame so every
    face is cropped from the frame it was found in."""
    graph_options = pyneat.GraphOptions()
    graph_options.advanced_execution.postprocess_target = "EV74"
    graph = pyneat.Graph(graph_options)
    decoded = pyneat.graphs.branch("decoded", ["model", "frame"])

    if cfg.live:
        encoded = encoded_source(cfg, source)
        decoder = pyneat.Graph("decoder")
        decoder.add(pyneat.nodes.sima_decode(decoder_options(cfg, source)))
        graph.connect(encoded, decoder)
        graph.connect(decoder, decoded)

        video = pyneat.Graph("video_sender")
        video.add(pyneat.groups.video_sender(video_options(cfg)))
        latest = pyneat.GraphLinkOptions()
        latest.policy = pyneat.GraphLinkPolicy.RealtimeLatestByStream
        graph.connect(encoded, video, latest)
    else:
        file_options = pyneat.VideoInputGroupOptions()
        file_options.path = cfg.video_path
        graph.connect(pyneat.groups.video_input(file_options), decoded)

    model = pyneat.Graph("model")
    model.connect(pyneat.nodes.input("model"), scrfd)
    detections = pyneat.Graph("detections")
    detections.add(pyneat.nodes.output("detections", pyneat.OutputOptions.every_frame(4)))
    frame = pyneat.Graph("frame")
    frame.add(pyneat.nodes.output("frame", pyneat.OutputOptions.every_frame(4)))
    joined = pyneat.graphs.combine(
        ["frame", "detections"], OUTPUT_NAME, pyneat.CombinePolicy.ByFrame
    )
    graph.connect(decoded, model)
    graph.connect(decoded, frame)
    graph.connect(model, detections)
    graph.connect(frame, joined)
    graph.connect(detections, joined)
    return graph


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


def find_field(sample, label: str):
    if getattr(sample, "stream_label", "") == label:
        return sample
    for field in getattr(sample, "fields", []):
        found = find_field(field, label)
        if found is not None:
            return found
    return None


def joined_field(sample, label: str, bundle_index: int):
    field = find_field(sample, label)
    fields = list(getattr(sample, "fields", []))
    if field is not None:
        return field
    if getattr(sample, "kind", None) == pyneat.SampleKind.Bundle and len(fields) > bundle_index:
        return fields[bundle_index]
    raise RuntimeError(f"joined output missing {label} field")


def tensor_dim(tensor, name: str) -> int:
    value = getattr(tensor, name)
    return int(value() if callable(value) else value)


def frame_bgr(sample):
    """H.264 decodes to NV12; a 4:2:2 USB camera decodes to NV16."""
    tensors = sample_tensors(joined_field(sample, "frame", 0))
    if not tensors:
        raise RuntimeError("joined frame field has no tensor")
    tensor = tensors[0]
    width, height = tensor_dim(tensor, "width"), tensor_dim(tensor, "height")
    if tensor.image_format() == pyneat.PixelFormat.NV16:
        # The payload is the whole mapped buffer, so the Y and UV planes are read at their
        # own offsets and strides; each row keeps its own chroma by interleaving Y and UV
        # into packed 4:2:2 (Y0 U0 Y1 V0).
        payload = np.frombuffer(tensor.copy_payload_bytes(), dtype=np.uint8)
        planes = {plane.role: plane for plane in tensor.planes}

        def plane_rows(role):
            plane = planes[role]
            stride = plane.strides_bytes[0]
            end = plane.byte_offset + stride * height
            if end > payload.size:
                raise RuntimeError(f"NV16 payload smaller than its {role} plane")
            return payload[plane.byte_offset : end].reshape((height, stride))[:, :width]

        packed = np.dstack([plane_rows(pyneat.PlaneRole.Y), plane_rows(pyneat.PlaneRole.UV)])
        return cv2.cvtColor(packed, cv2.COLOR_YUV2BGR_YUY2)

    if not tensor.is_nv12():
        raise RuntimeError("decoded frame is neither NV12 nor NV16")
    # The decoder pads its planes; contiguous() packs them.
    payload = np.frombuffer(tensor.contiguous().copy_payload_bytes(), dtype=np.uint8)
    expected = width * height * 3 // 2
    if payload.size < expected:
        raise RuntimeError(f"NV12 payload too small: {payload.size} < {expected}")
    nv12 = payload[:expected].reshape((height * 3 // 2, width))
    return cv2.cvtColor(nv12, cv2.COLOR_YUV2BGR_NV12)


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


def send_metadata(sender, sample, faces: list[Face], matches: list, source: Source) -> None:
    objects = [
        insight_object(index, face, name, similarity, source)
        for index, (face, (name, similarity)) in enumerate(zip(faces, matches), start=1)
    ]
    timestamp_ms = int(sample.pts_ns // 1_000_000) if sample.pts_ns >= 0 else -1
    frame_id = str(sample.frame_id) if sample.frame_id >= 0 else ""
    sender.send_metadata(
        "object-detection",
        json.dumps({"objects": objects}, separators=(",", ":")),
        timestamp_ms,
        frame_id,
    )


def run_recognition(cfg: AppConfig) -> None:
    people = []
    if Path(cfg.gallery_path).exists():
        people = load_gallery(cfg.gallery_path)
    else:
        print(
            f"[warn] no gallery at {cfg.gallery_path}; every face is labelled {cfg.unknown_label}",
            file=sys.stderr,
        )

    source = open_source(cfg)
    arcface = build_tensor_run(cfg.arcface_model, ARCFACE_SIZE)
    graph = build_graph(cfg, source, build_scrfd(cfg, source))

    metadata_sender = None
    summary = f"source={source.width}x{source.height}@{source.fps} people={len(people)}"
    if cfg.live:
        metadata_options = pyneat.MetadataSenderOptions()
        metadata_options.host = cfg.insight_host
        metadata_options.channel = 0
        metadata_options.metadata_port_base = cfg.metadata_port
        metadata_sender = pyneat.MetadataSender(metadata_options)
        summary += (
            f" insight={cfg.insight_host} video={video_options(cfg).video_port}"
            f" metadata={metadata_sender.metadata_port()}"
        )
    print(summary, flush=True)

    run = graph.build(run_options(cfg))
    letterbox = letterbox_for(source.width, source.height)
    processed = 0
    first_frame = None
    last_frame = float("inf")
    try:
        while cfg.frames <= 0 or processed < cfg.frames:
            sample = run.pull(OUTPUT_NAME, PULL_TIMEOUT_MS)
            if sample is None:
                # pyneat reports a closed output as None, like a timeout, so check the run itself.
                if run.last_error():
                    raise RuntimeError(f"runtime error: {run.last_error()}")
                if not run.running():
                    raise RuntimeError(f"{OUTPUT_NAME} output closed unexpectedly")
                if not cfg.live and time.perf_counter() - last_frame > VIDEO_END_SECONDS:
                    break
                continue
            if first_frame is None:
                first_frame = time.perf_counter()

            outputs = sample_tensors(joined_field(sample, "detections", 1))
            faces = decode_scrfd(
                [np.asarray(tensor.to_numpy(copy=False)) for tensor in outputs], cfg, letterbox
            )
            matches = []
            if faces:
                bgr = frame_bgr(sample)
                matches = [
                    match_embedding(embed_face(arcface, bgr, face.landmarks), people, cfg)
                    for face in faces
                ]
            processed += 1
            last_frame = time.perf_counter()
            if metadata_sender is not None:
                send_metadata(metadata_sender, sample, faces, matches, source)
            else:
                names = ", ".join(f"{name} {similarity:.3f}" for name, similarity in matches)
                print(f"frame {processed}: {names or 'no faces'}")
    finally:
        run.close()
        arcface.close()
        seconds = last_frame - first_frame if first_frame is not None else 0.0
        fps = processed / seconds if seconds > 0.0 else 0.0
        print(f"processed={processed} fps={fps:.1f}", flush=True)


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        if not args.config.exists():
            print(f"Error: config file not found: {args.config}", file=sys.stderr)
            return 2
        cfg = load_app_config(args.config, args.enroll)
        if args.validate_config_only:
            mode = "enroll" if args.enroll else "recognize"
            print(f"Config validated: {args.config} (mode={mode})")
            return 0

        load_runtime_dependencies()
        if args.enroll:
            run_enrollment(cfg)
        else:
            run_recognition(cfg)
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"[ERR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
