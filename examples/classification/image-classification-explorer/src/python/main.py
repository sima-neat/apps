"""Classify a single image or a directory of images with one or more models
and generate a browsable HTML report plus JSON/CSV results."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import sys
import time
import tempfile
import urllib.error
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")
# Marker proving a directory was produced by this application; publish_report
# only ever replaces a directory that carries it (or an empty one).
REPORT_MARKER = ".image-classification-explorer-report"
# The shipped config references the bundled label map by its in-package path,
# which is relative to the example directory rather than the caller's cwd. Only
# this exact reference falls back to the bundled copy; any other missing
# label_map path is a configuration error.
BUNDLED_LABEL_MAP_REF = "src/common/imagenet_labels.txt"
# Profile names travel through config keys, report columns, CSV/JSON fields and
# the HTML controls, and the C++ side addresses them through dotted, colon-
# separated config keys. Restrict them to a portable set so both languages
# accept exactly the same names instead of diverging on exotic YAML keys.
PROFILE_NAME_RE = re.compile(r"[A-Za-z0-9_-]+")
REPORT_ENTRIES = ("report.json", "report.csv", "report.html", "thumbnails", REPORT_MARKER)
# Python computes softmax in float32 and C++ accumulates in double, so the two
# agree only to about seven digits - well beyond the precision a float32 model
# output carries. Report a rounded value so both emit identical numbers.
PROBABILITY_DECIMALS = 6
# The C++ ScalarConfig parses scalars with std::stoi, which is limited to int32.
# Reject anything outside that range so both entrypoints accept the same values.
INT32_MIN, INT32_MAX = -2**31, 2**31 - 1

ClassSummary = dict[str, dict[str, dict[str, Any]]]  # model -> class id (str) -> {label, count}


@dataclass
class ModelProfile:
    name: str
    path: str
    input_width: int
    input_height: int
    preprocess: str
    output: str
    num_classes: int
    label_map: str | None
    top_k: int
    labels: list[str] = field(default_factory=list)


@dataclass
class Prediction:
    top_k: list[tuple[int, str, float]]
    inference_ms: float


@dataclass
class ImageResult:
    image_path: Path
    # (size, mtime_ns) captured once, so every model and the thumbnail are known
    # to describe the same bytes even if the file is replaced mid-run.
    fingerprint: tuple[int, int] | None = None
    predictions: dict[str, Prediction] = field(default_factory=dict)  # model name -> Prediction
    errors: dict[str, str] = field(default_factory=dict)  # model name -> error message


def stable_digest(value: str) -> str:
    """FNV-1a (64-bit), so thumbnail names are identical across runs, machines
    and both implementations. Python's built-in hash() is salted per process and
    would rename every thumbnail on each run."""
    digest = 0xCBF29CE484222325
    for byte in value.encode("utf-8"):
        digest = ((digest ^ byte) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"{digest:016x}"


# --- Inputs: fingerprinting, decoding and the fallback image cache ---------------


def file_fingerprint(path: Path) -> tuple[int, int]:
    info = path.stat()
    return (info.st_size, info.st_mtime_ns)


def check_unchanged(path: Path, expected: tuple[int, int] | None) -> None:
    """Refuse to mix results from different versions of the same input file."""
    if expected is not None and file_fingerprint(path) != expected:
        raise ValueError(f"input changed while the run was in progress: {path}")


def is_decodable_image(path: Path) -> bool:
    import cv2

    try:
        return cv2.imread(str(path), cv2.IMREAD_COLOR) is not None
    except cv2.error:
        return False  # some builds raise instead of returning None on bad data


def fallback_cache_path(url: str, base: Path) -> Path:
    """Cache path for a fallback URL.

    The URL is part of the file name, so different URLs never share a cache
    entry and there is no separate "which URL is this?" marker that could be
    left paired with another run's bytes. The digest is the same FNV-1a the C++
    implementation uses, so both share one cache entry rather than downloading
    the same image twice."""
    return base.with_name(f"{base.stem}-{stable_digest(url)}{base.suffix}")


def download_image(url: str, base: Path) -> Path:
    """Download a fallback image into a URL-keyed cache entry.

    The download lands on a process-private temporary file, is decoded before
    it is published, and is then moved into place with a single atomic rename.
    Concurrent runs (same URL or not) therefore cannot observe or leave a
    half-updated cache entry, and a non-image payload served with HTTP 200
    (e.g. a proxy error page) is never cached."""
    dest = fallback_cache_path(url, base)
    if dest.exists():
        if is_decodable_image(dest):
            return dest
        dest.unlink(missing_ok=True)  # truncated or corrupted since it was cached

    print(f"Downloading {url} ...")
    temporary = dest.with_name(f"{dest.name}.tmp-{os.getpid()}")
    try:
        urllib.request.urlretrieve(url, temporary)
        if not is_decodable_image(temporary):
            raise ValueError("downloaded file is not a decodable image")
        temporary.replace(dest)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        temporary.unlink(missing_ok=True)
        raise FileNotFoundError(f"failed to download {url}: {exc}") from exc
    return dest


# --- Configuration: reading config.yaml and validating every setting -------------


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    # An empty file is an empty configuration; a file holding a bare scalar is
    # not a configuration at all, and `or {}` would hide that.
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(f"{config_path} must contain a YAML mapping at the top level")
    return loaded


def config_int(value: Any, key: str, default: int) -> int:
    """Read an integer config value the way the C++ ScalarConfig does.

    PyYAML hands back real Python objects, so `top_k: 1.9` arrives as a float
    that int() would silently truncate to 1 while C++ rejects the same file as
    non-integral. Accept only genuine integers (or their exact text form) so
    both entrypoints run the same settings."""
    if value is None:
        return default
    if isinstance(value, bool):
        raise ValueError(f"{key} must be an integer, got {value!r}")
    if isinstance(value, int):
        return _check_int32(value, key)
    if isinstance(value, str):
        text = value.strip()
        try:
            return _check_int32(int(text, 10), key)
        except ValueError as exc:
            if "out of range" in str(exc):
                raise
            raise ValueError(f"{key} must be an integer, got {value!r}") from None
    raise ValueError(f"{key} must be an integer, got {value!r}")


def config_float(value: Any, key: str, default: float) -> float:
    """Read a floating-point config value the way the C++ ScalarConfig does.

    ScalarConfig parses every scalar from text, so a quoted `"0.2"` is a number
    there; rejecting it here would make the two entrypoints accept different
    configurations."""
    if value is None:
        return default
    if isinstance(value, bool):
        raise ValueError(f"{key} must be a number, got {value!r}")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            raise ValueError(f"{key} must be a number, got {value!r}") from None
    raise ValueError(f"{key} must be a number, got {value!r}")


def _check_int32(value: int, key: str) -> int:
    if not INT32_MIN <= value <= INT32_MAX:
        raise ValueError(f"{key} is out of range for a 32-bit integer: {value}")
    return value


def normalize_extension(value: str) -> str:
    """Accept `jpg` as well as `.jpg`. Path.suffix always carries the dot, so an
    entry without one would silently match nothing."""
    text = value.strip().lower()
    return text if text.startswith(".") else f".{text}"


def config_section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    """Read a top-level section.

    Absent or null means "use the defaults"; anything else must be a mapping.
    `raw.get(name) or {}` cannot make that distinction, because a falsy scalar
    such as `io: false` or `runtime: 0` collapses to an empty mapping and the
    defaults are applied to a configuration the customer got wrong."""
    value = raw.get(name)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"`{name}` must be a mapping, got {value!r}")
    return value


def config_str(value: Any, key: str, default: str | None) -> str | None:
    """Read a string config value the way the C++ ScalarConfig does.

    `key: null` means "not set" and falls back to the default rather than
    becoming the text "None". Other scalars are rendered as the text C++ reads
    from the same file: it sees every value as text and cannot tell `path: 5`
    from `path: "5"`, so rejecting one here would make the two entrypoints
    accept different configurations. Booleans are rendered YAML-style
    (`true`/`false`) rather than Python-style for the same reason."""
    if value is None:
        return default
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    raise ValueError(f"{key} must be a scalar, got {value!r}")


def load_profiles(raw: dict[str, Any]) -> list[ModelProfile]:
    models_cfg = config_section(raw, "models")
    if not models_cfg:
        raise ValueError("config.yaml must define at least one entry under `models`")

    profiles = []
    for raw_name, cfg in models_cfg.items():
        # PyYAML turns unquoted scalars into Python objects, so `1:` arrives as an
        # int, `01:` as the int 1 and `true:` as a bool - none of which match the
        # raw text C++ reads from the same file. Require a quoted (string) key so
        # both entrypoints see exactly the same profile name.
        if not isinstance(raw_name, str):
            raise ValueError(
                f"models: profile name {raw_name!r} is not a string; quote it in config.yaml"
            )
        # A duplicate key cannot reach here: PyYAML keeps the last of two
        # identical keys, and only string keys are accepted, so no two distinct
        # keys can render as the same name.
        name = raw_name
        if not isinstance(cfg, dict):
            raise ValueError(f"models.{name} must be a mapping")
        profile = ModelProfile(
            name=name,
            path=config_str(cfg.get("path"), f"models.{name}.path", "") or "",
            input_width=config_int(cfg.get("input_width"), f"models.{name}.input_width", 224),
            input_height=config_int(cfg.get("input_height"), f"models.{name}.input_height", 224),
            preprocess=config_str(cfg.get("preprocess"), f"models.{name}.preprocess", "imagenet"),
            output=config_str(cfg.get("output"), f"models.{name}.output", "softmax"),
            num_classes=config_int(cfg.get("num_classes"), f"models.{name}.num_classes", 1000),
            label_map=config_str(cfg.get("label_map"), f"models.{name}.label_map", None),
            top_k=config_int(cfg.get("top_k"), f"models.{name}.top_k", 5),
        )
        if not PROFILE_NAME_RE.fullmatch(name):
            raise ValueError(
                f"models.{name}: profile names may only contain letters, digits, "
                "'_' and '-'"
            )
        if not profile.path:
            raise ValueError(f"models.{name}.path is required")
        if profile.preprocess != "imagenet":
            raise ValueError(
                f"models.{name}.preprocess={profile.preprocess!r} is not supported; "
                "only 'imagenet' is implemented"
            )
        if profile.output != "softmax":
            raise ValueError(
                f"models.{name}.output={profile.output!r} is not supported; "
                "only 'softmax' is implemented (raw per-class scores, softmax applied, "
                "index i maps to label_map[i])"
            )
        if profile.top_k <= 0:
            raise ValueError(f"models.{name}.top_k must be positive, got {profile.top_k}")
        if profile.num_classes <= 0:
            raise ValueError(
                f"models.{name}.num_classes must be positive, got {profile.num_classes}"
            )
        if profile.input_width <= 0 or profile.input_height <= 0:
            raise ValueError(
                f"models.{name}.input_width/input_height must be positive, got "
                f"{profile.input_width}x{profile.input_height}"
            )
        profiles.append(profile)
    return profiles


def load_label_map(path: str | None, num_classes: int) -> list[str]:
    if not path:
        return [str(i) for i in range(num_classes)]
    label_path = Path(path)
    if not label_path.exists() and Path(path).as_posix() == BUNDLED_LABEL_MAP_REF:
        # Resolve the shipped reference next to this script, regardless of the
        # caller's cwd (model.path stays cwd-relative: it points at a file the
        # customer downloaded). A missing custom path is NOT redirected here.
        bundled = Path(__file__).resolve().parents[1] / "common" / Path(BUNDLED_LABEL_MAP_REF).name
        if bundled.exists():
            label_path = bundled
    try:
        with label_path.open("r", encoding="utf-8") as handle:
            # Positional: physical line index == class id, so blank lines are never
            # dropped (that would silently shift every later label).
            labels = [line.strip() for line in handle.read().splitlines()]
    except OSError as exc:
        raise ValueError(f"failed to open label map {label_path}: {exc}") from exc
    if len(labels) < num_classes:
        raise ValueError(
            f"label map {label_path} has {len(labels)} entries, expected at least {num_classes}"
        )
    for class_id, label in enumerate(labels[:num_classes]):
        if not label:
            raise ValueError(
                f"label map {label_path} line {class_id + 1} is blank; every class id "
                f"0..{num_classes - 1} needs a label"
            )
    return labels


def discover_images(input_path: str | None, extensions: tuple[str, ...],
                     fallback_url: str, fallback_dest: Path) -> tuple[list[Path], list[str]]:
    """Return (images, skipped_descriptions) in deterministic order."""
    if not input_path:
        return [download_image(fallback_url, fallback_dest)], []

    path = Path(input_path)
    if path.is_file():
        if path.suffix.lower() not in extensions:
            return [], [f"{path}: unsupported extension {path.suffix or '(none)'}"]
        return [path], []

    if not path.is_dir():
        raise FileNotFoundError(f"input path does not exist: {path}")

    images: list[Path] = []
    skipped: list[str] = []
    for entry in sorted(path.iterdir(), key=lambda p: p.name):
        if not entry.is_file():
            continue
        if entry.suffix.lower() not in extensions:
            skipped.append(f"{entry}: unsupported extension {entry.suffix or '(none)'}")
            continue
        images.append(entry)

    if not images and not skipped:
        raise FileNotFoundError(f"no image files found under {path}")
    return images, skipped


# --- Neat inference: preprocessing, model construction and execution -------------


def load_rgb_resized(path: str, width: int, height: int):
    """Load an image with OpenCV and return an RGB uint8 HWC array."""
    import cv2

    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"failed to read image: {path}")
    # INTER_AREA matches sima_examples::load_rgb_resized (support/runtime/example_utils.cpp)
    # used by the C++ implementation, so both languages preprocess images identically.
    bgr = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def softmax(x):
    import numpy as np

    e = np.exp(x - np.max(x))
    return e / e.sum()


def tensor_to_numpy_dense(tensor) -> Any:
    """Convert a dense pyneat tensor to a NumPy array without DLPack."""
    import numpy as np
    import pyneat

    dtype_map = {
        pyneat.TensorDType.UInt8: np.uint8,
        pyneat.TensorDType.Int8: np.int8,
        pyneat.TensorDType.UInt16: np.uint16,
        pyneat.TensorDType.Int16: np.int16,
        pyneat.TensorDType.Int32: np.int32,
        pyneat.TensorDType.Float32: np.float32,
        pyneat.TensorDType.Float64: np.float64,
    }
    np_dtype = dtype_map.get(tensor.dtype)
    if np_dtype is None:
        raise TypeError(f"Unsupported tensor dtype for NumPy conversion: {tensor.dtype}")

    shape = tuple(int(x) for x in tensor.shape)
    arr = np.frombuffer(tensor.copy_dense_bytes_tight(), dtype=np_dtype)
    if shape:
        arr = arr.reshape(shape)
    return arr


def build_model(profile: ModelProfile):
    import pyneat

    if profile.preprocess != "imagenet":
        raise ValueError(
            f"models.{profile.name}.preprocess={profile.preprocess!r} is not supported; "
            "only 'imagenet' is implemented"
        )

    opt = pyneat.ModelOptions()
    opt.preprocess.kind = pyneat.InputKind.Image
    opt.preprocess.color_convert.input_format = pyneat.PreprocessColorFormat.RGB
    opt.preprocess.input_max_width = profile.input_width
    opt.preprocess.input_max_height = profile.input_height
    opt.preprocess.input_max_depth = 3
    opt.preprocess.preset = pyneat.NormalizePreset.ImageNet
    return pyneat.Model(profile.path, opt)


def classify(model, profile: ModelProfile, image_path: Path, timeout_ms: int) -> Prediction:
    import numpy as np
    import pyneat

    rgb = load_rgb_resized(str(image_path), profile.input_width, profile.input_height)
    tensor = pyneat.Tensor.from_numpy(rgb, copy=True, image_format=pyneat.PixelFormat.RGB)

    start = time.monotonic()
    outputs = model.run([tensor], timeout_ms=timeout_ms)
    inference_ms = (time.monotonic() - start) * 1000.0

    if not outputs:
        raise RuntimeError("model run returned empty output")
    scores = tensor_to_numpy_dense(outputs[0]).flatten().astype(np.float32)
    if scores.size < profile.num_classes:
        raise RuntimeError(f"expected at least {profile.num_classes} scores, got {scores.size}")
    scores = scores[: profile.num_classes]
    if not np.isfinite(scores).all():
        # NaN/inf would propagate into softmax and then into report.json, where
        # json.dumps emits a bare NaN that is not valid JSON.
        raise RuntimeError("model produced non-finite scores")

    probs = softmax(scores)
    # Tie-break by ascending class id so equal scores resolve deterministically
    # and identically to the C++ implementation's topk_with_softmax.
    top_indices = np.lexsort((np.arange(scores.size), -scores))[: profile.top_k]
    top_k = [
        (int(i), profile.labels[int(i)], float(probs[int(i)]))
        for i in top_indices
    ]
    return Prediction(top_k=top_k, inference_ms=inference_ms)


def run_all(profiles: list[ModelProfile], images: list[Path], timeout_ms: int) -> list[ImageResult]:
    # Fingerprint every input up front: models are loaded one at a time, so each
    # image is re-read per model, and agreement is only meaningful if those reads
    # saw the same bytes.
    results = []
    for image in images:
        try:
            results.append(ImageResult(image_path=image, fingerprint=file_fingerprint(image)))
        except OSError:
            results.append(ImageResult(image_path=image))

    for profile in profiles:
        print(f"Loading model '{profile.name}': {profile.path}")
        model = build_model(profile)
        try:
            for result in results:
                try:
                    check_unchanged(result.image_path, result.fingerprint)
                    prediction = classify(model, profile, result.image_path, timeout_ms)
                    # Only keep the prediction once the bytes behind it are proven
                    # unchanged, so a swapped input leaves an error and no result.
                    check_unchanged(result.image_path, result.fingerprint)
                    result.predictions[profile.name] = prediction
                except Exception as exc:  # noqa: BLE001 - per-image, per-model failures must not abort the run
                    print(f"  {result.image_path}: {profile.name} failed: {exc}", file=sys.stderr)
                    result.errors[profile.name] = str(exc)
        finally:
            # Release this model's accelerator resources before the next profile is
            # loaded, so only one model is ever resident at a time (as in C++, where
            # `model` is scoped to the loop body).
            del model

    return results


# --- Reporting: agreement, per-class counts and the JSON/CSV/HTML writers --------


def agreement(result: ImageResult, profile_names: list[str]) -> bool | None:
    """True/False only when every named model has a top-1 result; otherwise
    indeterminate (None) rather than silently agreeing/disagreeing over a
    partial subset. Identity is the class id, not the label: ImageNet has
    distinct classes that share a display label (e.g. 134/517 "crane")."""
    if len(profile_names) < 2:
        return None
    class_ids = []
    for name in profile_names:
        pred = result.predictions.get(name)
        if pred is None or not pred.top_k:
            return None
        class_ids.append(pred.top_k[0][0])
    return len(set(class_ids)) == 1


def build_class_summary(results: list[ImageResult], profiles: list[ModelProfile]) -> ClassSummary:
    """Count top-1 predictions per model, keyed by class id so distinct classes
    that share a label are never merged."""
    summary: ClassSummary = {p.name: {} for p in profiles}
    for result in results:
        for name, pred in result.predictions.items():
            if not pred.top_k:
                continue
            class_id, label, _ = pred.top_k[0]
            entry = summary.setdefault(name, {}).setdefault(str(class_id), {"label": label, "count": 0})
            entry["count"] += 1
    return summary


def write_json_report(path: Path, results: list[ImageResult], profiles: list[ModelProfile],
                       skipped: list[str], class_summary: ClassSummary,
                       timing: dict[str, Any]) -> None:
    payload = {
        "models": [p.name for p in profiles],
        "images": [],
        "skipped": skipped,
        "class_summary": class_summary,
        "timing": timing,
    }
    for result in results:
        entry: dict[str, Any] = {"path": str(result.image_path)}
        if result.predictions:
            entry["predictions"] = {
                name: {
                    "top_k": [
                        {"class_id": c, "label": lbl,
                         "probability": round(p, PROBABILITY_DECIMALS)}
                        for c, lbl, p in pred.top_k
                    ],
                    "inference_ms": pred.inference_ms,
                }
                for name, pred in result.predictions.items()
            }
            entry["agreement"] = agreement(result, [p.name for p in profiles])
        if result.errors:
            entry["errors"] = result.errors
        payload["images"].append(entry)

    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_csv_report(path: Path, results: list[ImageResult], profiles: list[ModelProfile]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        # csv.writer defaults to CRLF; the C++ writer emits LF, and the two
        # reports must be byte-identical for the same run.
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["image", "model", "status", "top1_class_id", "top1_label",
                          "top1_probability", "inference_ms", "top_k"])
        for result in results:
            for profile in profiles:
                pred = result.predictions.get(profile.name)
                if pred is not None and pred.top_k:
                    top1 = pred.top_k[0]
                    top_k_str = ";".join(f"{lbl}:{p:.4f}" for _, lbl, p in pred.top_k)
                    writer.writerow([
                        result.image_path, profile.name, "ok", top1[0], top1[1],
                        f"{top1[2]:.4f}", f"{pred.inference_ms:.2f}", top_k_str,
                    ])
                elif profile.name in result.errors:
                    writer.writerow([
                        result.image_path, profile.name, "error", "", "", "", "",
                        result.errors[profile.name],
                    ])
                else:
                    writer.writerow([result.image_path, profile.name, "no_result", "", "", "", "", ""])


def make_thumbnail(image_path: Path, thumb_dir: Path, max_side: int = 160,
                   fingerprint: tuple[int, int] | None = None) -> str | None:
    import cv2

    # The thumbnail must show the bytes the predictions were made from.
    try:
        check_unchanged(image_path, fingerprint)
    except (OSError, ValueError):
        return None
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    try:
        # Re-check after the read, as inference does: a file replaced between
        # the check above and this read would otherwise put a thumbnail of the
        # new bytes next to predictions made from the old ones.
        check_unchanged(image_path, fingerprint)
    except (OSError, ValueError):
        return None
    h, w = img.shape[:2]
    scale = min(1.0, max_side / max(h, w))  # shrink only; never upscale a small image
    resized = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))))
    thumb_dir.mkdir(parents=True, exist_ok=True)
    thumb_name = f"{stable_digest(str(image_path))}.jpg"
    thumb_path = thumb_dir / thumb_name
    try:
        written = cv2.imwrite(str(thumb_path), resized)
    except cv2.error as exc:
        raise OSError(f"failed to write thumbnail {thumb_path}: {exc}") from exc
    if not written:
        raise OSError(f"failed to write thumbnail {thumb_path}")
    return f"thumbnails/{thumb_name}"


def bundled_asset(name: str) -> str:
    """Read a file shipped next to this script under src/common.

    The report's CSS and JavaScript live there rather than inside either
    entrypoint, so both emit exactly the same markup and cannot drift apart."""
    path = Path(__file__).resolve().parents[1] / "common" / name
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise OSError(f"failed to read bundled report asset {path}: {exc}") from exc


def html_escape(value: str) -> str:
    return (
        value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    )


def write_html_report(path: Path, results: list[ImageResult], profiles: list[ModelProfile],
                       skipped: list[str], class_summary: ClassSummary,
                       output_dir: Path) -> None:
    profile_names = [p.name for p in profiles]
    rows = []
    for idx, result in enumerate(results):
        thumb = make_thumbnail(result.image_path, output_dir / "thumbnails",
                               fingerprint=result.fingerprint)

        top1_by_model = {
            name: {"class_id": pred.top_k[0][0], "label": pred.top_k[0][1],
                   "prob": round(pred.top_k[0][2], PROBABILITY_DECIMALS)}
            for name, pred in result.predictions.items() if pred.top_k
        }
        # Compact separators and sorted keys match nlohmann::json::dump(), which
        # stores objects sorted, so both implementations emit the same attribute
        # text. The browser looks these up by name, so order is presentational.
        top1_json = html_escape(
            json.dumps(top1_by_model, separators=(",", ":"), sort_keys=True)
        )
        cells = []
        for name in profile_names:
            pred = result.predictions.get(name)
            if pred is not None and pred.top_k:
                top_str = "<br>".join(f"{html_escape(lbl)} ({p:.2%})" for _, lbl, p in pred.top_k)
                cells.append(
                    f'<td data-model-col="{html_escape(name)}">{top_str}<br>'
                    f'<span class="timing">{pred.inference_ms:.1f} ms</span></td>'
                )
            elif name in result.errors:
                cells.append(
                    f'<td data-model-col="{html_escape(name)}" class="error">'
                    f'error: {html_escape(result.errors[name])}</td>'
                )
            else:
                cells.append(f'<td data-model-col="{html_escape(name)}">no result</td>')

        has_error = "1" if result.errors else "0"
        # alt="" marks the thumbnail decorative: the path is in the next cell,
        # so a screen reader should not announce it twice.
        img_cell = f'<img src="{thumb}" alt="">' if thumb else ""
        rows.append(
            f'<tr class="row" data-has-error="{has_error}" data-idx="{idx}" '
            f'data-top1="{top1_json}"><td>{img_cell}</td>'
            f'<td>{html_escape(str(result.image_path))}</td>{"".join(cells)}'
            f'<td class="agree-cell">&mdash;</td></tr>\n'
        )

    header_cols = "".join(
        f'<th data-model-col="{html_escape(name)}">{html_escape(name)}</th>' for name in profile_names
    )
    model_checkboxes = "".join(
        f'<label><input type="checkbox" class="model-checkbox" value="{html_escape(name)}" checked> '
        f'{html_escape(name)}</label>'
        for name in profile_names
    )
    # Most frequent first, then by class id: counts tie often, and ordering by
    # insertion would differ from the C++ map, which is ordered by class id.
    summary_rows = "".join(
        f"<tr><td>{html_escape(name)}</td><td>{html_escape(class_id)}</td>"
        f"<td>{html_escape(entry['label'])}</td><td>{entry['count']}</td></tr>"
        for name, per_class in class_summary.items()
        for class_id, entry in sorted(
            per_class.items(), key=lambda kv: (-kv[1]["count"], int(kv[0]))
        )
    )
    skipped_rows = "".join(f"<li>{html_escape(s)}</li>" for s in skipped)

    report_css = bundled_asset("report.css")
    report_js = bundled_asset("report.js")
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Image Classification Explorer Report</title>
<style>
{report_css}</style>
</head>
<body>
<h1>Image Classification Explorer Report</h1>
<p>Models: {html_escape(', '.join(profile_names))} &middot; Images: {len(results)} &middot; Skipped: {len(skipped)}</p>
<div id="controls">
  <div class="dropdown" id="modelDropdown">
    <button type="button" class="dropdown-btn" id="modelDropdownBtn">All models &#9662;</button>
    <div class="dropdown-panel" id="modelDropdownPanel">
      {model_checkboxes}
      <div class="dropdown-actions">
        <button type="button" id="modelSelectAll">All</button>
        <button type="button" id="modelSelectNone">None</button>
      </div>
    </div>
  </div>
  <select id="filterResult">
    <option value="">All results</option>
    <option value="agree">Agreement</option>
    <option value="disagree">Disagreement</option>
    <option value="error">Errors</option>
  </select>
  <input id="filterClass" placeholder="Filter by predicted class...">
  <input id="minConfidence" type="number" min="0" max="100" step="1" placeholder="Min confidence %">
  <select id="sortBy">
    <option value="">Sort: default order</option>
    <option value="confidence">Sort: confidence (high to low)</option>
    <option value="class">Sort: predicted class (A-Z)</option>
    <option value="result">Sort: result</option>
  </select>
</div>
<table id="reportTable">
<thead><tr><th>Image</th><th>Path</th>{header_cols}<th>Models Agree?</th></tr></thead>
<tbody>
{''.join(rows)}</tbody>
</table>
<h2>Per-class summary</h2>
<table><thead><tr><th>Model</th><th>Class id</th><th>Predicted class</th><th>Count</th></tr></thead><tbody>{summary_rows}</tbody></table>
<h2>Skipped files</h2>
<ul>{skipped_rows if skipped_rows else '<li>None</li>'}</ul>
<script>
{report_js}</script>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


# --- Publication: swapping the report directory into place safely ----------------


def process_is_running(pid: int) -> bool:
    """True when a process with this id still exists (it may be another run of
    this application mid-swap, whose backup must not be touched)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True  # exists but is not ours to signal
    return True


def _backup_pid(backup: Path) -> int | None:
    suffix = backup.name.rsplit("-", 1)[-1]
    return int(suffix) if suffix.isdigit() else None


@contextmanager
def publication_lock(output_dir: Path):
    """Serialize publication to one output directory.

    Recovery and the swap must happen under one lock: without it two runs can
    both observe "no other publisher", and the second then sees the momentary
    gap while the first has its output renamed aside. A lock left by a process
    that no longer exists is reclaimed."""
    # Scope: this guards the common case of a second run started by hand while
    # one is publishing. It is not a general mutual-exclusion primitive - two
    # processes can still both enter if they collide inside the microseconds
    # between creating this file and writing the pid into it. Closing that
    # window properly means an advisory flock(), which is only worth adding if
    # concurrent publication to one output_dir becomes a supported workflow; it
    # is not one today, and the README says so.
    lock_path = output_dir.parent / f".{output_dir.name}.lock"
    fd = None
    for attempt in (0, 1):
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            break
        except FileExistsError:
            holder = None
            try:
                holder = int(lock_path.read_text(encoding="utf-8").strip())
            except (OSError, ValueError):
                pass
            if attempt == 0 and (holder is None or holder == os.getpid()
                                 or not process_is_running(holder)):
                lock_path.unlink(missing_ok=True)  # stale: its owner is gone
                continue
            raise OSError(
                f"another run is publishing to {output_dir}; retry once it has finished"
            ) from None
    if fd is None:
        raise OSError(f"could not acquire the publication lock for {output_dir}")
    try:
        os.write(fd, f"{os.getpid()}\n".encode())
        os.close(fd)
        fd = None
        yield
    finally:
        if fd is not None:
            os.close(fd)
        lock_path.unlink(missing_ok=True)


def recover_interrupted_publish(output_dir: Path) -> bool:
    """Clean up after a publish that was killed part-way through its swap.

    publish_report renames the old report aside to `.<name>.previous-<pid>`,
    moves the new one into place, then deletes the backup. A process killed
    between those steps leaves either `output_dir` absent (restore the backup)
    or the backup orphaned (delete it). Backups belonging to a process that is
    still running are left alone, as is anything that does not carry the
    report marker.

    Returns True when a live process owns a backup, meaning a publish is in
    flight and this run must not touch output_dir."""
    parent = output_dir.parent
    if not parent.is_dir():
        return False

    # Staging directories of dead owners leak disk space: their `finally` cleanup
    # never ran. Ours is removed by publish_report itself.
    for staging in parent.glob(f".{output_dir.name}.staging-*"):
        if not staging.is_dir():
            continue
        pid = _backup_pid(staging)
        if pid is not None and pid != os.getpid() and process_is_running(pid):
            continue
        shutil.rmtree(staging, ignore_errors=True)
        print(f"Removed an abandoned report staging directory: {staging.name}", file=sys.stderr)

    backups = [p for p in parent.glob(f".{output_dir.name}.previous-*")
               if p.is_dir() and (p / REPORT_MARKER).is_file()]
    # A backup whose process is still alive belongs to a publish that is mid-swap:
    # it still needs that directory to roll back, and its own rename will fill
    # output_dir shortly. Never restore or delete those.
    # A backup bearing our own pid cannot belong to a concurrent invocation: it is
    # a stale one from a killed run whose pid the OS has since recycled onto us.
    # Treating it as live would leave it in place and then fail our own rename
    # onto that path, blocking every later run.
    def owned_by_live_process(backup: Path) -> bool:
        pid = _backup_pid(backup)
        return pid is not None and pid != os.getpid() and process_is_running(pid)

    live_owner = any(owned_by_live_process(p) for p in backups)
    abandoned = [p for p in backups if not owned_by_live_process(p)]
    if not abandoned:
        return live_owner

    if not output_dir.exists():
        newest = max(abandoned, key=lambda p: p.stat().st_mtime)
        newest.rename(output_dir)
        print(f"Recovered an interrupted report publication: restored {newest.name} to "
              f"{output_dir}", file=sys.stderr)
        abandoned.remove(newest)

    # Anything left belongs to a finished publish that never got to delete its
    # backup.
    for leftover in abandoned:
        shutil.rmtree(leftover, ignore_errors=True)
        print(f"Removed a leftover report backup: {leftover.name}", file=sys.stderr)
    return live_owner


def publish_report(output_dir: Path, results: list[ImageResult], profiles: list[ModelProfile],
                   skipped: list[str], class_summary: ClassSummary,
                   timing: dict[str, Any]) -> None:
    """Write the complete report into a sibling staging directory, then swap the
    whole `output_dir` for it. Readers never see a mixture of old and new files,
    and a failure at any point leaves the previous report in place.

    `output_dir` is owned by this application: an existing directory is only
    replaced if it is empty or carries the REPORT_MARKER written by a previous
    run and holds nothing else, so a customer directory (even one that happens
    to contain a `report.html`) can never be swapped away."""
    output_dir = output_dir.resolve()
    parent = output_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    with publication_lock(output_dir):
        _publish_locked(output_dir, results, profiles, skipped, class_summary, timing)


def _publish_locked(output_dir: Path, results: list[ImageResult], profiles: list[ModelProfile],
                    skipped: list[str], class_summary: ClassSummary,
                    timing: dict[str, Any]) -> None:
    """The body of publish_report, run while holding the publication lock."""
    parent = output_dir.parent
    if recover_interrupted_publish(output_dir):
        # Another run has output_dir renamed aside and will rename its own staging
        # into that path. Publishing now would take the path out from under it.
        raise OSError(
            f"another run is publishing to {output_dir}; retry once it has finished"
        )
    if output_dir.exists():
        if not output_dir.is_dir():
            raise OSError(f"output_dir {output_dir} exists and is not a directory")
        entries = sorted(p.name for p in output_dir.iterdir())
        foreign = [name for name in entries if name not in REPORT_ENTRIES]
        if entries and REPORT_MARKER not in entries:
            raise OSError(
                f"output_dir {output_dir} was not created by this application "
                f"(missing {REPORT_MARKER}); use an empty or dedicated directory"
            )
        if foreign:
            raise OSError(
                f"output_dir {output_dir} contains entries that are not part of a previous "
                f"report ({', '.join(foreign[:3])}); use a dedicated directory"
            )
    parent.mkdir(parents=True, exist_ok=True)

    tag = str(os.getpid())
    staging = parent / f".{output_dir.name}.staging-{tag}"
    previous = parent / f".{output_dir.name}.previous-{tag}"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir()
    try:
        write_json_report(staging / "report.json", results, profiles, skipped, class_summary, timing)
        write_csv_report(staging / "report.csv", results, profiles)
        write_html_report(staging / "report.html", results, profiles, skipped, class_summary, staging)
        (staging / REPORT_MARKER).write_text("generated by image-classification-explorer\n",
                                             encoding="utf-8")

        had_previous = output_dir.exists()
        if had_previous:
            output_dir.rename(previous)
        try:
            staging.rename(output_dir)
        except BaseException:
            # Includes KeyboardInterrupt: roll back to the previous report. A
            # hard kill between the two renames is recovered on the next run by
            # recover_interrupted_publish().
            if had_previous and not output_dir.exists():
                previous.rename(output_dir)
            raise
        if had_previous:
            shutil.rmtree(previous, ignore_errors=True)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


# --- Entrypoint ------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Image classification explorer")
    default_config = Path(__file__).resolve().parents[1] / "common" / "config.yaml"
    parser.add_argument("--config", type=Path, default=default_config, help="Path to YAML configuration")
    args = parser.parse_args()

    # Reading and interpreting the config all happens in one controlled error path,
    # so an unreadable file, invalid YAML, or a bad value reports `Invalid
    # configuration: ...` and exit 2 rather than a traceback (matching C++).
    #
    # Relative paths in config.yaml (model artifacts, label maps, output dir) resolve
    # against the current working directory, matching every other example in this repo:
    # customers run commands from the installed `prebuilt-apps/` root, not from here.
    try:
        raw = load_config(args.config)
        if not isinstance(raw, dict):
            raise ValueError(f"{args.config} must contain a YAML mapping at the top level")
        io_cfg = config_section(raw, "io")
        runtime = config_section(raw, "runtime")
        timeout_ms = config_int(runtime.get("timeout_ms"), "runtime.timeout_ms", 20000)
        if timeout_ms <= 0:
            raise ValueError(f"runtime.timeout_ms must be positive, got {timeout_ms}")
        # `extensions:` left null in YAML arrives as None; str(None) would turn the
        # accepted list into ("none",) and reject every real image.
        raw_extensions = config_str(io_cfg.get("extensions"), "io.extensions",
                                    ",".join(DEFAULT_EXTENSIONS))
        extensions = tuple(
            normalize_extension(e) for e in raw_extensions.split(",") if e.strip()
        ) or DEFAULT_EXTENSIONS
        fallback_url = config_str(
            io_cfg.get("fallback_image_url"), "io.fallback_image_url",
            "https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master/"
            "n01443537_goldfish.JPEG",
        )
        input_path = config_str(io_cfg.get("input"), "io.input", None)
        output_dir = Path(config_str(io_cfg.get("output_dir"), "io.output_dir", "report"))
        # P8: fail on a malformed validation block now, not after a successful run.
        validation = config_section(raw, "validation")
        expected_class_id = validation.get("expected_class_id")
        if expected_class_id is not None:
            expected_class_id = config_int(expected_class_id, "validation.expected_class_id", 0)
        min_probability = config_float(
            validation.get("min_probability"), "validation.min_probability", 0.0
        )
        profiles = load_profiles(raw)
        for profile in profiles:
            profile.labels = load_label_map(profile.label_map, profile.num_classes)
    except (OSError, yaml.YAMLError, TypeError, ValueError) as exc:
        print(f"Invalid configuration: {exc}", file=sys.stderr)
        return 2

    try:
        images, skipped = discover_images(
            input_path, extensions, fallback_url,
            # Honour TMPDIR, as the C++ side does via fs::temp_directory_path().
            # Same name the C++ side derives from sima_examples::default_goldfish_path(),
            # so one download serves both implementations.
            Path(tempfile.gettempdir()) / "sima_imagenet_goldfish.jpg",
        )
    except OSError as exc:  # missing path, unreadable directory, failed download
        print(str(exc), file=sys.stderr)
        return 3

    for entry in skipped:
        print(f"Skipping {entry}")

    results: list[ImageResult] = []
    total_ms = 0.0
    if not images:
        print("No images to classify.", file=sys.stderr)
    else:
        print(f"Classifying {len(images)} image(s) with {len(profiles)} model(s): "
              f"{', '.join(p.name for p in profiles)}")

        start = time.monotonic()
        try:
            results = run_all(profiles, images, timeout_ms)
        except Exception as exc:  # noqa: BLE001 - a whole-model failure (e.g. bad path) is fatal
            print(f"Error: {exc}", file=sys.stderr)
            return 6
        total_ms = (time.monotonic() - start) * 1000.0

    class_summary = build_class_summary(results, profiles)
    timing = {
        "total_ms": total_ms,
        "image_count": len(images),
        "model_count": len(profiles),
    }

    try:
        publish_report(output_dir, results, profiles, skipped, class_summary, timing)
    except OSError as exc:
        print(f"Error: failed to write report to {output_dir}: {exc}", file=sys.stderr)
        return 6

    images_with_errors = sum(1 for r in results if r.errors)
    print(f"Done in {total_ms:.1f} ms. Report written to {output_dir}")
    print("  report.html, report.json, report.csv")
    if images_with_errors:
        print(f"  {images_with_errors} image(s) had at least one model failure (see report.json)")

    used_fallback_sample = not input_path
    if expected_class_id is not None and used_fallback_sample and len(images) == 1 and profiles:
        first = results[0]
        pred = first.predictions.get(profiles[0].name)
        if pred and pred.top_k:
            top1_id = pred.top_k[0][0]
            top1_prob = pred.top_k[0][2]
            if top1_id != expected_class_id or top1_prob < min_probability:
                print(
                    f"Note: {profiles[0].name} top1={top1_id} ({top1_prob:.4f}) did not match "
                    f"expected_class_id={expected_class_id} (min_probability={min_probability}); "
                    "see report for details.",
                    file=sys.stderr,
                )

    return 0


if __name__ == "__main__":
    sys.exit(main())
