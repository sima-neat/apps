"""Local model setup, decoding and numerical helpers for this example."""
from __future__ import annotations

from ament_index_python.packages import get_package_share_directory
import argparse

import csv

import datetime

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

REPORT_MARKER = ".image-classification-explorer-report"

BUNDLED_LABEL_MAP_REF = "src/common/imagenet_labels.txt"

PROFILE_NAME_RE = re.compile(r"[A-Za-z0-9_-]+")

REPORT_ENTRIES = ("report.json", "report.csv", "report.html", "thumbnails", REPORT_MARKER)

THUMBNAIL_NAME = re.compile(r"[0-9a-f]{16}\.jpg\Z")

PROBABILITY_DECIMALS = 6

CONFIG_DIRECTORY: Path | None = None

INT32_MIN, INT32_MAX = -2**31, 2**31 - 1

ClassSummary = dict[str, dict[str, Any]]

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
    # The model's evaluation preprocessing: scale the shorter image side to
    # resize_shorter with this filter, then cut the centred input_width x
    # input_height window out of the result.
    resize_shorter: int = 256
    interpolation: str = "bilinear"

def stable_digest(value: str) -> str:
    """FNV-1a (64-bit), so thumbnail names are identical across runs, machines
    and both implementations. Python's built-in hash() is salted per process and
    would rename every thumbnail on each run."""
    digest = 0xCBF29CE484222325
    for byte in value.encode("utf-8"):
        digest = ((digest ^ byte) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"{digest:016x}"

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
        raise FileNotFoundError(f"failed to download fallback image: {url}: {exc}") from exc
    return dest

def decode_yaml_escapes(text: str) -> str:
    """Decode the escapes a double-quoted YAML scalar carries.

    PyYAML decodes them for a quoted value and leaves them literal for an
    unquoted one; ScalarConfig unquotes without decoding, so C++ never sees the
    difference. Decoding unconditionally on both sides is what makes the same
    file mean the same thing - the alternative leaves one spelling agreeing and
    the other not. The cost is that a literal backslash in a value is read as an
    escape, identically in both entrypoints."""
    # A single-quoted YAML scalar escapes an apostrophe by doubling it, so
    # `'report''s'` is `report's`. ScalarConfig strips the outer quotes and
    # leaves the pair, and the quote style is gone before either side sees it,
    # so both collapse unconditionally.
    text = text.replace("''", "'")
    out: list[str] = []
    i = 0
    simple = {"0": "\0", "a": "\a", "b": "\b", "t": "\t", "\t": "\t", "n": "\n",
              "v": "\v", "f": "\f", "r": "\r", "e": "\x1b", " ": " ",
              '"': '"', "/": "/", "\\": "\\"}
    while i < len(text):
        if text[i] != "\\" or i + 1 >= len(text):
            out.append(text[i])
            i += 1
            continue
        esc = text[i + 1]
        if esc in simple:
            out.append(simple[esc])
            i += 2
            continue
        width = {"x": 2, "u": 4, "U": 8}.get(esc)
        digits = text[i + 2:i + 2 + width] if width else ""
        if width and len(digits) == width and all(ch in "0123456789abcdefABCDEF" for ch in digits):
            code = int(digits, 16)
            # The code point itself, as PyYAML decodes the same escape inside a
            # double-quoted scalar. This used to substitute "?" above ASCII, to
            # mirror a C++ decoder that could not emit a wider code point - but
            # that one encodes UTF-8 now, so keeping the substitution here made
            # `io.output_dir: report\u00e9` publish to `reporté` in C++ and
            # `report?` in Python: the same divergence, the other way round.
            #
            # A lone surrogate is not a character and nothing can encode it, so
            # it keeps the old marker, as does a value beyond Unicode - both are
            # rejected downstream by their real spelling.
            if code <= 0x7F or (code <= 0x10FFFF and not 0xD800 <= code <= 0xDFFF):
                out.append(chr(code))
            else:
                out.append("?")
            i += 2 + width
            continue
        out.append(text[i])
        i += 1
    return "".join(out)

def parse_yaml_float(text: str) -> float | None:
    """The fourth YAML scalar type, matched the way PyYAML matches it.

    A '.' or an exponent is required so this never claims a value the integer
    reader owns, and the whole string must match so a path like `models/v1.0` or
    an extension `.jpg` is left alone."""
    body = text.strip()
    if not body:
        return None
    # YAML spells these with a leading dot, and only with it: `.nan` is a float
    # while a bare `nan` stays a string.
    special = {".nan": float("nan"), ".inf": float("inf"), "+.inf": float("inf"),
               "-.inf": float("-inf")}
    if body.lower() in special:
        return special[body.lower()]
    if body[0] in "+-":
        body = body[1:]
    body = body.replace("_", "")
    if not body or ("." not in body and "e" not in body.lower()):
        return None
    try:
        value = float(body)
    except ValueError:
        return None
    return -value if text.strip()[0] == "-" else value

def parse_yaml_bool(text: str) -> bool | None:
    """Recognise the YAML 1.1 Boolean spellings PyYAML resolves.

    PyYAML turns `yes` into True and Python renders that as "true", while
    ScalarConfig passes the word `yes` through unchanged. Both sides canonicalise
    so the same file names the same thing."""
    lowered = text.strip().lower()
    if lowered in {"true", "yes", "on"}:
        return True
    if lowered in {"false", "no", "off"}:
        return False
    return None

def parse_yaml_int(text: str) -> int | None:
    """Parse an integer with the YAML 1.1 rules PyYAML applies to a bare scalar.

    A quoted `"010"` reaches this function as text, and ScalarConfig cannot see
    the quotes, so C++ reads every value as YAML and gets 8. Reading it as base
    10 here would make the same file mean 10 to one entrypoint and 8 to the
    other."""
    body = text.strip()
    if not body:
        return None
    negative = body[0] == "-"
    if body[0] in "+-":
        body = body[1:]
    body = body.replace("_", "")
    if not body:
        return None
    # YAML 1.1 sexagesimal: `1:20` is 80, which PyYAML resolves, so a quoted
    # spelling of the same value must mean the same thing here.
    if ":" in body:
        # PyYAML's grammar exactly: [1-9][0-9]* then one or more :[0-5]?[0-9].
        # So `1:20` is 80 while `1:60`, `1:99` and `0:30` stay strings.
        parts = body.split(":")
        if not parts[0] or not parts[0].isdigit() or parts[0][0] == "0":
            return None
        total = int(parts[0])
        for part in parts[1:]:
            if not part.isdigit() or len(part) > 2 or int(part) > 59:
                return None
            total = total * 60 + int(part)
        return -total if negative else total
    base = 10
    lowered = body.lower()
    if lowered.startswith(("0x", "0b", "0o")):
        base = {"x": 16, "b": 2, "o": 8}[lowered[1]]
        body = body[2:]
    elif len(body) > 1 and body[0] == "0":
        base = 8  # YAML 1.1 bare octal
    if not body:
        return None
    try:
        parsed = int(body, base)
    except ValueError:
        return None
    return -parsed if negative else parsed

def yaml_text(value: Any) -> str:
    """Render a parsed YAML value as the text C++ reads from the same file.

    Error messages name the offending value, and repr() is the wrong renderer
    for that: it writes a string in quotes and a bool as `True`, neither of
    which is what the customer wrote or what the C++ entrypoint prints for the
    same file. A collection is named by its shape, because ScalarConfig keeps
    `[a,b]` verbatim while PyYAML hands back a parsed list - the two can agree
    on the shape but never on the spelling."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (datetime.date, datetime.datetime)):
        # PyYAML resolves an unquoted `2026-10-02` to a date; ScalarConfig hands
        # C++ the text. isoformat() is that text, so a dated output directory
        # names the same thing in both instead of being rejected here.
        return value.isoformat()
    if isinstance(value, (list, tuple)):
        return "a list"
    if isinstance(value, dict):
        return "a mapping"
    return str(value)

def config_int(value: Any, key: str, default: int) -> int:
    """Read an integer config value the way the C++ ScalarConfig does.

    PyYAML hands back real Python objects, so `top_k: 1.9` arrives as a float
    that int() would silently truncate to 1 while C++ rejects the same file as
    non-integral. Accept only genuine integers (or their exact text form) so
    both entrypoints run the same settings."""
    if value is None:
        return default
    if isinstance(value, bool):
        raise ValueError(f"{key} must be an integer, got {yaml_text(value)}")
    if isinstance(value, int):
        return _check_int32(value, key)
    if isinstance(value, str):
        decoded = scalar_text(value)
        if decoded is None:
            return default
        parsed = parse_yaml_int(decoded)
        if parsed is None:
            raise ValueError(f"{key} must be an integer, got {yaml_text(value)}")
        return _check_int32(parsed, key)
    raise ValueError(f"{key} must be an integer, got {yaml_text(value)}")

def _check_int32(value: int, key: str) -> int:
    if not INT32_MIN <= value <= INT32_MAX:
        raise ValueError(f"{key} is out of range for a 32-bit integer: {value}")
    return value

def normalize_extension(value: str) -> str:
    """Accept `jpg` as well as `.jpg`. Path.suffix always carries the dot, so an
    entry without one would silently match nothing."""
    text = value.strip().lower()
    return text if text.startswith(".") else f".{text}"

def scalar_text(value: str) -> str | None:
    """Apply ScalarConfig's scalar rules to a string, or None when it reads as null.

    One place, because every reader needs it: C++ takes its text from
    config_scalar, so decoding or null-handling in only the string path left the
    integer and float settings disagreeing."""
    decoded = decode_yaml_escapes(value).strip()
    if decoded == "~" or decoded.lower() == "null":
        return None
    return decoded

def config_section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    """Read a top-level section.

    Absent or null means "use the defaults"; anything else must be a mapping.
    `raw.get(name) or {}` cannot make that distinction, because a falsy scalar
    such as `io: false` or `validation: 0` collapses to an empty mapping and the
    defaults are applied to a configuration the customer got wrong."""
    value = raw.get(name)
    if value is None:
        return {}
    # ScalarConfig unquotes before testing for null, so `io: "null"` is absent to
    # C++ and the defaults apply. Python saw a string and rejected the file.
    if isinstance(value, str) and scalar_text(value) is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"`{name}` must be a mapping, got {yaml_text(value)}")
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
    if isinstance(value, (datetime.date, datetime.datetime)):
        return value.isoformat()
    if isinstance(value, str):
        text = scalar_text(value)
        if text is None:
            return default
        if "\0" in text:
            # YAML permits "\0", the filesystem does not. Rejected while reading
            # the configuration rather than left to the path APIs: Path.resolve()
            # raises a bare ValueError that escaped as a traceback and exit 1,
            # outside the documented 2/3/6 contract, and C++ silently truncated
            # the value at the NUL and wrote the report to a directory the
            # customer had not named.
            raise ValueError(f"{key} contains a NUL character; remove it from config.yaml")
        if len(text) >= 2 and ((text[0] == "[" and text[-1] == "]")
                               or (text[0] == "{" and text[-1] == "}")):
            # C++ cannot tell this from an unquoted flow collection, which both
            # reject, so a quoted one is rejected here too rather than accepted
            # by one entrypoint only.
            shape = "a list" if text[0] == "[" else "a mapping"
            raise ValueError(f"{key} must be a scalar, got {shape}")
        # Likewise C++ cannot tell `010` from `"010"`; both canonicalise, so
        # both name the same directory either way.
        number = parse_yaml_int(text)
        if number is not None:
            return str(number)
        number = parse_yaml_float(text)
        if number is not None:
            return str(number)
        flag = parse_yaml_bool(text)
        if flag is not None:
            return "true" if flag else "false"
        return text
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    raise ValueError(f"{key} must be a scalar, got {yaml_text(value)}")

REPORT_FORMATS = ("html", "json", "csv")

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
                f"models: profile name {yaml_text(raw_name)} is not a string; quote it in config.yaml"
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
        # Without resize_shorter the window keeps the standard ImageNet 87.5%
        # centre crop (256 for 224), in integer arithmetic C++ repeats exactly.
        profile.resize_shorter = config_int(
            cfg.get("resize_shorter"), f"models.{name}.resize_shorter",
            default_resize_shorter(profile.input_width, profile.input_height),
        )
        profile.interpolation = config_str(
            cfg.get("interpolation"), f"models.{name}.interpolation", "bilinear"
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
                f"models.{name}.preprocess={yaml_text(profile.preprocess)} is not supported; "
                "only 'imagenet' is implemented"
            )
        if profile.output != "softmax":
            raise ValueError(
                f"models.{name}.output={yaml_text(profile.output)} is not supported; "
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
        if profile.input_width > MAX_INPUT_DIMENSION or profile.input_height > MAX_INPUT_DIMENSION:
            raise ValueError(
                f"models.{name}.input_width/input_height exceeds the supported maximum of "
                f"{MAX_INPUT_DIMENSION}, got {profile.input_width}x{profile.input_height}; "
                f"check the value against the model's documented input size"
            )
        if profile.interpolation not in INTERPOLATIONS:
            raise ValueError(
                f"models.{name}.interpolation={yaml_text(profile.interpolation)} is not "
                f"supported; use one of: {', '.join(INTERPOLATIONS)}"
            )
        if not max(profile.input_width, profile.input_height) <= profile.resize_shorter <= MAX_INPUT_DIMENSION:
            raise ValueError(
                f"models.{name}.resize_shorter must be between the larger input side "
                f"({max(profile.input_width, profile.input_height)}) and {MAX_INPUT_DIMENSION}, "
                f"got {profile.resize_shorter}"
            )
        profiles.append(profile)
    return profiles

MAX_NUMERIC_LABELS = 100_000

MAX_INPUT_DIMENSION = 8192

INTERPOLATIONS = {"bilinear": "INTER_LINEAR", "bicubic": "INTER_CUBIC"}

def default_resize_shorter(width: int, height: int) -> int:
    """The shorter side for the standard ImageNet 87.5% centre crop: 256 for 224.

    Taken from the larger input side, so a non-square window fits images of
    either orientation, and capped at MAX_INPUT_DIMENSION so every accepted
    input size gets a valid default.
    """
    return min(max(width, height) * 8 // 7, MAX_INPUT_DIMENSION)

def load_label_map(path: str | None, num_classes: int) -> list[str]:
    if not path:
        if num_classes > MAX_NUMERIC_LABELS:
            raise ValueError(
                f"num_classes={num_classes} exceeds the supported maximum of "
                f"{MAX_NUMERIC_LABELS} for generated numeric labels; supply a "
                f"label_map, or check the value"
            )
        return [str(i) for i in range(num_classes)]
    label_path = Path(path)
    if not label_path.exists() and Path(path).as_posix() == BUNDLED_LABEL_MAP_REF:
        # Resolve the shipped reference next to this script, regardless of the
        # caller's cwd (model.path stays cwd-relative: it points at a file the
        # customer downloaded). A missing custom path is NOT redirected here.
        for bundled in bundled_candidates(Path(BUNDLED_LABEL_MAP_REF).name):
            if bundled.exists():
                label_path = bundled
                break
    try:
        raw = label_path.read_bytes()
    except OSError as exc:
        # Worded and detailed as the C++ entrypoint words it: strerror alone,
        # because str(exc) would repeat the path that already precedes it.
        raise ValueError(
            f"failed to open label map {label_path}: {exc.strerror or exc}"
        ) from exc
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        # Decoded from bytes rather than through an encoding= handle so the
        # reported offset is absolute in the file and not into whatever chunk
        # the text reader happened to buffer. The message replaces the raw codec
        # one ("'utf-8' codec can't decode byte ... in position ..."), which the
        # C++ entrypoint cannot reproduce and which does not name the file.
        raise ValueError(
            f"label map {label_path} is not valid UTF-8 "
            f"(byte 0x{raw[exc.start]:02x} at offset {exc.start}); save the file as UTF-8"
        ) from exc
    # Positional: physical line index == class id, so blank lines are never
    # dropped (that would silently shift every later label).
    labels = [line.strip() for line in text.splitlines()]
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

def is_utf8_path(path: Path) -> bool:
    """Whether the whole path round-trips as UTF-8.

    `os.fsencode` recovers the bytes the filesystem actually holds, including
    the ones Python represents as surrogate escapes; a strict decode then says
    whether they are text both entrypoints can agree on.
    """
    try:
        os.fsencode(path).decode("utf-8")
    except (UnicodeDecodeError, UnicodeEncodeError, ValueError):
        return False
    return True

def _printable_path(path: Path) -> str:
    """The path as text safe to put in a report that declares itself UTF-8."""
    return os.fsencode(path).decode("utf-8", "replace")

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
        if not is_utf8_path(entry):
            # A name the filesystem holds as bytes that are not UTF-8. Python
            # carries those through surrogate escapes, and handing such a str to
            # the native decoder segfaulted the interpreter mid-run; C++ streamed
            # the raw bytes into a document declared UTF-8 and exited 0 with a
            # mangled name. Skipping is what both can agree on, and it is how
            # every other unusable entry in the directory is already treated.
            skipped.append(f"{_printable_path(entry)}: file name is not valid UTF-8")
            continue
        if entry.suffix.lower() not in extensions:
            skipped.append(f"{entry}: unsupported extension {entry.suffix or '(none)'}")
            continue
        images.append(entry)

    if not images and not skipped:
        raise FileNotFoundError(f"no image files found under {path}")
    return images, skipped

def softmax(scores):
    """Softmax computed exactly as sima_examples::topk_with_softmax does.

    The C++ helper subtracts the float32 maximum, evaluates exp in double and
    accumulates the denominator sequentially. NumPy would use pairwise
    summation and, with float32 exps, drift from C++ by around 1e-7 - small,
    but enough to change a reported value at the sixth decimal for some inputs.
    Mirroring the arithmetic makes the two reports agree because they computed
    the same thing, not because a rounding step happened to hide the gap."""
    import numpy as np

    shifted = (scores - np.max(scores)).astype(np.float64)
    exponentials = np.exp(shifted)
    total = 0.0
    for value in exponentials:  # sequential, matching the C++ accumulation order
        total += float(value)
    # ScoredIndex::prob is a float, so C++ narrows each probability to float32
    # before reporting it. Narrowing here too is what makes the sixth decimal
    # agree; without it the two differ for roughly one value in 130.
    return (exponentials / total).astype(np.float32)

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

    # preprocess is validated in load_profiles, before any input is read.
    opt = pyneat.ModelOptions()
    opt.preprocess.kind = pyneat.InputKind.Image
    opt.preprocess.color_convert.input_format = pyneat.PreprocessColorFormat.RGB
    opt.preprocess.input_max_width = profile.input_width
    opt.preprocess.input_max_height = profile.input_height
    opt.preprocess.input_max_depth = 3
    opt.preprocess.preset = pyneat.NormalizePreset.ImageNet
    return pyneat.Model(profile.path, opt)

def bundled_candidates(name: str) -> list[Path]:
    """Where a file shipped under src/common may be found.

    Beside this script covers every layout Python runs in. The directory of the
    configuration in use is listed too, because config.yaml ships in src/common
    beside these files: it is what lets the C++ binary find them from any
    working directory, and both implementations searching the same places keeps
    them aligned."""
    candidates = []
    if CONFIG_DIRECTORY is not None:
        candidates.append(CONFIG_DIRECTORY / name)
    candidates.append(Path(get_package_share_directory('neat_image_classification_explorer')) / "common" / name)
    return candidates
