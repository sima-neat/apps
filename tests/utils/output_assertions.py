"""Shared assertions for what an application actually produced.

Most e2e suites check that the process exited 0 and that some non-empty files
appeared. Neither says the application worked: a truncated write, a decoder
that emitted nothing but grey, or a pipeline stuck on one frame all satisfy
`st_size > 0`. These helpers assert on the content instead, and are written to
be reusable so each application's suite stays about that application.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Sequence

import cv2

# The multi-stream applications name every frame after the stream that produced
# it, e.g. stream_1_frame_40.jpg, and write them all into one directory.
_STREAM_IN_NAME = re.compile(r"^stream[_-]?(\d+)[_-]")

# What the directory-based applications accept, from their shared is_image().
_IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp"})


def supported_image_files(input_dir: Path) -> list[Path]:
    """The images a directory-based application will process, in its own order.

    Suites size their output expectation with this rather than with a constant,
    so that an application which annotates the first image and exits cleanly is
    not mistaken for one that processed the directory.
    """
    return sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
    )


def saved_image_files(output_dir: Path, *, exclude: Iterable[str] = ("config.yaml",)) -> list[Path]:
    """Every file an e2e run saved, newest-last, excluding config artifacts."""
    skip = set(exclude)
    files = [
        path
        for path in sorted(output_dir.rglob("*"))
        if path.is_file() and path.name not in skip
    ]
    return files


def assert_frames_decode(paths: Sequence[Path], *, min_side: int = 16) -> list["cv2.typing.MatLike"]:
    """Every saved frame is a decodable image of plausible size.

    `st_size > 0` passes on a truncated JPEG; this does not. Returns the decoded
    frames so a caller can assert further without re-reading them.
    """
    assert paths, "no output frames were saved"

    frames = []
    for path in paths:
        image = cv2.imread(str(path))
        assert image is not None, (
            f"{path.name} is {path.stat().st_size} bytes but does not decode as an image"
        )
        height, width = image.shape[:2]
        assert width >= min_side and height >= min_side, (
            f"{path.name} decoded to an implausible {width}x{height}"
        )
        frames.append(image)
    return frames


def group_frames_by_stream(
    paths: Sequence[Path], frames: Sequence["cv2.typing.MatLike"]
) -> dict[str, list[tuple[Path, "cv2.typing.MatLike"]]]:
    """Split saved frames into one group per stream, keeping filename order.

    Frames that carry no stream in their name — the single-stream and
    still-image suites — stay together as one group, which is the whole run.
    """
    groups: dict[str, list[tuple[Path, "cv2.typing.MatLike"]]] = {}
    for path, frame in zip(paths, frames):
        match = _STREAM_IN_NAME.match(path.name)
        label = f"stream {match.group(1)}" if match else "the run"
        groups.setdefault(label, []).append((path, frame))
    return groups


def assert_every_stream_advances(
    frames: Sequence["cv2.typing.MatLike"], paths: Sequence[Path]
) -> None:
    """Every stream produced moving video, not one frame written repeatedly.

    A pipeline that stalls on its first frame, or a decoder handing back the
    same buffer every pull, still writes the expected number of files. This is
    the cheapest assertion that separates that from a working stream.

    Each stream is compared only against itself. A multi-stream run writes all
    of its streams into one directory, so comparing the directory as a whole
    would be satisfied the moment any one stream moved, and a stream frozen
    beside a working one — the failure this is here to catch — would pass.

    Assumes the source has motion, which is true of the looping test streams
    the e2e configuration points at. A deliberately static source would need
    this check skipped rather than loosened.
    """
    for label, group in group_frames_by_stream(paths, frames).items():
        if len(group) < 2:
            continue

        first = group[0][1]
        if any(
            frame.shape != first.shape or cv2.norm(frame, first, cv2.NORM_L1) > 0
            for _, frame in group[1:]
        ):
            continue

        names = ", ".join(path.name for path, _ in group[:4])
        raise AssertionError(
            f"all {len(group)} frames saved for {label} are pixel-identical "
            f"({names}...); the pipeline is not advancing through the stream"
        )


def assert_saved_frames_are_usable(
    output_dir: Path, minimum: int, *, min_side: int = 16
) -> list[Path]:
    """The output check for an application that annotates a directory of images.

    Replaces the common `len(files) >= n` plus `st_size > 0` pair with the same
    count check followed by assertions that the frames decode. Nothing here
    asks the frames to differ: a batch of images is whatever the caller pointed
    the application at, and two identical inputs correctly produce two identical
    outputs. What a batch run does owe is one output per input, which is why
    `minimum` should come from `supported_image_files(input_dir)`.
    """
    paths, _ = _counted_and_decoded(output_dir, minimum, min_side)
    return paths


def assert_streamed_frames_are_usable(
    output_dir: Path, minimum: int, *, min_side: int = 16
) -> list[Path]:
    """The same check for an application that samples frames out of a stream.

    Adds the assertion a stream can be held to and a batch cannot: the frames
    have to move. The e2e configuration points these suites at looping test
    streams, so every stream is expected to advance; a deliberately static
    source would need this call replaced rather than loosened.
    """
    paths, frames = _counted_and_decoded(output_dir, minimum, min_side)
    assert_every_stream_advances(frames, paths)
    return paths


def _counted_and_decoded(
    output_dir: Path, minimum: int, min_side: int
) -> tuple[list[Path], list["cv2.typing.MatLike"]]:
    paths = saved_image_files(output_dir)
    assert len(paths) >= minimum, (
        f"expected at least {minimum} saved frames, got {len(paths)}"
    )
    return paths, assert_frames_decode(paths, min_side=min_side)
