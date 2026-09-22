"""Shared assertions for what an application actually produced.

Most e2e suites check that the process exited 0 and that some non-empty files
appeared. Neither says the application worked: a truncated write, a decoder
that emitted nothing but grey, or a pipeline stuck on one frame all satisfy
`st_size > 0`. These helpers assert on the content instead, and are written to
be reusable so each application's suite stays about that application.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import cv2


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


def assert_frames_are_not_all_identical(
    frames: Sequence["cv2.typing.MatLike"], paths: Sequence[Path]
) -> None:
    """The run produced moving video, not one frame written repeatedly.

    A pipeline that stalls on its first frame, or a decoder handing back the
    same buffer every pull, still writes the expected number of files. This is
    the cheapest assertion that separates that from a working stream.

    Assumes the source has motion, which is true of the looping test streams
    the e2e configuration points at. A deliberately static source would need
    this check skipped rather than loosened.
    """
    if len(frames) < 2:
        return

    first = frames[0]
    for index, frame in enumerate(frames[1:], start=1):
        if frame.shape != first.shape or cv2.norm(frame, first, cv2.NORM_L1) > 0:
            return

    names = ", ".join(path.name for path in paths[:4])
    raise AssertionError(
        f"all {len(frames)} saved frames are pixel-identical ({names}...); "
        "the pipeline is not advancing through the stream"
    )


def assert_saved_frames_are_usable(
    output_dir: Path, minimum: int, *, min_side: int = 16
) -> list[Path]:
    """The whole output check for an application that saves annotated frames.

    Replaces the common `len(files) >= n` plus `st_size > 0` pair with the same
    count check followed by assertions that the frames decode and advance.
    """
    paths = saved_image_files(output_dir)
    assert len(paths) >= minimum, (
        f"expected at least {minimum} saved frames, got {len(paths)}"
    )
    frames = assert_frames_decode(paths, min_side=min_side)
    assert_frames_are_not_all_identical(frames, paths)
    return paths
