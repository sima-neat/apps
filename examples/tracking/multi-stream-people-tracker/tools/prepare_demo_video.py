#!/usr/bin/env python3
"""Build an Insight-compatible H.264 demo video from an image sequence."""

from __future__ import annotations

import argparse
from collections import Counter
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
import tempfile
from typing import Any, Sequence


SUPPORTED_FPS = (30, 60)
SUPPORTED_SUFFIXES = {".jpeg", ".jpg", ".pgm", ".png", ".ppm"}
JPEG_SOF_MARKERS = {
    0xC0,
    0xC1,
    0xC2,
    0xC3,
    0xC5,
    0xC6,
    0xC7,
    0xC9,
    0xCA,
    0xCB,
    0xCD,
    0xCE,
    0xCF,
}


class VideoPreparationError(RuntimeError):
    """An actionable input, encoder, or output-contract error."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Encode a numbered image sequence as deterministic, low-latency "
            "H.264 suitable for Insight."
        )
    )
    parser.add_argument(
        "--frames-dir",
        required=True,
        type=Path,
        help="Directory containing one numbered JPEG, PNG, PGM, or PPM per frame.",
    )
    parser.add_argument(
        "--output", required=True, type=Path, help="Destination .mp4 path."
    )
    parser.add_argument(
        "--fps",
        type=int,
        choices=SUPPORTED_FPS,
        default=30,
        help="Output frame rate (default: 30).",
    )
    parser.add_argument(
        "--source-id",
        help="Optional dataset/sequence identifier recorded in provenance.",
    )
    parser.add_argument(
        "--source-reference",
        help="Optional source URL or citation recorded in provenance.",
    )
    parser.add_argument(
        "--force", action="store_true", help="Replace existing output and provenance."
    )
    return parser.parse_args(argv)


def _last_numeric_index(path: Path) -> int:
    matches = list(re.finditer(r"\d+", path.stem))
    if not matches:
        raise VideoPreparationError(
            f"frame filename has no numeric index: {path.name}"
        )
    return int(matches[-1].group())


def discover_frames(frames_dir: Path) -> tuple[list[Path], int]:
    if not frames_dir.is_dir():
        raise VideoPreparationError(f"frames directory does not exist: {frames_dir}")

    candidates = sorted(
        (
            path
            for path in frames_dir.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
        ),
        key=lambda path: (_last_numeric_index(path), path.name),
    )
    if not candidates:
        supported = ", ".join(sorted(SUPPORTED_SUFFIXES))
        raise VideoPreparationError(
            f"no supported frames found in {frames_dir} (expected {supported})"
        )

    indexed = [(_last_numeric_index(path), path) for path in candidates]
    indices = [index for index, _ in indexed]
    if len(set(indices)) != len(indices):
        duplicates = sorted(
            index for index, count in Counter(indices).items() if count > 1
        )
        raise VideoPreparationError(
            "duplicate numeric frame indices: " + ", ".join(map(str, duplicates))
        )

    expected = list(range(indices[0], indices[0] + len(indices)))
    if indices != expected:
        missing = sorted(set(expected) - set(indices))
        detail = ", ".join(map(str, missing[:10]))
        if len(missing) > 10:
            detail += ", ..."
        raise VideoPreparationError(f"frame index sequence has gaps; missing: {detail}")

    formats = {
        ".jpg" if path.suffix.lower() in {".jpg", ".jpeg"} else path.suffix.lower()
        for path in candidates
    }
    if len(formats) != 1:
        raise VideoPreparationError(
            "all frames must use one image format; found: "
            + ", ".join(sorted(formats))
        )
    return candidates, indices[0]


def _jpeg_dimensions(path: Path) -> tuple[int, int]:
    data = path.read_bytes()
    if not data.startswith(b"\xff\xd8"):
        raise VideoPreparationError(f"invalid JPEG signature: {path}")

    offset = 2
    while offset + 4 <= len(data):
        if data[offset] != 0xFF:
            offset += 1
            continue
        while offset < len(data) and data[offset] == 0xFF:
            offset += 1
        if offset >= len(data):
            break
        marker = data[offset]
        offset += 1
        if marker in {0x01, *range(0xD0, 0xD9)}:
            continue
        if marker in {0xD9, 0xDA} or offset + 2 > len(data):
            break
        segment_length = struct.unpack_from(">H", data, offset)[0]
        if segment_length < 2 or offset + segment_length > len(data):
            break
        if marker in JPEG_SOF_MARKERS and segment_length >= 7:
            height, width = struct.unpack_from(">HH", data, offset + 3)
            return width, height
        offset += segment_length
    raise VideoPreparationError(f"JPEG dimensions were not found: {path}")


def _png_dimensions(path: Path) -> tuple[int, int]:
    header = path.read_bytes()[:24]
    if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        raise VideoPreparationError(f"invalid PNG signature: {path}")
    if header[12:16] != b"IHDR":
        raise VideoPreparationError(f"PNG is missing its leading IHDR chunk: {path}")
    return struct.unpack(">II", header[16:24])


def _pnm_dimensions(path: Path) -> tuple[int, int]:
    data = path.read_bytes()[:65536]
    tokens: list[bytes] = []
    offset = 0
    while offset < len(data) and len(tokens) < 4:
        while offset < len(data) and chr(data[offset]).isspace():
            offset += 1
        if offset < len(data) and data[offset] == ord("#"):
            newline = data.find(b"\n", offset)
            if newline < 0:
                break
            offset = newline + 1
            continue
        end = offset
        while end < len(data) and not chr(data[end]).isspace():
            end += 1
        if end > offset:
            tokens.append(data[offset:end])
        offset = end

    if len(tokens) < 4 or tokens[0] not in {b"P5", b"P6"}:
        raise VideoPreparationError(f"invalid binary PGM/PPM header: {path}")
    try:
        return int(tokens[1]), int(tokens[2])
    except ValueError as error:
        raise VideoPreparationError(f"invalid PGM/PPM dimensions: {path}") from error


def image_dimensions(path: Path) -> tuple[int, int]:
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return _jpeg_dimensions(path)
    if suffix == ".png":
        return _png_dimensions(path)
    if suffix in {".pgm", ".ppm"}:
        return _pnm_dimensions(path)
    raise VideoPreparationError(f"unsupported frame format: {path}")


def validate_dimensions(frames: Sequence[Path]) -> tuple[int, int]:
    width, height = image_dimensions(frames[0])
    if width <= 0 or height <= 0:
        raise VideoPreparationError(f"invalid frame dimensions: {width}x{height}")
    if width % 2 or height % 2:
        raise VideoPreparationError(
            f"H.264 yuv420p requires even dimensions; found {width}x{height}"
        )

    for frame in frames[1:]:
        current = image_dimensions(frame)
        if current != (width, height):
            raise VideoPreparationError(
                f"frame dimensions differ: {frame.name} is {current[0]}x{current[1]}, "
                f"expected {width}x{height}"
            )
    return width, height


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ordered_frames_sha256(frames: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for frame in frames:
        name = frame.name.encode("utf-8")
        digest.update(len(name).to_bytes(8, "big"))
        digest.update(name)
        size = frame.stat().st_size
        digest.update(size.to_bytes(8, "big"))
        with frame.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _required_executable(name: str) -> str:
    executable = shutil.which(name)
    if executable is None:
        raise VideoPreparationError(
            f"required SDK executable is not installed or not on PATH: {name}"
        )
    return executable


def _run_json(command: Sequence[str]) -> dict[str, Any]:
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise VideoPreparationError(
            f"command failed ({result.returncode}): {' '.join(command)}\n"
            f"{result.stderr.strip()}"
        )
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise VideoPreparationError(
            f"command returned invalid JSON: {' '.join(command)}"
        ) from error


def _probe_video(
    ffprobe: str, video: Path
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    stream_document = _run_json(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            (
                "stream=codec_name,profile,level,pix_fmt,r_frame_rate,avg_frame_rate,"
                "nb_frames,has_b_frames,width,height"
            ),
            "-of",
            "json",
            str(video),
        ]
    )
    streams = stream_document.get("streams", [])
    if len(streams) != 1:
        raise VideoPreparationError(
            f"expected exactly one video stream, found {len(streams)}"
        )

    frame_document = _run_json(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_frames",
            "-show_entries",
            "frame=key_frame,pict_type",
            "-of",
            "json",
            str(video),
        ]
    )
    packet_document = _run_json(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_packets",
            "-show_entries",
            "packet=pts,dts,duration,flags",
            "-of",
            "json",
            str(video),
        ]
    )
    return streams[0], frame_document.get("frames", []), packet_document.get(
        "packets", []
    )


def _annex_b_nal_types(ffmpeg: str, video: Path) -> list[int]:
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video),
        "-map",
        "0:v:0",
        "-c:v",
        "copy",
        "-bsf:v",
        "h264_mp4toannexb",
        "-f",
        "h264",
        "-",
    ]
    result = subprocess.run(command, capture_output=True)
    if result.returncode != 0:
        detail = result.stderr.decode(errors="replace")
        raise VideoPreparationError(
            f"could not inspect H.264 access units: {detail}"
        )

    data = result.stdout
    starts: list[tuple[int, int]] = []
    offset = 0
    while offset + 3 <= len(data):
        if data[offset : offset + 4] == b"\x00\x00\x00\x01":
            starts.append((offset, 4))
            offset += 4
        elif data[offset : offset + 3] == b"\x00\x00\x01":
            starts.append((offset, 3))
            offset += 3
        else:
            offset += 1
    return [
        data[start + prefix] & 0x1F
        for start, prefix in starts
        if start + prefix < len(data)
    ]


def _access_units(nal_types: Sequence[int]) -> list[list[int]]:
    units: list[list[int]] = []
    current: list[int] = []
    for nal_type in nal_types:
        if nal_type == 9:
            if current:
                units.append(current)
            current = [nal_type]
        elif current:
            current.append(nal_type)
    if current:
        units.append(current)
    return units


def validate_encoded_video(
    ffmpeg: str,
    ffprobe: str,
    video: Path,
    *,
    fps: int,
    frame_count: int,
    width: int,
    height: int,
) -> dict[str, Any]:
    stream, frames, packets = _probe_video(ffprobe, video)
    expected_rate = f"{fps}/1"
    expected_keys = list(range(0, frame_count, fps))
    actual_keys = [
        index
        for index, frame in enumerate(frames)
        if int(frame.get("key_frame", 0)) == 1
    ]

    requirements = {
        "codec_name": "h264",
        "level": 31,
        "pix_fmt": "yuv420p",
        "r_frame_rate": expected_rate,
        "avg_frame_rate": expected_rate,
        "has_b_frames": 0,
        "width": width,
        "height": height,
    }
    for field, expected in requirements.items():
        if stream.get(field) != expected:
            raise VideoPreparationError(
                f"encoded video violates {field}: {stream.get(field)!r} != {expected!r}"
            )
    if stream.get("profile") not in {"Baseline", "Constrained Baseline"}:
        raise VideoPreparationError(
            f"encoded video is not H.264 baseline: {stream.get('profile')!r}"
        )
    if len(frames) != frame_count or len(packets) != frame_count:
        raise VideoPreparationError(
            f"encoded frame count mismatch: frames={len(frames)}, "
            f"packets={len(packets)}, "
            f"expected={frame_count}"
        )
    if any(frame.get("pict_type") not in {"I", "P"} for frame in frames):
        raise VideoPreparationError("encoded video contains a picture other than I/P")
    if actual_keys != expected_keys:
        raise VideoPreparationError(
            f"keyframe cadence mismatch: {actual_keys} != {expected_keys}"
        )

    try:
        pts = [int(packet["pts"]) for packet in packets]
        dts = [int(packet["dts"]) for packet in packets]
    except (KeyError, TypeError, ValueError) as error:
        raise VideoPreparationError(
            "encoded packets are missing integer PTS/DTS"
        ) from error
    if pts != dts:
        raise VideoPreparationError("encoded video has reordered packets (PTS != DTS)")
    increments = {right - left for left, right in zip(pts, pts[1:])}
    if len(increments) > 1 or (increments and next(iter(increments)) <= 0):
        raise VideoPreparationError("encoded packet timestamps are not evenly spaced")

    units = _access_units(_annex_b_nal_types(ffmpeg, video))
    if len(units) != frame_count or any(not unit or unit[0] != 9 for unit in units):
        raise VideoPreparationError(
            f"expected one AUD-delimited access unit per frame; found {len(units)}"
        )
    idr_indices = [index for index, unit in enumerate(units) if 5 in unit]
    if idr_indices != expected_keys:
        raise VideoPreparationError(
            f"IDR cadence mismatch: {idr_indices} != {expected_keys}"
        )
    for index in idr_indices:
        idr_unit = units[index]
        idr_position = idr_unit.index(5)
        if 7 not in idr_unit[:idr_position] or 8 not in idr_unit[:idr_position]:
            raise VideoPreparationError(
                f"IDR access unit {index} does not repeat SPS/PPS before its slice"
            )

    return {
        "codec_name": stream["codec_name"],
        "profile": stream["profile"],
        "level": stream["level"],
        "pixel_format": stream["pix_fmt"],
        "width": stream["width"],
        "height": stream["height"],
        "frame_rate": stream["avg_frame_rate"],
        "frame_count": frame_count,
        "has_b_frames": stream["has_b_frames"],
        "keyframe_indices": actual_keys,
        "idr_indices": idr_indices,
        "aud_count": len(units),
        "timestamps_reordered": False,
    }


def _encoder_arguments(fps: int) -> list[str]:
    bitrate = "2M" if fps == 30 else "4M"
    x264_parameters = (
        f"keyint={fps}:min-keyint={fps}:no-scenecut=1:"
        "repeat-headers=1:aud=1:nal-hrd=none"
    )
    return [
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-tune",
        "zerolatency",
        "-profile:v",
        "baseline",
        "-level:v",
        "3.1",
        "-pix_fmt",
        "yuv420p",
        "-b:v",
        bitrate,
        "-r",
        str(fps),
        "-g",
        str(fps),
        "-keyint_min",
        str(fps),
        "-sc_threshold",
        "0",
        "-bf",
        "0",
        "-x264-params",
        x264_parameters,
        "-movflags",
        "+faststart",
    ]


def prepare_video(args: argparse.Namespace) -> tuple[Path, Path]:
    ffmpeg = _required_executable("ffmpeg")
    ffprobe = _required_executable("ffprobe")
    frames_dir = args.frames_dir.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.suffix.lower() != ".mp4":
        raise VideoPreparationError(f"output must use the .mp4 extension: {output}")
    provenance_path = output.with_suffix(".provenance.json")
    if not args.force:
        existing = [path for path in (output, provenance_path) if path.exists()]
        if existing:
            raise VideoPreparationError(
                "refusing to overwrite existing output: "
                + ", ".join(str(path) for path in existing)
            )

    frames, first_index = discover_frames(frames_dir)
    width, height = validate_dimensions(frames)
    frame_digest = ordered_frames_sha256(frames)
    output.parent.mkdir(parents=True, exist_ok=True)

    normalized_suffix = (
        ".jpg"
        if frames[0].suffix.lower() in {".jpg", ".jpeg"}
        else frames[0].suffix.lower()
    )
    encoder_arguments = _encoder_arguments(args.fps)
    with tempfile.TemporaryDirectory(
        prefix=f".{output.stem}.prepare-", dir=output.parent
    ) as temporary_directory:
        temporary_root = Path(temporary_directory)
        staged_frames = temporary_root / "frames"
        staged_frames.mkdir()
        for offset, frame in enumerate(frames):
            staged = staged_frames / f"frame_{offset:08d}{normalized_suffix}"
            staged.symlink_to(frame)

        encoded = temporary_root / output.name
        input_pattern = staged_frames / f"frame_%08d{normalized_suffix}"
        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-framerate",
            str(args.fps),
            "-start_number",
            "0",
            "-i",
            str(input_pattern),
            "-frames:v",
            str(len(frames)),
            "-fps_mode",
            "cfr",
            *encoder_arguments,
            str(encoded),
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            detail = result.stderr.strip()
            raise VideoPreparationError(
                f"FFmpeg encoding failed ({result.returncode}):\n{detail}"
            )

        media = validate_encoded_video(
            ffmpeg,
            ffprobe,
            encoded,
            fps=args.fps,
            frame_count=len(frames),
            width=width,
            height=height,
        )
        ffmpeg_version = subprocess.run(
            [ffmpeg, "-version"], capture_output=True, text=True, check=True
        ).stdout.splitlines()[0]
        output_digest = sha256_file(encoded)
        provenance = {
            "schema_version": 1,
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "source": {
                "id": args.source_id or frames_dir.name,
                "reference": args.source_reference,
                "frames_directory": str(frames_dir),
                "first_frame": frames[0].name,
                "last_frame": frames[-1].name,
                "first_numeric_index": first_index,
                "frame_count": len(frames),
                "ordered_frames_sha256": frame_digest,
                "ordered_frames_sha256_contract": (
                    "sha256(u64be(name_length) || name_utf8 || "
                    "u64be(file_size) || file_bytes) for each frame"
                ),
            },
            "encoding": {
                "fps": args.fps,
                "duration_seconds": len(frames) / args.fps,
                "ffmpeg_version": ffmpeg_version,
                "arguments": encoder_arguments,
            },
            "validation": media,
            "output": {
                "file": output.name,
                "size_bytes": encoded.stat().st_size,
                "sha256": output_digest,
            },
        }
        temporary_provenance = temporary_root / provenance_path.name
        temporary_provenance.write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(encoded, output)
        os.replace(temporary_provenance, provenance_path)

    return output, provenance_path


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        output, provenance = prepare_video(args)
    except (OSError, VideoPreparationError, subprocess.SubprocessError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(f"video: {output}")
    print(f"provenance: {provenance}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
