#!/usr/bin/env python3
# Copyright 2026 SiMa Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Create real moving, no-B-frame App16 videos with unique channel markers."""

from __future__ import annotations

import argparse
import colorsys
import json
import math
import shlex
import subprocess
import sys
from pathlib import Path


def channel_ids(value: str) -> list[int]:
    try:
        result = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("channel IDs must be comma-separated integers") from exc
    if not result or any(item < 0 for item in result) or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("channel IDs must be nonnegative and unique")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="real moving source video")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--channel-ids", type=channel_ids, default=list(range(24)))
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--duration-seconds", type=float, default=120.0)
    parser.add_argument(
        "--font-file",
        type=Path,
        default=Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    )
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.width <= 0 or args.height <= 0 or args.fps <= 0 or args.duration_seconds <= 0:
        parser.error("dimensions, FPS, and duration must be positive")
    if args.width < 160 or args.height < 120:
        parser.error("identity and temporal markers require at least 160x120 video")
    return args


def channel_colors(channels: list[int]) -> dict[int, list[int]]:
    count = len(channels)
    colors: dict[int, list[int]] = {}
    for ordinal, channel in enumerate(channels):
        red, green, blue = colorsys.hsv_to_rgb(ordinal / count, 0.85, 0.90)
        colors[channel] = [round(red * 255), round(green * 255), round(blue * 255)]
    return colors


def ffmpeg_command(args: argparse.Namespace, channel: int, rgb: list[int], output: Path) -> list[str]:
    color = "".join(f"{component:02x}" for component in rgb)
    keyint = args.fps
    period_frames = max(2, round(args.duration_seconds * args.fps))
    temporal_bits = max(2, math.ceil(math.log2(period_frames)))
    filters = [
        f"fps={args.fps},scale={args.width}:{args.height}:flags=lanczos,format=yuv420p,"
        f"drawbox=x=0:y=0:w=128:h=64:color=0x{color}@1.0:t=fill,"
        f"drawtext=fontfile={args.font_file}:text=CH{channel:02d}:"
        "x=42:y=16:fontsize=28:fontcolor=white:borderw=2:bordercolor=black,"
        f"drawbox=x=0:y=64:w={16 + temporal_bits * 12}:h=32:color=black@1.0:t=fill"
    ]
    for bit in range(temporal_bits):
        modulus = 1 << (bit + 1)
        threshold = 1 << bit
        # Escaped commas belong to the timeline expression rather than the
        # filter chain. Each white cell encodes one LSB-first frame-counter bit.
        enable = (
            f"gte(mod(floor(t*{args.fps})\\,{modulus})\\,{threshold})"
        )
        filters.append(
            f"drawbox=x={8 + bit * 12}:y=70:w=8:h=20:color=white@1.0:t=fill:"
            f"enable='{enable}'"
        )
    video_filter = ",".join(filters)
    return [
        args.ffmpeg,
        "-y" if args.overwrite else "-n",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-stream_loop",
        "-1",
        "-i",
        str(args.input),
        "-t",
        str(args.duration_seconds),
        "-an",
        "-vf",
        video_filter,
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-tune",
        "zerolatency",
        "-pix_fmt",
        "yuv420p",
        "-g",
        str(keyint),
        "-keyint_min",
        str(keyint),
        "-bf",
        "0",
        "-refs",
        "1",
        "-sc_threshold",
        "0",
        "-x264-params",
        "bframes=0:repeat-headers=1",
        "-fps_mode",
        "cfr",
        str(output),
    ]


def verify_output(args: argparse.Namespace, output: Path) -> None:
    command = [
        args.ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height,avg_frame_rate,has_b_frames",
        "-of",
        "json",
        str(output),
    ]
    payload = json.loads(subprocess.check_output(command, text=True))
    streams = payload.get("streams", [])
    if len(streams) != 1:
        raise RuntimeError(f"{output}: ffprobe did not report exactly one video stream")
    stream = streams[0]
    expected_rate = f"{args.fps}/1"
    if (
        stream.get("codec_name") != "h264"
        or stream.get("width") != args.width
        or stream.get("height") != args.height
        or stream.get("avg_frame_rate") != expected_rate
        or stream.get("has_b_frames") != 0
    ):
        raise RuntimeError(f"{output}: unexpected encoded stream: {stream}")


def verify_temporal_prefix(args: argparse.Namespace, output: Path, bits: int) -> None:
    frames = min(16, max(2, round(args.duration_seconds * args.fps)))
    crop_width = (bits - 1) * 12 + 8
    command = [
        args.ffmpeg,
        "-v",
        "error",
        "-i",
        str(output),
        "-vf",
        f"crop={crop_width}:20:8:70",
        "-frames:v",
        str(frames),
        "-pix_fmt",
        "rgb24",
        "-f",
        "rawvideo",
        "-",
    ]
    raw = subprocess.check_output(command)
    frame_bytes = crop_width * 20 * 3
    if len(raw) != frames * frame_bytes:
        raise RuntimeError(f"{output}: temporal marker decode returned {len(raw)} bytes")
    decoded: list[int] = []
    for frame_index in range(frames):
        frame = raw[frame_index * frame_bytes : (frame_index + 1) * frame_bytes]
        code = 0
        for bit in range(bits):
            x = bit * 12 + 4
            y = 10
            offset = (y * crop_width + x) * 3
            if sum(frame[offset : offset + 3]) / 3 >= 128:
                code |= 1 << bit
        decoded.append(code)
    if decoded != list(range(frames)):
        raise RuntimeError(f"{output}: temporal marker prefix mismatch: {decoded}")


def measure_decoded_marker_rgb(
    args: argparse.Namespace,
    output: Path,
    marker: dict[str, int],
    sample_frames: int = 8,
) -> list[int]:
    """Measure the marker after H.264/YUV conversion, not its requested RGB."""
    command = [
        args.ffmpeg,
        "-v",
        "error",
        "-i",
        str(output),
        "-vf",
        f"crop={marker['width']}:{marker['height']}:{marker['x']}:{marker['y']}",
        "-frames:v",
        str(sample_frames),
        "-pix_fmt",
        "rgb24",
        "-f",
        "rawvideo",
        "-",
    ]
    raw = subprocess.check_output(command)
    frame_bytes = marker["width"] * marker["height"] * 3
    if not raw or len(raw) % frame_bytes != 0:
        raise RuntimeError(f"{output}: decoded marker returned {len(raw)} bytes")
    pixels = len(raw) // 3
    return [round(sum(raw[channel::3]) / pixels) for channel in range(3)]


def require_unique_marker_colors(channels: dict[str, dict[str, object]]) -> None:
    colors = [tuple(row["rgb"]) for row in channels.values()]
    if len(set(colors)) != len(colors):
        raise RuntimeError("decoded channel marker RGB values are not unique")


def main() -> int:
    args = parse_args()
    if not args.dry_run:
        if not args.input.is_file():
            raise RuntimeError(f"input video does not exist: {args.input}")
        if not args.font_file.is_file():
            raise RuntimeError(f"drawtext font does not exist: {args.font_file}")
        args.output_dir.mkdir(parents=True, exist_ok=True)

    colors = channel_colors(args.channel_ids)
    temporal_bits = max(
        2, math.ceil(math.log2(max(2, round(args.duration_seconds * args.fps))))
    )
    outputs: dict[int, Path] = {
        channel: args.output_dir / f"channel-{channel:02d}-{args.width}x{args.height}p{args.fps}-no-b.mp4"
        for channel in args.channel_ids
    }
    marker = {"x": 8, "y": 8, "width": 24, "height": 24}
    manifest = {
        "version": 1,
        "description": "App16 real-moving fixtures with stable per-channel color and text markers",
        "width": args.width,
        "height": args.height,
        "fps": args.fps,
        # This patch is inside the solid box but outside the human-readable CHxx text.
        "marker": marker,
        "marker_rgb_basis": (
            "intended-pre-encode-dry-run"
            if args.dry_run
            else "mean-decoded-rgb24-first-8-frames"
        ),
        "temporal": {
            "x": 8,
            "y": 70,
            "bit_width": 8,
            "bit_height": 20,
            "bit_stride": 12,
            "bits": temporal_bits,
            "period_frames": max(2, round(args.duration_seconds * args.fps)),
            "fps": args.fps,
            "luma_threshold": 128,
            "sync_tolerance_frames": 6,
        },
        "tolerance": 45.0,
        "channels": {
            str(channel): {
                "rgb": colors[channel],
                "intended_rgb": colors[channel],
                "file": outputs[channel].name,
            }
            for channel in args.channel_ids
        },
    }

    for channel in args.channel_ids:
        command = ffmpeg_command(args, channel, colors[channel], outputs[channel])
        print(shlex.join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True)
            verify_output(args, outputs[channel])
            verify_temporal_prefix(args, outputs[channel], temporal_bits)
            manifest["channels"][str(channel)]["rgb"] = measure_decoded_marker_rgb(
                args, outputs[channel], marker
            )

    require_unique_marker_colors(manifest["channels"])

    manifest_path = args.output_dir / "identity-manifest.json"
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        print(f"identity fixture generation failed: {exc}", file=sys.stderr)
        sys.exit(2)
