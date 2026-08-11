#!/usr/bin/env python3
"""Dependency-free video-generation contract test for a pristine SDK image."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


EXAMPLE_ROOT = Path(__file__).resolve().parents[2]
GENERATOR = EXAMPLE_ROOT / "tools" / "prepare_demo_video.py"

GENERATOR_SPEC = importlib.util.spec_from_file_location("prepare_demo_video", GENERATOR)
assert GENERATOR_SPEC and GENERATOR_SPEC.loader
prepare_demo_video = importlib.util.module_from_spec(GENERATOR_SPEC)
GENERATOR_SPEC.loader.exec_module(prepare_demo_video)


def write_pgm_frame(
    path: Path, frame_index: int, *, width: int = 96, height: int = 64
) -> None:
    pixels = bytearray(width * height)
    for y in range(height):
        for x in range(width):
            pixels[y * width + x] = (x * 3 + y * 5 + frame_index * 7) % 256
    path.write_bytes(f"P5\n{width} {height}\n255\n".encode() + pixels)


def write_pgm_sequence(
    directory: Path,
    frame_count: int = 121,
    *,
    width: int = 96,
    height: int = 64,
) -> None:
    directory.mkdir()
    for frame_index in range(frame_count):
        write_pgm_frame(
            directory / f"frame_{frame_index:06d}.pgm",
            frame_index,
            width=width,
            height=height,
        )


def ffprobe_json(video: Path, *, frames: bool = False, packets: bool = False) -> dict:
    command = ["ffprobe", "-v", "error", "-select_streams", "v:0"]
    if frames:
        command += [
            "-show_frames",
            "-show_entries",
            "frame=key_frame,pict_type",
        ]
    elif packets:
        command += [
            "-show_packets",
            "-show_entries",
            "packet=pts,dts,duration",
        ]
    else:
        command += [
            "-show_entries",
            (
                "stream=codec_name,profile,level,pix_fmt,r_frame_rate,avg_frame_rate,"
                "nb_frames,has_b_frames,width,height"
            ),
        ]
    command += ["-of", "json", str(video)]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


class TestPrepareDemoVideoSdk(unittest.TestCase):
    def test_selects_a_conformant_h264_level(self) -> None:
        cases = {
            (640, 512, 30): ("3.1", 31),
            (1280, 720, 30): ("3.1", 31),
            (1280, 720, 60): ("3.2", 32),
            (1920, 1080, 30): ("4.0", 40),
            (1920, 1080, 60): ("4.2", 42),
            (1600, 1600, 60): ("5.1", 51),
        }
        for dimensions, expected in cases.items():
            with self.subTest(dimensions=dimensions):
                self.assertEqual(
                    prepare_demo_video.select_h264_level(*dimensions), expected
                )

    def test_encodes_720p60_at_level_3_2(self) -> None:
        with tempfile.TemporaryDirectory(prefix="prepare-demo-video-720p60-") as root:
            root_path = Path(root)
            frames_dir = root_path / "frames"
            write_pgm_sequence(frames_dir, frame_count=2, width=1280, height=720)
            video = root_path / "demo-720p60.mp4"

            subprocess.run(
                [
                    sys.executable,
                    str(GENERATOR),
                    "--frames-dir",
                    str(frames_dir),
                    "--fps",
                    "60",
                    "--output",
                    str(video),
                ],
                check=True,
            )

            stream = ffprobe_json(video)["streams"][0]
            self.assertEqual(stream["level"], 32)
            self.assertEqual(stream["r_frame_rate"], "60/1")
            self.assertEqual(stream["has_b_frames"], 0)

    def test_rejects_a_frame_index_gap_before_encoding(self) -> None:
        with tempfile.TemporaryDirectory(prefix="prepare-demo-video-gap-") as root:
            root_path = Path(root)
            frames_dir = root_path / "frames"
            frames_dir.mkdir()
            write_pgm_frame(frames_dir / "frame_000000.pgm", 0)
            write_pgm_frame(frames_dir / "frame_000002.pgm", 2)
            output = root_path / "should-not-exist.mp4"

            result = subprocess.run(
                [
                    sys.executable,
                    str(GENERATOR),
                    "--frames-dir",
                    str(frames_dir),
                    "--output",
                    str(output),
                ],
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 2)
            self.assertIn("frame index sequence has gaps; missing: 1", result.stderr)
            self.assertFalse(output.exists())

    def test_generates_insight_compatible_30_and_60_fps(self) -> None:
        with tempfile.TemporaryDirectory(prefix="prepare-demo-video-sdk-") as root:
            root_path = Path(root)
            frames_dir = root_path / "frames"
            write_pgm_sequence(frames_dir)

            for fps in (30, 60):
                with self.subTest(fps=fps):
                    video = root_path / f"demo-{fps}.mp4"
                    subprocess.run(
                        [
                            sys.executable,
                            str(GENERATOR),
                            "--frames-dir",
                            str(frames_dir),
                            "--fps",
                            str(fps),
                            "--output",
                            str(video),
                        ],
                        check=True,
                    )

                    stream = ffprobe_json(video)["streams"][0]
                    self.assertEqual(stream["codec_name"], "h264")
                    self.assertIn(
                        stream["profile"], {"Baseline", "Constrained Baseline"}
                    )
                    self.assertEqual(stream["level"], 31)
                    self.assertEqual(stream["pix_fmt"], "yuv420p")
                    self.assertEqual(stream["r_frame_rate"], f"{fps}/1")
                    self.assertEqual(stream["avg_frame_rate"], f"{fps}/1")
                    self.assertEqual(int(stream["nb_frames"]), 121)
                    self.assertEqual(stream["has_b_frames"], 0)
                    self.assertEqual((stream["width"], stream["height"]), (96, 64))

                    frames = ffprobe_json(video, frames=True)["frames"]
                    self.assertEqual(len(frames), 121)
                    self.assertLessEqual(
                        {frame["pict_type"] for frame in frames}, {"I", "P"}
                    )
                    keyframes = [
                        index
                        for index, frame in enumerate(frames)
                        if int(frame["key_frame"]) == 1
                    ]
                    self.assertEqual(keyframes, list(range(0, 121, fps)))

                    packets = ffprobe_json(video, packets=True)["packets"]
                    self.assertEqual(len(packets), 121)
                    pts = [int(packet["pts"]) for packet in packets]
                    dts = [int(packet["dts"]) for packet in packets]
                    self.assertEqual(pts, dts)
                    self.assertEqual(len({b - a for a, b in zip(pts, pts[1:])}), 1)

                    provenance = json.loads(
                        video.with_suffix(".provenance.json").read_text()
                    )
                    self.assertEqual(provenance["source"]["frame_count"], 121)
                    self.assertEqual(provenance["encoding"]["fps"], fps)
                    self.assertEqual(provenance["validation"]["aud_count"], 121)
                    self.assertEqual(provenance["validation"]["idr_indices"], keyframes)
                    self.assertEqual(provenance["validation"]["has_b_frames"], 0)
                    self.assertFalse(provenance["validation"]["timestamps_reordered"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
