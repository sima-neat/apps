# Copied from support/templates/python/sources.py;
# edit the template and run scripts/sync_utils.py.
"""Probe a video source for its frame size and rate before the graph is built."""

from __future__ import annotations

from fractions import Fraction
import subprocess

import cv2
import pyneat

from .frames import first_tensor_from_sample, tensor_dim


def fps_from_rate(value: str) -> int:
    """A rounded frame rate from an ffprobe rate such as "30000/1001"; 0 when it is unknown."""
    if not value or value in {"0/0", "0/1"}:
        return 0
    try:
        fps = float(Fraction(value)) if "/" in value else float(value)
    except (ValueError, ZeroDivisionError):
        return 0
    return int(round(fps)) if fps > 0 else 0


def _int_from_probe(value: str | None) -> int:
    try:
        return int(value or 0)
    except ValueError:
        return 0


def probe_ffprobe(url: str, rtsp_tcp: bool = False, tls_verify: bool = True) -> tuple[int, int, int]:
    """Width, height and frame rate of the first video stream, from ffprobe.

    Each value is 0 when ffprobe is missing, fails, takes longer than 5 seconds, or does not
    report it; the caller falls back to another probe.
    """
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-rw_timeout",
        "5000000",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,avg_frame_rate",
        "-of",
        "default=nw=1",
    ]
    if rtsp_tcp:
        cmd.extend(["-rtsp_transport", "tcp"])
    if not tls_verify:
        cmd.extend(["-tls_verify", "0"])
    cmd.append(url)
    try:
        result = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=5)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return 0, 0, 0
    if result.returncode != 0:
        return 0, 0, 0
    values = {}
    for line in result.stdout.splitlines():
        key, sep, value = line.partition("=")
        if sep:
            values[key] = value
    fps = fps_from_rate(values.get("avg_frame_rate", "")) or fps_from_rate(
        values.get("r_frame_rate", "")
    )
    return _int_from_probe(values.get("width")), _int_from_probe(values.get("height")), fps


def probe_rtsp(url: str) -> tuple[int, int, int]:
    """Width, height and frame rate from OpenCV; the frame rate is 0 when OpenCV has none."""
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        raise RuntimeError(f"failed to open RTSP source for probing: {url}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = int(round(cap.get(cv2.CAP_PROP_FPS) or 0))
    cap.release()
    if width <= 0 or height <= 0:
        raise RuntimeError("failed to probe RTSP frame size")
    return width, height, fps


def probe_decoded_size(source) -> tuple[int, int]:
    """Width and height of the first frame a decoded-source graph produces.

    The source runs in a graph of its own, which is closed before this returns.
    """
    graph = pyneat.Graph("source_probe")
    graph.add(source)
    graph.add(pyneat.nodes.output("frame", pyneat.OutputOptions.every_frame(1)))

    run_options = pyneat.RunOptions()
    run_options.preset = pyneat.RunPreset.Realtime
    run_options.queue_depth = 3
    run_options.overflow_policy = pyneat.OverflowPolicy.KeepLatest
    run_options.output_memory = pyneat.OutputMemory.ZeroCopy
    run = graph.build(run_options)
    try:
        sample = run.pull("frame")
    finally:
        run.close()
    if sample is None:
        raise RuntimeError("failed to probe decoded source frame")
    tensor = first_tensor_from_sample(sample)
    if tensor is None:
        raise RuntimeError("decoded source probe did not produce a tensor")
    return tensor_dim(tensor, "width"), tensor_dim(tensor, "height")
