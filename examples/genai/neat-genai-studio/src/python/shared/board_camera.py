"""Grab still frames from a camera attached to the devkit board itself.

Shared by the CLI (``/camera`` mode) and the Flask UI backend (the
``/board-camera/*`` endpoints): unlike the browser webcam the web UI normally
uses, this reads a ``/dev/video*`` device physically plugged into the Modalix
board. To avoid a heavy dependency it shells out to whichever capture tool the
board already has — ffmpeg, fswebcam, or libcamera/rpicam — and only falls
back to OpenCV if it happens to be importable. Stdlib only.
"""

from __future__ import annotations

import glob
import os
import re
import shutil
import subprocess
import tempfile

DEFAULT_DEVICE = "/dev/video0"


def cam_label(device):
    """Human-friendly device label: a bare index becomes /dev/videoN."""
    dev = str(device)
    return f"/dev/video{dev}" if dev.isdigit() else dev


def default_camera_device():
    """The device used when none is given: $NEAT_CAMERA_DEVICE or /dev/video0."""
    return (os.environ.get("NEAT_CAMERA_DEVICE") or "").strip() or DEFAULT_DEVICE


def normalize_camera_device(device):
    """Validate a user-supplied camera selector and return its /dev node.

    Accepts a bare index (``"2"``) or a ``/dev/video*`` path; anything else
    raises ``ValueError``. This keeps request-supplied strings from reaching
    the capture subprocesses as arbitrary filesystem paths.
    """
    dev = str(device).strip()
    if dev.isdigit():
        return f"/dev/video{dev}"
    if re.fullmatch(r"/dev/video\d+", dev):
        return dev
    raise ValueError(f"invalid camera device {device!r} — use an index or a /dev/video* path")


def list_camera_devices():
    """Enumerate /dev/video* nodes with their driver-reported names.

    Returns ``[{"device": "/dev/video0", "name": "..."}, ...]`` sorted by
    index. The name comes from /sys/class/video4linux (best effort, may be
    empty). Note V4L2 often exposes several nodes per physical camera; only
    some of them can actually capture.
    """
    def _index(node):
        m = re.search(r"(\d+)$", node)
        return int(m.group(1)) if m else 0

    devices = []
    for node in sorted(glob.glob("/dev/video[0-9]*"), key=_index):
        name = ""
        try:
            with open(f"/sys/class/video4linux/video{_index(node)}/name",
                      encoding="utf-8") as fh:
                name = fh.read().strip()
        except OSError:
            pass
        devices.append({"device": node, "name": name})
    return devices


def _read_if_nonempty(path):
    """Return the file's bytes if it exists and is non-empty, else None."""
    try:
        if os.path.getsize(path) > 0:
            with open(path, "rb") as fh:
                return fh.read()
    except OSError:
        pass
    return None


def capture_camera_frame(device=None, timeout=12):
    """Grab a single still frame from the board's camera and return
    ``(jpeg_bytes, tool_label)``.

    The Studio runs on the Modalix board, so this reads a camera physically
    attached to the board (unlike the web UI's default source, the browser's
    camera). Raises ``RuntimeError`` with guidance if nothing on the box can
    grab a frame.

    ``device`` may be an index (``0`` → ``/dev/video0``) or a node path; it
    defaults to ``$NEAT_CAMERA_DEVICE`` or ``/dev/video0``.
    """
    dev = (str(device).strip() if device is not None else "") or default_camera_device()
    dev_node = f"/dev/video{dev}" if dev.isdigit() else dev

    errors = []
    fd, tmp_path = tempfile.mkstemp(prefix="neat-cam-", suffix=".jpg")
    os.close(fd)

    def _run(cmd):
        subprocess.run(cmd, check=True, timeout=timeout,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    try:
        # 1) ffmpeg via V4L2 — near-ubiquitous, honors a timeout, grabs one frame.
        ff = shutil.which("ffmpeg")
        if ff:
            try:
                _run([ff, "-nostdin", "-hide_banner", "-loglevel", "error",
                      "-f", "v4l2", "-i", dev_node,
                      "-frames:v", "1", "-q:v", "2", "-y", tmp_path])
                data = _read_if_nonempty(tmp_path)
                if data:
                    return data, "ffmpeg"
            except Exception as exc:  # noqa: BLE001
                errors.append(f"ffmpeg: {exc}")

        # 2) fswebcam — the classic USB-webcam still grabber.
        fs = shutil.which("fswebcam")
        if fs:
            try:
                _run([fs, "-q", "--no-banner", "-d", dev_node,
                      "-r", "1280x720", "-S", "3", "--jpeg", "90", tmp_path])
                data = _read_if_nonempty(tmp_path)
                if data:
                    return data, "fswebcam"
            except Exception as exc:  # noqa: BLE001
                errors.append(f"fswebcam: {exc}")

        # 3) libcamera / rpicam still — CSI cameras on ARM boards.
        for tool in ("libcamera-still", "rpicam-still"):
            binp = shutil.which(tool)
            if not binp:
                continue
            try:
                _run([binp, "-n", "-t", "800",
                      "--width", "1280", "--height", "720", "-o", tmp_path])
                data = _read_if_nonempty(tmp_path)
                if data:
                    return data, tool
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{tool}: {exc}")

        # 4) OpenCV — only if it is already installed (not a declared dependency).
        try:
            import cv2  # type: ignore  # noqa: I001
            # cv2 takes an integer index, so map "/dev/video2" → 2 (matching the
            # dev_node the other tools use); refuse a node we can't map rather
            # than silently opening index 0 (the wrong camera).
            if dev.isdigit():
                idx = int(dev)
            else:
                mnum = re.search(r"(\d+)$", dev)
                if mnum is None:
                    raise RuntimeError(f"cannot map {dev_node} to a camera index")
                idx = int(mnum.group(1))
            cap = cv2.VideoCapture(idx)
            try:
                ok, frame = False, None
                for _ in range(5):   # let auto-exposure settle for a frame or two
                    ok, frame = cap.read()
                    if ok and frame is not None:
                        break
                if ok and frame is not None:
                    ok2, buf = cv2.imencode(".jpg", frame)
                    if ok2:
                        return bytes(buf), "opencv"
                errors.append("opencv: no frame")
            finally:
                cap.release()
        except ImportError:
            pass
        except Exception as exc:  # noqa: BLE001
            errors.append(f"opencv: {exc}")

        detail = "; ".join(errors) if errors else "no capture tool found on PATH"
        raise RuntimeError(
            f"could not grab a frame from {dev_node} ({detail}). "
            "Install ffmpeg, fswebcam, or libcamera-apps on the board, or set "
            "NEAT_CAMERA_DEVICE to the right /dev/video* node.")
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
