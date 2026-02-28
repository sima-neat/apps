#!/usr/bin/env python3
"""Serve multiple RTSP streams from a single video file.

Each stream is mounted at /stream0, /stream1, ... /streamN-1.
Useful for testing multi-pipe examples like 4PipesYOLOOptiview.

Usage:
    python rtsp_multi_file_server.py /path/to/video.mp4 --streams 4
    python rtsp_multi_file_server.py /path/to/video.mp4 --streams 2 --width 1280 --height 720 --fps 15
"""

import argparse
import glob
import os
import socket
import sys
import time

# gi (PyGObject) is installed system-wide and may not be visible inside a venv.
for p in glob.glob("/usr/lib/python3*/dist-packages"):
    if p not in sys.path:
        sys.path.insert(0, p)

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstRtspServer", "1.0")
from gi.repository import GLib, Gst, GstRtspServer


def _gst_quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _build_launch(video: str, width: int, height: int, fps: int) -> str:
    src = f"filesrc location={_gst_quote(video)} ! qtdemux ! h264parse"
    if width > 0 and height > 0 and fps > 0:
        gop = max(1, fps)
        return (
            f"( {src} ! avdec_h264 ! videoconvert ! videoscale ! videorate"
            f" ! video/x-raw,format=I420,width={width},height={height},framerate={fps}/1"
            f" ! x264enc tune=zerolatency speed-preset=ultrafast bitrate=2000"
            f" key-int-max={gop} bframes=0 byte-stream=true"
            f" ! h264parse config-interval=-1"
            f" ! rtph264pay name=pay0 pt=96 )"
        )
    return f"( {src} config-interval=-1 ! rtph264pay name=pay0 pt=96 )"


def _on_eos(bus, message, pipeline):
    pipeline.seek_simple(
        Gst.Format.TIME,
        Gst.SeekFlags.FLUSH | Gst.SeekFlags.KEY_UNIT,
        0,
    )


def _on_media_configure(factory, media):
    pipeline = media.get_element()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message::eos", _on_eos, pipeline)


def main() -> int:
    p = argparse.ArgumentParser(description="Serve multiple RTSP streams from a video file")
    p.add_argument("video", help="Path to video file (H.264 MP4)")
    p.add_argument("--streams", type=int, default=4, help="Number of streams to create (default: 4)")
    p.add_argument("--port", default="8554", help="RTSP server port (default: 8554)")
    p.add_argument("--no-loop", action="store_true", help="Disable video looping")
    p.add_argument("--width", type=int, default=0, help="Transcode output width (0 = passthrough)")
    p.add_argument("--height", type=int, default=0, help="Transcode output height (0 = passthrough)")
    p.add_argument("--fps", type=int, default=0, help="Transcode output fps (0 = passthrough)")
    args = p.parse_args()

    video = os.path.abspath(args.video)
    if not os.path.isfile(video):
        print(f"Error: file not found: {video}", file=sys.stderr)
        return 1

    num_streams = max(1, min(args.streams, 16))
    transcode = args.width > 0 and args.height > 0 and args.fps > 0

    Gst.init(None)

    server = GstRtspServer.RTSPServer.new()
    server.set_service(args.port)
    mounts = server.get_mount_points()
    launch = _build_launch(video, args.width, args.height, args.fps)

    for i in range(num_streams):
        factory = GstRtspServer.RTSPMediaFactory.new()
        factory.set_launch(launch)
        factory.set_shared(not transcode)
        if not args.no_loop:
            factory.connect("media-configure", _on_media_configure)
        factory.set_suspend_mode(GstRtspServer.RTSPSuspendMode.NONE)
        mounts.add_factory(f"/stream{i}", factory)

    attach_id = server.attach(None)
    if not attach_id:
        print(f"Error: failed to bind on port {args.port}", file=sys.stderr)
        return 2

    loop_str = "looping" if not args.no_loop else "single-play"
    mode_str = f"transcode {args.width}x{args.height}@{args.fps}" if transcode else "passthrough"

    # Pre-warm stream0.
    loop = GLib.MainLoop()
    ctx = loop.get_context()
    for _ in range(10):
        ctx.iteration(False)
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(5)
        s.connect(("127.0.0.1", int(args.port)))
        s.sendall(
            f"DESCRIBE rtsp://localhost:{args.port}/stream0 RTSP/1.0\r\n"
            f"CSeq: 1\r\nAccept: application/sdp\r\n\r\n".encode()
        )
        s.recv(4096)
        s.close()
        time.sleep(1)
        for _ in range(50):
            ctx.iteration(False)
    except Exception as e:
        print(f"Warning: pre-warm failed: {e}", file=sys.stderr)

    urls = [f"rtsp://localhost:{args.port}/stream{i}" for i in range(num_streams)]
    print(f"RTSP server running ({num_streams} streams, {loop_str}, {mode_str})")
    for url in urls:
        print(f"  {url}")
    print("Press Ctrl+C to stop")

    try:
        GLib.MainLoop().run()
    except KeyboardInterrupt:
        print("\nStopped")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
