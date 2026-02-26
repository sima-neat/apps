#!/usr/bin/env python3
"""Serve a local video file as an RTSP stream using GStreamer."""

import argparse
import glob
import os
import sys

# gi (PyGObject) is installed system-wide and may not be visible inside a venv.
# Add the system dist-packages so the import works regardless of venv state.
for p in glob.glob("/usr/lib/python3*/dist-packages"):
    if p not in sys.path:
        sys.path.insert(0, p)

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstRtspServer", "1.0")
from gi.repository import GLib, Gst, GstRtspServer


def _gst_quote(value: str) -> str:
    """Quote a string for use as a GStreamer launch property value."""
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _build_launch(video: str, transcode: bool, width: int, height: int, fps: int, bitrate_kbps: int) -> str:
    if not transcode:
        return (
            f"( filesrc location={_gst_quote(video)} ! qtdemux ! h264parse config-interval=-1"
            f" ! rtph264pay name=pay0 pt=96 )"
        )

    # Decoder-friendly H.264 output for local testing (e.g., downscale 4K samples).
    gop = max(1, fps)
    return (
        f"( filesrc location={_gst_quote(video)} ! qtdemux ! h264parse ! avdec_h264"
        f" ! videoconvert ! videoscale ! videorate"
        f" ! video/x-raw,format=I420,width={width},height={height},framerate={fps}/1"
        f" ! x264enc tune=zerolatency speed-preset=ultrafast bitrate={bitrate_kbps}"
        f" key-int-max={gop} bframes=0 byte-stream=true"
        f" ! video/x-h264,profile=constrained-baseline,level=(string)3.1"
        f" ! h264parse config-interval=-1"
        f" ! rtph264pay name=pay0 pt=96 )"
    )


def _on_eos(bus, message, pipeline):
    """Seek back to start on end-of-stream to loop the video."""
    pipeline.seek_simple(
        Gst.Format.TIME,
        Gst.SeekFlags.FLUSH | Gst.SeekFlags.KEY_UNIT,
        0,
    )


def _on_media_configure(factory, media):
    """Attach an EOS handler to each new media instance for looping."""
    pipeline = media.get_element()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message::eos", _on_eos, pipeline)


def main() -> int:
    p = argparse.ArgumentParser(description="Serve a video file over RTSP")
    p.add_argument("video", help="Path to video file (H.264 MP4)")
    p.add_argument("--port", default="8554", help="RTSP server port (default: 8554)")
    p.add_argument("--path", default="/stream", help="RTSP mount path (default: /stream)")
    p.add_argument("--no-loop", action="store_true", help="Disable video looping")
    p.add_argument(
        "--transcode",
        action="store_true",
        help="Re-encode to a decoder-friendly H.264 stream (useful for high-res/high-profile sources)",
    )
    p.add_argument("--width", type=int, default=1280, help="Transcode output width (default: 1280)")
    p.add_argument("--height", type=int, default=720, help="Transcode output height (default: 720)")
    p.add_argument("--fps", type=int, default=15, help="Transcode output fps (default: 15)")
    p.add_argument(
        "--bitrate-kbps",
        type=int,
        default=2000,
        help="Transcode video bitrate in kbps (default: 2000)",
    )
    p.add_argument(
        "--shared",
        action="store_true",
        help="Use a shared RTSP media pipeline (default: disabled for --transcode, enabled otherwise)",
    )
    args = p.parse_args()

    video = os.path.abspath(args.video)
    if not os.path.isfile(video):
        print(f"Error: file not found: {video}", file=sys.stderr)
        return 1

    Gst.init(None)

    server = GstRtspServer.RTSPServer.new()
    server.set_service(args.port)

    factory = GstRtspServer.RTSPMediaFactory.new()
    factory.set_launch(
        _build_launch(
            video=video,
            transcode=args.transcode,
            width=max(1, args.width),
            height=max(1, args.height),
            fps=max(1, args.fps),
            bitrate_kbps=max(100, args.bitrate_kbps),
        )
    )
    shared = args.shared or not args.transcode
    factory.set_shared(shared)

    if not args.no_loop:
        factory.connect("media-configure", _on_media_configure)

    factory.set_suspend_mode(GstRtspServer.RTSPSuspendMode.NONE)

    mounts = server.get_mount_points()
    mounts.add_factory(args.path, factory)
    attach_id = server.attach(None)
    if not attach_id:
        print(
            f"Error: failed to bind RTSP server on port {args.port} (address in use?)",
            file=sys.stderr,
        )
        return 2

    url = f"rtsp://localhost:{args.port}{args.path}"
    loop_str = "looping" if not args.no_loop else "single-play"
    mode_str = (
        f"transcode {max(1, args.width)}x{max(1, args.height)}@{max(1, args.fps)} "
        f"{max(100, args.bitrate_kbps)}kbps"
        if args.transcode
        else "passthrough"
    )
    share_str = "shared" if shared else "non-shared"

    # Pre-warm: trigger media creation so the first real client doesn't get 503.
    import socket
    import time
    loop = GLib.MainLoop()
    ctx = loop.get_context()
    # Let the server start processing
    for _ in range(10):
        ctx.iteration(False)
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(5)
        s.connect(("127.0.0.1", int(args.port)))
        s.sendall(
            f"DESCRIBE rtsp://localhost:{args.port}{args.path} RTSP/1.0\r\n"
            f"CSeq: 1\r\nAccept: application/sdp\r\n\r\n".encode()
        )
        resp = s.recv(4096).decode(errors="replace")
        s.close()
        status_line = resp.splitlines()[0] if resp else "<no response>"
        if not status_line.startswith("RTSP/1.0 200"):
            print(
                f"Warning: RTSP self-check returned {status_line} for {url}",
                file=sys.stderr,
            )
        # Let the media pipeline preroll
        time.sleep(1)
        for _ in range(50):
            ctx.iteration(False)
    except Exception as e:
        print(f"Warning: RTSP self-check failed for {url}: {e}", file=sys.stderr)

    print(f"RTSP server running at {url} ({loop_str}, {mode_str}, {share_str})")
    print("Press Ctrl+C to stop")

    try:
        GLib.MainLoop().run()
    except KeyboardInterrupt:
        print("\nStopped")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
