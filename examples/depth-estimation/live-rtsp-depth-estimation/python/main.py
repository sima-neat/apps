"""MiDaS v2.1 RTSP depth overlay example using pyneat."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
import re
import sys
import time

import cv2
import numpy as np
import pyneat


@dataclass(frozen=True)
class DepthModelProfile:
    name: str
    input_format: str  # "BGR" or "RGB"
    default_size: int
    depth_order: str  # "row_major" or "column_major"


MIDAS_V21_PROFILE = DepthModelProfile(
    name="midas_v21_small_256", input_format="BGR", default_size=256, depth_order="row_major"
)
DEPTH_ANYTHING_V2_VITS_PROFILE = DepthModelProfile(
    name="depth_anything_v2_vits", input_format="RGB", default_size=518, depth_order="column_major"
)


def detect_model_profile(model_path: str) -> DepthModelProfile:
    name = os.path.basename(model_path).lower()
    if "offline-depth-map-generation" in name:
        return DEPTH_ANYTHING_V2_VITS_PROFILE
    return MIDAS_V21_PROFILE


def tensor_to_numpy(t: pyneat.Tensor) -> np.ndarray:
    return np.asarray(t.to_numpy(copy=True))


def tensor_bgr_from_decoded(t: pyneat.Tensor) -> np.ndarray:
    arr = tensor_to_numpy(t)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"unexpected decoded tensor shape {arr.shape}")
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def iter_tensors(sample: pyneat.Sample):
    if sample.kind == pyneat.SampleKind.Tensor and sample.tensor is not None:
        yield sample.tensor
    for field in sample.fields:
        yield from iter_tensors(field)


def first_tensor(sample: pyneat.Sample) -> pyneat.Tensor | None:
    for t in iter_tensors(sample):
        return t
    return None


def depth_tensor_from_sample(sample: pyneat.Sample) -> pyneat.Tensor | None:
    tensors = list(iter_tensors(sample))
    if not tensors:
        return None
    for t in tensors:
        sem = getattr(t, "semantic", None)
        image_sem = getattr(sem, "image", None) if sem is not None else None
        if image_sem is None and t.dtype != pyneat.TensorDType.UInt8:
            return t
    return tensors[0]


def depth_colormap_from_tensor(t: pyneat.Tensor, *, depth_order: str) -> np.ndarray:
    arr = tensor_to_numpy(t).reshape(-1)
    spatial = [int(d) for d in t.shape if int(d) > 1]
    if len(spatial) >= 2:
        h, w = spatial[0], spatial[1]
    elif len(spatial) == 1:
        h = w = spatial[0]
    else:
        raise ValueError(f"cannot infer depth dims from {tuple(t.shape)}")
    total = h * w
    if arr.size < total:
        raise ValueError("depth tensor payload too small")

    if depth_order == "column_major":
        # Depth Anything V2 exports values grouped by columns (idx = x*h + y).
        depth = np.asarray(arr[:total], dtype=np.float32).reshape(w, h).T
    else:
        # MiDaS path: row-major (idx = y*w + x).
        depth = np.asarray(arr[:total], dtype=np.float32).reshape(h, w)

    if np.isfinite(depth).all() and float(depth.max()) > float(depth.min()):
        depth_u8 = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        depth_u8 = np.zeros((h, w), dtype=np.uint8)
    return cv2.applyColorMap(depth_u8, cv2.COLORMAP_INFERNO)


def probe_rtsp(url: str) -> tuple[int, int, int]:
    """Probe the live stream once so downstream caps match the real source."""
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        raise RuntimeError(f"failed to open RTSP source for probing: {url}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = int(round(cap.get(cv2.CAP_PROP_FPS) or 0))
    cap.release()
    if width <= 0 or height <= 0:
        raise RuntimeError("failed to probe RTSP frame size")
    return width, height, max(0, fps)


def build_rtsp_run(
    url: str,
    width: int,
    height: int,
    stream_fps: int,
    latency_ms: int,
    tcp: bool,
    sample_every: int,
    fallback_width: int = 0,
    fallback_height: int = 0,
):
    ro = pyneat.RtspDecodedInputOptions()
    ro.url = url
    ro.latency_ms = latency_ms
    ro.tcp = tcp
    ro.payload_type = 96
    ro.insert_queue = True
    ro.out_format = "BGR"
    ro.decoder_raw_output = False
    ro.decoder_name = "decoder"
    ro.auto_caps_from_stream = True
    ro.use_videoconvert = False
    ro.use_videoscale = True
    if fallback_width > 0:
        ro.fallback_h264_width = fallback_width
    if fallback_height > 0:
        ro.fallback_h264_height = fallback_height
    if stream_fps > 0:
        ro.fallback_h264_fps = stream_fps
    ro.output_caps.enable = True
    ro.output_caps.format = "BGR"
    ro.output_caps.width = width
    ro.output_caps.height = height
    if stream_fps > 0:
        ro.output_caps.fps = stream_fps
    ro.output_caps.memory = pyneat.CapsMemory.SystemMemory

    sess = pyneat.Session()
    sess.add(pyneat.groups.rtsp_decoded_input(ro))
    sess.add(pyneat.nodes.output(pyneat.OutputOptions.every_frame(max(1, sample_every))))
    run = sess.build()
    return sess, run


def build_depth_model(model_path: str, width: int, height: int, profile: DepthModelProfile):
    opt = pyneat.ModelOptions()
    opt.media_type = "video/x-raw"
    opt.format = profile.input_format
    opt.input_max_width = width
    opt.input_max_height = height
    opt.input_max_depth = 3
    model = pyneat.Model(model_path, opt)
    return model


def make_model(model_path: str, width: int, height: int, profile: DepthModelProfile):
    """Named model stage for lifecycle readability."""
    return build_depth_model(model_path, width, height, profile)


def render_depth_overlay(
    model: pyneat.Model,
    bgr: np.ndarray,
    width: int,
    height: int,
    alpha: float,
    model_profile: DepthModelProfile,
    timings: dict[str, float] | None = None,
) -> np.ndarray:
    if bgr.shape[1] != width or bgr.shape[0] != height:
        bgr = cv2.resize(bgr, (width, height))
    model_input = bgr
    if model_profile.input_format == "RGB":
        model_input = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    t0 = time.perf_counter()
    sample = model.run(model_input, timeout_ms=5000)
    t1 = time.perf_counter()
    depth_t = depth_tensor_from_sample(sample)
    if depth_t is None:
        raise RuntimeError("Model output missing depth tensor")

    depth_cm = depth_colormap_from_tensor(depth_t, depth_order=model_profile.depth_order)
    if depth_cm.shape[1] != width or depth_cm.shape[0] != height:
        depth_cm = cv2.resize(depth_cm, (width, height))
    out = cv2.addWeighted(bgr, 1.0 - alpha, depth_cm, alpha, 0.0)
    if timings is not None:
        timings["model_s"] = t1 - t0
        timings["post_s"] = time.perf_counter() - t1
    return out


def parse_queue2_depth(report: str) -> int | None:
    m = re.search(r"queue2 depth=(\d+)", report)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


class StageProfiler:
    """Lightweight cumulative stage profiler for optional debug output."""

    ORDER = [
        ("pull_ok", "pull"),
        ("tensor_to_bgr", "tensor"),
        ("model_run", "model"),
        ("postprocess", "post"),
        ("writer_write", "write"),
        ("frame_total", "frame"),
    ]

    def __init__(self, enabled: bool):
        self.enabled = enabled
        self._skip_next_frame = enabled  # hide first-frame warm-up from stage averages
        self._sum_s: dict[str, float] = {}
        self._cnt: dict[str, int] = {}
        self._max_s: dict[str, float] = {}

    def skip_next_frame(self) -> None:
        if self.enabled:
            self._skip_next_frame = True

    def begin_frame(self) -> bool:
        if not self.enabled:
            return False
        if self._skip_next_frame:
            self._skip_next_frame = False
            return False
        return True

    def add(self, name: str, dt_s: float, *, include: bool) -> None:
        if not self.enabled or not include:
            return
        self._sum_s[name] = self._sum_s.get(name, 0.0) + dt_s
        self._cnt[name] = self._cnt.get(name, 0) + 1
        self._max_s[name] = max(self._max_s.get(name, 0.0), dt_s)

    def _mean_ms(self, name: str) -> float:
        c = self._cnt.get(name, 0)
        if c <= 0:
            return 0.0
        return 1000.0 * self._sum_s.get(name, 0.0) / float(c)

    def _max_ms(self, name: str) -> float:
        return 1000.0 * self._max_s.get(name, 0.0)

    def print(self, label: str) -> None:
        if not self.enabled:
            return
        print("--------------------------")
        print(f"[PROFILE] {label}")
        print("[PROFILE]   stage         avg(ms)   max(ms)   n")
        for key, pretty in self.ORDER:
            c = self._cnt.get(key, 0)
            if c <= 0:
                continue
            print(
                f"[PROFILE]   {pretty:<12} "
                f"{self._mean_ms(key):8.1f} {self._max_ms(key):8.1f} {c:4d}"
                )


def print_queue_profile(run: pyneat.Run, queue2_depth: int | None) -> None:
    rs = run.stats()
    ins = run.input_stats()
    pending = max(0, int(rs.outputs_ready) - int(rs.outputs_pulled))
    print("[PROFILE]   queue                            value")
    print(f"[PROFILE]   run_output_pending              {pending}")
    print(f"[PROFILE]   run_outputs_ready_total         {int(rs.outputs_ready)}")
    print(f"[PROFILE]   run_outputs_pulled_total        {int(rs.outputs_pulled)}")
    print(f"[PROFILE]   run_outputs_dropped_total       {int(rs.outputs_dropped)}")
    print(f"[PROFILE]   input_dropped_frames_total      {int(ins.dropped_frames)}")
    if queue2_depth is not None:
        print(f"[PROFILE]   gst_queue2_depth_config         {queue2_depth}")


def handle_reconnect(
    rtsp_url: str,
    width: int,
    height: int,
    fps: int,
    latency_ms: int,
    tcp: bool,
    sample_every: int,
    profile_enabled: bool,
    profiler: StageProfiler,
) -> tuple[pyneat.Session, pyneat.Run, int | None]:
    """Reconnect stage: rebuild source session/run with unchanged options."""
    rtsp_session, rtsp_run = build_rtsp_run(
        rtsp_url, width, height, fps, latency_ms, tcp, sample_every
    )
    queue2_depth = parse_queue2_depth(rtsp_run.report()) if profile_enabled else None
    profiler.skip_next_frame()
    return rtsp_session, rtsp_run, queue2_depth


def shutdown(writer, rtsp_run) -> None:
    """Teardown stage: release writer then close RTSP run."""
    try:
        if writer is not None:
            writer.release()
    except Exception:
        pass
    try:
        if rtsp_run is not None:
            rtsp_run.close()
    except Exception:
        pass


def run_frame_loop(
    *,
    model,
    rtsp_url: str,
    width: int,
    height: int,
    fps: int,
    latency_ms: int,
    tcp: bool,
    sample_every: int,
    alpha: float,
    model_profile: DepthModelProfile,
    profile_enabled: bool,
    log_every: int,
    frame_limit: int,
    output_file: str,
):
    """Processing stage: pull -> infer -> overlay -> write with reconnect handling."""
    writer = None
    rtsp_session, rtsp_run = build_rtsp_run(rtsp_url, width, height, fps, latency_ms, tcp, sample_every)
    queue2_depth = parse_queue2_depth(rtsp_run.report()) if profile_enabled else None
    profiler = StageProfiler(profile_enabled)
    reconnect_attempts = 0
    max_reconnect_attempts = 8
    processed = 0
    log_window_start = None
    log_window_start_frames = 0

    while processed < frame_limit:
        t_frame0 = time.perf_counter()
        t_pull0 = time.perf_counter()
        t = rtsp_run.pull_tensor(timeout_ms=5000)
        pull_dt = time.perf_counter() - t_pull0
        if t is None:
            if processed > 0 and reconnect_attempts < max_reconnect_attempts:
                reconnect_attempts += 1
                print(
                    f"RTSP pull timed out; reconnecting ({reconnect_attempts}/{max_reconnect_attempts})",
                    file=sys.stderr,
                )
                try:
                    rtsp_run.close()
                except Exception:
                    pass
                rtsp_run = None
                time.sleep(0.5)
                rtsp_session, rtsp_run, queue2_depth = handle_reconnect(
                    rtsp_url,
                    width,
                    height,
                    fps,
                    latency_ms,
                    tcp,
                    sample_every,
                    profile_enabled,
                    profiler,
                )
                continue
            print("RTSP pull timed out / stream closed", file=sys.stderr)
            break

        profile_frame = profiler.begin_frame()
        profiler.add("pull_ok", pull_dt, include=profile_frame)
        reconnect_attempts = 0
        if log_window_start is None:
            log_window_start = time.perf_counter()

        t_bgr0 = time.perf_counter()
        bgr = tensor_bgr_from_decoded(t)
        profiler.add("tensor_to_bgr", time.perf_counter() - t_bgr0, include=profile_frame)
        render_timings: dict[str, float] | None = {} if profile_enabled else None
        overlay = render_depth_overlay(
            model, bgr, width, height, alpha, model_profile, timings=render_timings
        )
        if render_timings:
            profiler.add("model_run", render_timings.get("model_s", 0.0), include=profile_frame)
            profiler.add("postprocess", render_timings.get("post_s", 0.0), include=profile_frame)
        if writer is None:
            writer = cv2.VideoWriter(
                output_file,
                cv2.VideoWriter_fourcc(*"mp4v"),
                max(1, fps),
                (width, height),
                True,
            )
            if not writer.isOpened():
                raise RuntimeError(f"Failed to open output video: {output_file}")
        t_write0 = time.perf_counter()
        writer.write(overlay)
        profiler.add("writer_write", time.perf_counter() - t_write0, include=profile_frame)
        processed += 1
        profiler.add("frame_total", time.perf_counter() - t_frame0, include=profile_frame)

        if processed % log_every == 0:
            now = time.perf_counter()
            interval_frames = processed - log_window_start_frames
            interval_elapsed = max(1e-6, now - log_window_start)
            print(
                f"[PROGRESS] frames={processed} "
                f"FPS={interval_frames/interval_elapsed:.2f} "
                f"(last {interval_frames} frames)"
            )
            log_window_start = now
            log_window_start_frames = processed
            profiler.print(f"frames={processed}")
            if profile_enabled and rtsp_run is not None:
                print_queue_profile(rtsp_run, queue2_depth)

    profiler.print("summary")
    if profile_enabled and rtsp_run is not None:
        print_queue_profile(rtsp_run, queue2_depth)
    return processed, writer, rtsp_session, rtsp_run


def main() -> int:
    p = argparse.ArgumentParser(description="RTSP depth estimation (MiDaS v2.1 / Depth Anything V2)")
    p.add_argument(
        "--model",
        required=True,
        help="Path to supported depth compiled model package (e.g. midas_v21_small_256 or depth_anything_v2_vits)",
    )
    p.add_argument("--rtsp", required=True, help="RTSP URL")
    p.add_argument("--frames", type=int, default=300, help="Number of frames to process")
    p.add_argument(
        "--width",
        type=int,
        default=0,
        help="Output/model width (default: model-dependent, 256 for MiDaS, 518 for Depth Anything V2)",
    )
    p.add_argument(
        "--height",
        type=int,
        default=0,
        help="Output/model height (default: model-dependent, 256 for MiDaS, 518 for Depth Anything V2)",
    )
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--latency-ms", type=int, default=200)
    p.add_argument("--tcp", action="store_true")
    p.add_argument("--alpha", type=float, default=0.6)
    p.add_argument(
        "--sample-every",
        type=int,
        default=1,
        help="Process every Nth decoded frame from the RTSP stream (default: 1)",
    )
    p.add_argument(
        "--log-every",
        type=int,
        default=100,
        help="Print progress/profile every N processed frames (default: 100)",
    )
    p.add_argument(
        "--output-file",
        "--output",
        dest="output_file",
        type=str,
        default="midas_depth_overlay.mp4",
        help="Path to write the depth-overlay video (MP4)",
    )
    p.add_argument("--profile", action="store_true", help="Print simple per-stage timing breakdowns")
    args = p.parse_args()

    writer = None
    rtsp_run = None
    try:
        model_profile = detect_model_profile(args.model)
        width = args.width if args.width > 0 else model_profile.default_size
        height = args.height if args.height > 0 else model_profile.default_size
        print(
            f"Using model profile: {model_profile.name} "
            f"(input_format={model_profile.input_format}, size={width}x{height})"
        )
        # Lifecycle stage: make model runtime.
        model = make_model(args.model, width, height, model_profile)
        out_dir = os.path.dirname(os.path.abspath(args.output_file))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        source_width, source_height, source_fps = probe_rtsp(args.rtsp)
        print(
            f"[init] probed RTSP decode dims {source_width}x{source_height}"
            + (f" @{source_fps} fps" if source_fps > 0 else "")
        )

        profiler = StageProfiler(args.profile)
        log_every = max(1, args.log_every)
        processed = 0
        log_window_start = None
        log_window_start_frames = 0
        reconnect_attempts = 0
        max_reconnect_attempts = 8
        queue2_depth = None
        # Keep a reference to the Session object alive for the lifetime of the Run.
        rtsp_session, rtsp_run = build_rtsp_run(
            args.rtsp,
            width,
            height,
            source_fps,
            args.latency_ms,
            args.tcp,
            args.sample_every,
            fallback_width=source_width,
            fallback_height=source_height,
        )
        if args.profile:
            queue2_depth = parse_queue2_depth(rtsp_run.report())
        while processed < args.frames:
            t_frame0 = time.perf_counter()
            t_pull0 = time.perf_counter()
            t = rtsp_run.pull_tensor(timeout_ms=5000)
            pull_dt = time.perf_counter() - t_pull0
            if t is None:
                if processed > 0 and reconnect_attempts < max_reconnect_attempts:
                    reconnect_attempts += 1
                    print(
                        f"RTSP pull timed out; reconnecting ({reconnect_attempts}/{max_reconnect_attempts})",
                        file=sys.stderr,
                    )
                    try:
                        rtsp_run.close()
                    except Exception:
                        pass
                    rtsp_run = None
                    time.sleep(0.5)
                    rtsp_session, rtsp_run = build_rtsp_run(
                        args.rtsp,
                        width,
                        height,
                        source_fps,
                        args.latency_ms,
                        args.tcp,
                        args.sample_every,
                        fallback_width=source_width,
                        fallback_height=source_height,
                    )
                    if args.profile:
                        queue2_depth = parse_queue2_depth(rtsp_run.report())
                    profiler.skip_next_frame()
                    continue
                print("RTSP pull timed out / stream closed", file=sys.stderr)
                break

            profile_frame = profiler.begin_frame()
            profiler.add("pull_ok", pull_dt, include=profile_frame)
            reconnect_attempts = 0
            if log_window_start is None:
                log_window_start = time.perf_counter()
            t_bgr0 = time.perf_counter()
            bgr = tensor_bgr_from_decoded(t)
            profiler.add("tensor_to_bgr", time.perf_counter() - t_bgr0, include=profile_frame)
            render_timings: dict[str, float] | None = {} if args.profile else None
            overlay = render_depth_overlay(
                model, bgr, width, height, args.alpha, model_profile, timings=render_timings
            )
            if render_timings:
                profiler.add("model_run", render_timings.get("model_s", 0.0), include=profile_frame)
                profiler.add("postprocess", render_timings.get("post_s", 0.0), include=profile_frame)
            if writer is None:
                writer = cv2.VideoWriter(
                    args.output_file,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    max(1, args.fps),
                    (width, height),
                    True,
                )
                if not writer.isOpened():
                    print(f"Failed to open output video: {args.output_file}", file=sys.stderr)
                    return 5
            t_write0 = time.perf_counter()
            writer.write(overlay)
            profiler.add("writer_write", time.perf_counter() - t_write0, include=profile_frame)
            processed += 1
            profiler.add("frame_total", time.perf_counter() - t_frame0, include=profile_frame)
            if processed % log_every == 0:
                now = time.perf_counter()
                interval_frames = processed - log_window_start_frames
                interval_elapsed = max(1e-6, now - log_window_start)
                print(
                    f"[PROGRESS] frames={processed} "
                    f"FPS={interval_frames/interval_elapsed:.2f} "
                    f"(last {interval_frames} frames)"
                )
                log_window_start = now
                log_window_start_frames = processed
                profiler.print(f"frames={processed}")
                if args.profile and rtsp_run is not None:
                    print_queue_profile(rtsp_run, queue2_depth)
        if processed == 0:
            shutdown(writer, rtsp_run)
            writer = None
            rtsp_run = None
            rtsp_session = None
            if os.path.exists(args.output_file):
                try:
                    os.remove(args.output_file)
                except OSError:
                    pass
            print("No frames were written; output file was not created.", file=sys.stderr)
            return 8
        shutdown(writer, rtsp_run)
        writer = None
        rtsp_run = None
        rtsp_session = None
        print(f"Wrote depth overlay video to: {os.path.abspath(args.output_file)}")
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2
    finally:
        # Always finalize resources, even if processing raises.
        shutdown(writer, rtsp_run)


if __name__ == "__main__":
    raise SystemExit(main())
