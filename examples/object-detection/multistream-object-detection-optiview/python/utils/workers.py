"""Worker threads, profiling, and app orchestration for the Python example."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
import os
import sys
import threading
import time
from typing import Any

from .config import AppConfig, VideoMode, json_output_enabled
from .image_utils import draw_detection_boxes, save_debug_frame
from .model_family import ModelFamily
from .pipeline import (
    RtspProbe,
    RuntimeModules,
    SessionRun,
    build_detection_run,
    build_optiview_json_output,
    build_optiview_video_run,
    build_source_run,
    effective_writer_fps,
    load_runtime_modules,
    optiview_json_port_for_stream,
    optiview_video_port_for_stream,
    probe_rtsp,
    producer_emit_period_s,
)
from .sample_utils import (
    Detection,
    build_optiview_detection_payload,
    detections_from_detector_sample,
    optiview_frame_id,
    optiview_timestamp_ms,
    tensor_rgb_from_sample,
)


SOURCE_STARTUP_PULL_TIMEOUT_MS = 50000
SOURCE_PULL_TIMEOUT_MS = 10000
SOURCE_STARTUP_STAGGER_S = 0.5

_DEFAULT_PROFILE_INTERVAL_FRAMES = 200
_DEFAULT_LABELS_PATH = Path(__file__).resolve().parents[2] / "common" / "coco_label.txt"


@dataclass(frozen=True)
class DetectorRuntimeKey:
    family: ModelFamily
    width: int
    height: int


@dataclass(frozen=True)
class StreamProbeSpec:
    family: ModelFamily
    probe: RtspProbe


@dataclass
class StreamMetrics:
    processed: int = 0
    detections: int = 0
    saved: int = 0
    mailbox_drops: int = 0
    source_time_s: float = 0.0
    preproc_time_s: float = 0.0
    detect_time_s: float = 0.0
    video_push_time_s: float = 0.0
    json_time_s: float = 0.0
    publish_time_s: float = 0.0
    total_loop_time_s: float = 0.0
    wall_started_at_s: float | None = None
    wall_last_processed_at_s: float | None = None
    interval_source_s: float = 0.0
    interval_preproc_s: float = 0.0
    interval_detect_s: float = 0.0
    interval_video_s: float = 0.0
    interval_json_s: float = 0.0
    interval_publish_s: float = 0.0
    interval_loop_s: float = 0.0
    interval_frames: int = 0
    interval_wall_started_at_s: float | None = None


@dataclass
class FramePacket:
    frame: Any
    frame_index: int = -1
    source_time_s: float = 0.0


@dataclass
class StreamRuntime:
    index: int = 0
    url: str = ""
    family: ModelFamily = ModelFamily.AUTO
    runtime: RuntimeModules | None = None
    probe: RtspProbe | None = None
    source: SessionRun | None = None
    video: SessionRun | None = None
    video_enabled: bool = True
    json_sender: Any | None = None
    json_enabled: bool = True
    class_labels: list[str] = field(default_factory=list)
    metrics: StreamMetrics = field(default_factory=StreamMetrics)
    error_message: str = ""
    saw_first_source_frame: bool = False
    first_mailbox_push_logged: bool = False
    next_source_frame_index: int = 0
    next_allowed_emit_s: float | None = None


@dataclass
class DetectorRuntime:
    key: DetectorRuntimeKey
    runtime: SessionRun


@dataclass
class WorkerContext:
    index: int = 0
    detectors: list[DetectorRuntime] = field(default_factory=list)


class ReadyStreamQueue:
    def __init__(self) -> None:
        self._queue: deque[int] = deque()
        self._closed = False
        self._cv = threading.Condition()

    def push(self, stream_index: int) -> None:
        with self._cv:
            if self._closed:
                return
            self._queue.append(int(stream_index))
            self._cv.notify()

    def pop_wait(self, timeout_s: float | None) -> int | None:
        with self._cv:
            if timeout_s is None or timeout_s < 0:
                while not self._closed and not self._queue:
                    self._cv.wait()
            else:
                end = time.monotonic() + timeout_s
                while not self._closed and not self._queue:
                    remaining = end - time.monotonic()
                    if remaining <= 0:
                        return None
                    self._cv.wait(remaining)
            if not self._queue:
                return None
            return self._queue.popleft()

    def close(self) -> None:
        with self._cv:
            self._closed = True
            self._cv.notify_all()


class LatestFrameMailbox:
    def __init__(self, stream_index: int, capacity: int) -> None:
        self._stream_index = int(stream_index)
        self._capacity = max(1, int(capacity))
        self._queue: deque[Any] = deque()
        self._closed = False
        self._ready_notified = False
        self._in_flight = False
        self._mu = threading.Lock()

    def push(self, item: Any, ready_queue: ReadyStreamQueue) -> int:
        with self._mu:
            if self._closed:
                return 0
            dropped = 0
            while len(self._queue) >= self._capacity:
                self._queue.popleft()
                dropped += 1
            self._queue.append(item)
            if not self._in_flight and not self._ready_notified:
                ready_queue.push(self._stream_index)
                self._ready_notified = True
            return dropped

    def take_for_processing(self) -> Any | None:
        with self._mu:
            if not self._queue:
                self._ready_notified = False
                return None
            item = self._queue.popleft()
            self._in_flight = True
            self._ready_notified = False
            return item

    def complete(self, ready_queue: ReadyStreamQueue) -> None:
        with self._mu:
            self._in_flight = False
            if self._queue and not self._ready_notified:
                ready_queue.push(self._stream_index)
                self._ready_notified = True

    def close(self) -> None:
        with self._mu:
            self._closed = True

    def drained(self) -> bool:
        with self._mu:
            return self._closed and not self._queue and not self._in_flight


def startup_trace_enabled_from_env() -> bool:
    raw = os.getenv("SIMA_OPTIVIEW_STARTUP_TRACE")
    if raw is None:
        return False
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _emit_startup_trace(stream_index: int, message: str) -> None:
    if startup_trace_enabled_from_env():
        print(f"[startup trace stream {stream_index}] {message}", file=sys.stderr, flush=True)


def _now_steady_s() -> float:
    return time.perf_counter()


def format_video_build_error(stream_index: int, video_mode: VideoMode, detail: str) -> str:
    return (
        f"stream {stream_index} failed to build OptiView "
        f"{video_mode.value} video run: {detail}"
    )


def _load_class_labels(path: Path = _DEFAULT_LABELS_PATH) -> list[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _detector_runtime_key(family: ModelFamily, probe: RtspProbe) -> DetectorRuntimeKey:
    return DetectorRuntimeKey(family=family, width=probe.width, height=probe.height)


def collect_detector_runtime_keys(streams: list[StreamProbeSpec]) -> list[DetectorRuntimeKey]:
    keys: list[DetectorRuntimeKey] = []
    for stream in streams:
        key = _detector_runtime_key(stream.family, stream.probe)
        if key not in keys:
            keys.append(key)
    return keys


def _initialize_stream_runtime(
    runtime: RuntimeModules,
    index: int,
    url: str,
    cfg: AppConfig,
    family: ModelFamily,
    class_labels: list[str],
) -> StreamRuntime:
    probe = probe_rtsp(cfg, url)
    source = build_source_run(runtime, cfg, url, probe)
    json_enabled = json_output_enabled(cfg)
    json_sender = None
    if json_enabled:
        json_sender = build_optiview_json_output(runtime, cfg, index)
    return StreamRuntime(
        index=index,
        url=url,
        family=family,
        runtime=runtime,
        probe=probe,
        source=source,
        video_enabled=cfg.video_enabled,
        json_sender=json_sender,
        json_enabled=json_enabled,
        class_labels=list(class_labels),
    )


def _build_worker_context(
    runtime: RuntimeModules,
    worker_index: int,
    cfg: AppConfig,
    detector_keys: list[DetectorRuntimeKey],
) -> WorkerContext:
    context = WorkerContext(index=worker_index)
    for key in detector_keys:
        probe = RtspProbe(width=key.width, height=key.height, fps=0)
        context.detectors.append(
            DetectorRuntime(
                key=key,
                runtime=build_detection_run(runtime, cfg, key.family, probe),
            )
        )
    return context


def _build_worker_contexts(
    runtime: RuntimeModules,
    cfg: AppConfig,
    detector_keys: list[DetectorRuntimeKey],
) -> list[WorkerContext]:
    return [
        _build_worker_context(runtime, worker_index, cfg, detector_keys)
        for worker_index in range(max(cfg.worker_count, 0))
    ]


def _close_stream_runtime(stream: StreamRuntime) -> None:
    for runtime in (stream.video, stream.source):
        if runtime is None:
            continue
        try:
            runtime.run.close()
        except Exception:
            pass


def _close_worker_context(context: WorkerContext) -> None:
    for detector in context.detectors:
        try:
            detector.runtime.run.close()
        except Exception:
            pass


def _find_detector_runtime(
    context: WorkerContext,
    family: ModelFamily,
    probe: RtspProbe,
) -> DetectorRuntime:
    key = _detector_runtime_key(family, probe)
    for detector in context.detectors:
        if detector.key == key:
            return detector
    raise RuntimeError("missing detector runtime for stream geometry")


def _any_stream_failed(streams: list[StreamRuntime]) -> bool:
    return any(stream.error_message for stream in streams)


def _render_frame(
    runtime: RuntimeModules,
    stream: StreamRuntime,
    cfg: AppConfig,
    frame,
    detections: list[Detection],
):
    if cfg.video_mode is VideoMode.CLEAN:
        return frame
    return draw_detection_boxes(runtime, frame.copy(), detections, stream.class_labels)


def _optiview_objects(runtime: RuntimeModules, payload) -> list[Any]:
    objects: list[Any] = []
    for item in payload.objects:
        obj = runtime.pyneat.OptiViewObject()
        obj.x = int(item["x"])
        obj.y = int(item["y"])
        obj.w = int(item["w"])
        obj.h = int(item["h"])
        obj.score = float(item["score"])
        obj.class_id = int(item["class_id"])
        objects.append(obj)
    return objects


def _process_frame(
    worker_context: WorkerContext,
    stream: StreamRuntime,
    cfg: AppConfig,
    packet: FramePacket,
) -> None:
    assert stream.probe is not None
    assert stream.runtime is not None

    loop_start = _now_steady_s()
    if stream.metrics.wall_started_at_s is None:
        stream.metrics.wall_started_at_s = loop_start
    if stream.metrics.interval_wall_started_at_s is None:
        stream.metrics.interval_wall_started_at_s = loop_start

    detector = _find_detector_runtime(worker_context, stream.family, stream.probe)
    input_tensor = stream.runtime.pyneat.Tensor.from_numpy(
        stream.runtime.np.ascontiguousarray(packet.frame),
        copy=True,
        image_format=stream.runtime.pyneat.PixelFormat.RGB,
    )
    detect_t0 = _now_steady_s()
    det_sample = detector.runtime.run.run(input_tensor, timeout_ms=50000)
    detect_elapsed = _now_steady_s() - detect_t0
    if det_sample is None:
        raise RuntimeError(f"stream {stream.index} detect run timed out")

    preproc_elapsed = 0.0
    detections = detections_from_detector_sample(
        stream.family,
        det_sample,
        stream.probe.width,
        stream.probe.height,
    )

    needs_saved_frame = cfg.output_dir is not None and cfg.save_every > 0
    needs_rendered_frame = needs_saved_frame or (
        stream.video_enabled and cfg.video_mode is VideoMode.ANNOTATED
    )
    frame_out = None
    if needs_rendered_frame:
        frame_out = _render_frame(stream.runtime, stream, cfg, packet.frame, detections)

    publish_t0 = _now_steady_s()
    video_elapsed = 0.0
    if stream.video_enabled:
        video_t0 = _now_steady_s()
        if stream.video is None:
            try:
                stream.video = build_optiview_video_run(
                    stream.runtime,
                    cfg,
                    stream.probe,
                    stream.index,
                )
            except Exception as exc:
                raise RuntimeError(format_video_build_error(stream.index, cfg.video_mode, str(exc)))

        video_frame = packet.frame if cfg.video_mode is VideoMode.CLEAN else frame_out
        if not stream.video.run.push(
            video_frame,
            copy=True,
            image_format=stream.runtime.pyneat.PixelFormat.RGB,
        ):
            if cfg.video_mode is VideoMode.CLEAN:
                raise RuntimeError(f"stream {stream.index} OptiView clean video push failed")
            raise RuntimeError(f"stream {stream.index} OptiView video push failed")
        video_elapsed = _now_steady_s() - video_t0

    publish_wall_time_s = time.time()
    json_elapsed = 0.0
    if stream.json_enabled and stream.json_sender is not None:
        payload = build_optiview_detection_payload(
            detections,
            stream.probe.width,
            stream.probe.height,
            stream.class_labels,
        )
        objects = _optiview_objects(stream.runtime, payload)
        json_t0 = _now_steady_s()
        ok = stream.json_sender.send_detection(
            optiview_timestamp_ms(publish_wall_time_s, cfg.optiview_json_offset_ms),
            optiview_frame_id(det_sample, packet.frame_index),
            objects,
            payload.labels,
        )
        if not ok:
            raise RuntimeError(f"stream {stream.index} OptiView JSON send failed")
        json_elapsed = _now_steady_s() - json_t0

    if frame_out is not None and needs_saved_frame:
        if save_debug_frame(
            cfg.output_dir,
            stream.index,
            packet.frame_index,
            frame_out,
            cfg.save_every,
            runtime=stream.runtime,
        ):
            stream.metrics.saved += 1

    publish_elapsed = _now_steady_s() - publish_t0

    stream.metrics.processed += 1
    stream.metrics.detections += len(detections)
    stream.metrics.source_time_s += packet.source_time_s
    stream.metrics.interval_source_s += packet.source_time_s
    stream.metrics.preproc_time_s += preproc_elapsed
    stream.metrics.interval_preproc_s += preproc_elapsed
    stream.metrics.detect_time_s += detect_elapsed
    stream.metrics.interval_detect_s += detect_elapsed
    stream.metrics.video_push_time_s += video_elapsed
    stream.metrics.interval_video_s += video_elapsed
    stream.metrics.json_time_s += json_elapsed
    stream.metrics.interval_json_s += json_elapsed
    stream.metrics.publish_time_s += publish_elapsed
    stream.metrics.interval_publish_s += publish_elapsed
    total_elapsed = packet.source_time_s + preproc_elapsed + detect_elapsed + publish_elapsed
    stream.metrics.total_loop_time_s += total_elapsed
    stream.metrics.interval_loop_s += total_elapsed
    stream.metrics.interval_frames += 1
    stream.metrics.wall_last_processed_at_s = _now_steady_s()


def _all_mailboxes_drained(mailboxes: list[LatestFrameMailbox]) -> bool:
    return all(mailbox.drained() for mailbox in mailboxes)


def producer_thread(
    stream: StreamRuntime,
    cfg: AppConfig,
    mailbox: LatestFrameMailbox,
    ready_queue: ReadyStreamQueue,
    stop_event: threading.Event,
    startup_ready: threading.Event | None = None,
) -> None:
    assert stream.source is not None
    assert stream.runtime is not None

    try:
        _emit_startup_trace(stream.index, "source thread started")
        while not stop_event.is_set():
            if cfg.frames > 0 and stream.next_source_frame_index >= cfg.frames:
                break

            pull_timeout_ms = (
                SOURCE_PULL_TIMEOUT_MS
                if stream.first_mailbox_push_logged
                else SOURCE_STARTUP_PULL_TIMEOUT_MS
            )
            pull_t0 = _now_steady_s()
            sample = stream.source.run.pull(timeout_ms=pull_timeout_ms)
            pull_elapsed = _now_steady_s() - pull_t0
            if sample is None:
                try:
                    if not stream.source.run.running():
                        raise RuntimeError("source run stopped")
                except Exception:
                    raise
                continue

            if not stream.saw_first_source_frame:
                _emit_startup_trace(stream.index, "first decoded frame pulled")
                stream.saw_first_source_frame = True

            frame = tensor_rgb_from_sample(stream.runtime, sample)
            source_completed_at_s = _now_steady_s()
            should_emit = True
            emit_period_s = producer_emit_period_s(cfg, stream.probe)
            if emit_period_s > 0.0:
                if stream.next_allowed_emit_s is None:
                    num_streams = max(len(cfg.rtsp_urls), 1)
                    phase = emit_period_s * stream.index / num_streams if num_streams > 1 else 0.0
                    stream.next_allowed_emit_s = (
                        (source_completed_at_s // emit_period_s) + 1.0
                    ) * emit_period_s + phase
                if source_completed_at_s < stream.next_allowed_emit_s:
                    should_emit = False
                else:
                    while stream.next_allowed_emit_s <= source_completed_at_s:
                        stream.next_allowed_emit_s += emit_period_s

            if not stream.first_mailbox_push_logged:
                should_emit = True
            if not should_emit:
                stream.next_source_frame_index += 1
                continue

            packet = FramePacket(
                frame=frame,
                frame_index=stream.next_source_frame_index,
                source_time_s=pull_elapsed,
            )
            stream.metrics.mailbox_drops += mailbox.push(packet, ready_queue)
            if not stream.first_mailbox_push_logged:
                _emit_startup_trace(stream.index, "first mailbox push complete")
                stream.first_mailbox_push_logged = True
                if startup_ready is not None:
                    startup_ready.set()
            stream.next_source_frame_index += 1
    except Exception as exc:
        stream.error_message = str(exc)
        stop_event.set()
        if startup_ready is not None:
            startup_ready.set()
    finally:
        mailbox.close()
        try:
            stream.source.run.close()
        except Exception:
            pass


def detector_worker(
    worker_context: WorkerContext,
    streams: list[StreamRuntime],
    cfg: AppConfig,
    mailboxes: list[LatestFrameMailbox],
    ready_queue: ReadyStreamQueue,
    stop_event: threading.Event,
) -> None:
    try:
        while True:
            if stop_event.is_set() and _all_mailboxes_drained(mailboxes):
                return

            stream_index = ready_queue.pop_wait(0.1)
            if stream_index is None:
                if _all_mailboxes_drained(mailboxes):
                    return
                continue

            stream = streams[stream_index]
            mailbox = mailboxes[stream_index]
            packet = mailbox.take_for_processing()
            if packet is None:
                if _all_mailboxes_drained(mailboxes):
                    return
                continue

            try:
                _process_frame(worker_context, stream, cfg, packet)
            except Exception as exc:
                stream.error_message = str(exc)
                stop_event.set()

            mailbox.complete(ready_queue)
            if (
                cfg.profile
                and stream.metrics.processed > 0
                and (stream.metrics.processed % _DEFAULT_PROFILE_INTERVAL_FRAMES) == 0
            ):
                _print_interval_profile(stream)
    except Exception as exc:
        for stream in streams:
            if not stream.error_message:
                stream.error_message = str(exc)
        stop_event.set()


def _wall_clock_fps(frame_count: int, started_at_s: float | None, ended_at_s: float | None) -> float:
    if frame_count <= 0 or started_at_s is None or ended_at_s is None:
        return 0.0
    elapsed = ended_at_s - started_at_s
    if elapsed <= 0.0:
        return 0.0
    return frame_count / elapsed


def _print_interval_profile(stream: StreamRuntime) -> None:
    n = stream.metrics.interval_frames
    if n <= 0:
        return
    fps = _wall_clock_fps(
        n,
        stream.metrics.interval_wall_started_at_s,
        stream.metrics.wall_last_processed_at_s,
    )
    print(
        f"  [stream {stream.index}] frames {stream.metrics.processed - n}-{stream.metrics.processed - 1} | "
        f"source={stream.metrics.interval_source_s * 1000.0 / n:.6g}ms "
        f"preproc={stream.metrics.interval_preproc_s * 1000.0 / n:.6g}ms "
        f"detect={stream.metrics.interval_detect_s * 1000.0 / n:.6g}ms "
        f"video={stream.metrics.interval_video_s * 1000.0 / n:.6g}ms "
        f"json={stream.metrics.interval_json_s * 1000.0 / n:.6g}ms "
        f"publish={stream.metrics.interval_publish_s * 1000.0 / n:.6g}ms "
        f"loop={stream.metrics.interval_loop_s * 1000.0 / n:.6g}ms "
        f"throughput_fps={fps:.6g} mailbox_drops={stream.metrics.mailbox_drops}",
        flush=True,
    )
    stream.metrics.interval_source_s = 0.0
    stream.metrics.interval_preproc_s = 0.0
    stream.metrics.interval_detect_s = 0.0
    stream.metrics.interval_video_s = 0.0
    stream.metrics.interval_json_s = 0.0
    stream.metrics.interval_publish_s = 0.0
    stream.metrics.interval_loop_s = 0.0
    stream.metrics.interval_frames = 0
    stream.metrics.interval_wall_started_at_s = stream.metrics.wall_last_processed_at_s


def _print_profile_summary(streams: list[StreamRuntime]) -> None:
    print("\nProfile summary (averages per frame):", flush=True)
    for stream in streams:
        n = max(stream.metrics.processed, 1)
        fps = _wall_clock_fps(
            stream.metrics.processed,
            stream.metrics.wall_started_at_s,
            stream.metrics.wall_last_processed_at_s,
        )
        print(
            f"  [stream {stream.index}] {stream.metrics.processed} frames | "
            f"source={stream.metrics.source_time_s * 1000.0 / n:.6g}ms "
            f"preproc={stream.metrics.preproc_time_s * 1000.0 / n:.6g}ms "
            f"detect={stream.metrics.detect_time_s * 1000.0 / n:.6g}ms "
            f"video={stream.metrics.video_push_time_s * 1000.0 / n:.6g}ms "
            f"json={stream.metrics.json_time_s * 1000.0 / n:.6g}ms "
            f"publish={stream.metrics.publish_time_s * 1000.0 / n:.6g}ms "
            f"loop={stream.metrics.total_loop_time_s * 1000.0 / n:.6g}ms "
            f"throughput_fps={fps:.6g} mailbox_drops={stream.metrics.mailbox_drops} "
            f"detections={stream.metrics.detections}",
            flush=True,
        )


def run_app(cfg: AppConfig, family: ModelFamily) -> int:
    runtime = load_runtime_modules()
    if cfg.output_dir is not None:
        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    class_labels = _load_class_labels()
    streams: list[StreamRuntime] = []
    try:
        for index, url in enumerate(cfg.rtsp_urls):
            streams.append(
                _initialize_stream_runtime(
                    runtime,
                    index,
                    url,
                    cfg,
                    family,
                    class_labels,
                )
            )
    except Exception as exc:
        print(f"Error: failed to set up stream runtimes: {exc}", file=sys.stderr, flush=True)
        for stream in streams:
            _close_stream_runtime(stream)
        return 4

    detector_runtime_keys = collect_detector_runtime_keys(
        [
            StreamProbeSpec(family=stream.family, probe=stream.probe)
            for stream in streams
            if stream.probe is not None
        ]
    )

    for stream in streams:
        assert stream.probe is not None
        video = (
            str(optiview_video_port_for_stream(cfg.optiview_video_port_base, stream.index))
            if stream.video_enabled
            else "disabled"
        )
        json = (
            str(optiview_json_port_for_stream(cfg.optiview_json_port_base, stream.index))
            if stream.json_enabled
            else "disabled"
        )
        print(
            f"[stream {stream.index}] {stream.probe.width}x{stream.probe.height} "
            f"@{effective_writer_fps(cfg, stream.probe)}fps {stream.url} -> "
            f"optiview://{cfg.optiview_host} video={video} json={json}",
            flush=True,
        )

    ready_queue = ReadyStreamQueue()
    mailboxes = [LatestFrameMailbox(stream.index, cfg.mailbox_depth) for stream in streams]
    stop_event = threading.Event()
    worker_contexts: list[WorkerContext] = []
    worker_threads: list[threading.Thread] = []
    producer_threads: list[threading.Thread] = []

    try:
        for index, stream in enumerate(streams):
            startup_ready = threading.Event()
            thread = threading.Thread(
                target=producer_thread,
                args=(stream, cfg, mailboxes[index], ready_queue, stop_event, startup_ready),
                name=f"source-{index}",
            )
            producer_threads.append(thread)
            thread.start()
            if not startup_ready.wait(SOURCE_STARTUP_PULL_TIMEOUT_MS / 1000.0):
                _emit_startup_trace(stream.index, "startup wait_for timed out waiting for first decoded frame")
                stream.error_message = "startup timeout waiting for first decoded frame"
                stop_event.set()
                try:
                    stream.source.run.close()
                except Exception:
                    pass
                break
            if _any_stream_failed(streams):
                stop_event.set()
                break
            if index + 1 < len(streams) and SOURCE_STARTUP_STAGGER_S > 0:
                time.sleep(SOURCE_STARTUP_STAGGER_S)

        if not stop_event.is_set() and not _any_stream_failed(streams):
            worker_contexts = _build_worker_contexts(runtime, cfg, detector_runtime_keys)
            for worker_context in worker_contexts:
                thread = threading.Thread(
                    target=detector_worker,
                    args=(worker_context, streams, cfg, mailboxes, ready_queue, stop_event),
                    name=f"detect-{worker_context.index}",
                )
                worker_threads.append(thread)
                thread.start()

        for thread in producer_threads:
            thread.join()
    except KeyboardInterrupt:
        stop_event.set()
    except Exception as exc:
        stop_event.set()
        for mailbox in mailboxes:
            mailbox.close()
        for stream in streams:
            _close_stream_runtime(stream)
        for worker_context in worker_contexts:
            _close_worker_context(worker_context)
        for thread in producer_threads:
            if thread.is_alive():
                thread.join()
        for thread in worker_threads:
            if thread.is_alive():
                thread.join()
        print(f"Error: runtime setup failed: {exc}", file=sys.stderr, flush=True)
        return 4

    stop_event.set()
    for thread in worker_threads:
        thread.join()
    ready_queue.close()
    for stream in streams:
        _close_stream_runtime(stream)
    for worker_context in worker_contexts:
        _close_worker_context(worker_context)

    failed = False
    for stream in streams:
        if stream.error_message:
            failed = True
            print(f"[stream {stream.index}] error: {stream.error_message}", file=sys.stderr, flush=True)

    if failed:
        return 5
    if cfg.profile:
        _print_profile_summary(streams)
    return 0
