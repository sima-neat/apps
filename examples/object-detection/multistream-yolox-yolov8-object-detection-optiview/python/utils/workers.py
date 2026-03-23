"""Worker-pool orchestration for the multistream detection example."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import queue
import threading
import time
from typing import Any

from .config import AppConfig
from .image_utils import (
    QuantTessCpuPreprocState,
    build_cpu_quanttess_preproc_state,
    cpu_quanttess_input,
    draw_detection_boxes,
    save_debug_frame,
)
from .pipeline import (
    _SOURCE_PULL_TIMEOUT_MS,
    _SOURCE_STARTUP_PULL_TIMEOUT_MS,
    _SOURCE_STARTUP_STAGGER_S,
    QuantTessCpuPreproc,
    RtspProbe,
    RuntimeModules,
    build_detection_run,
    build_optiview_json_output,
    build_optiview_video_run,
    build_source_run,
    effective_writer_fps,
    load_detector_model,
    load_runtime_modules,
    optiview_json_port_for_stream,
    optiview_video_port_for_stream,
    probe_rtsp,
    read_preproc_contract,
    source_output_every_n,
)
from .sample_utils import (
    detections_from_detector_sample,
    make_optiview_detection_payload,
    optiview_frame_id,
    optiview_timestamp_ms,
    tensor_rgb_from_sample,
)


_DEFAULT_PROFILE_INTERVAL_FRAMES = 200
_DEFAULT_LABELS_PATH = Path(__file__).resolve().parents[2] / "common" / "coco_label.txt"


@dataclass
class StreamMetrics:
    pulled: int = 0
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
    _interval_source_s: float = 0.0
    _interval_preproc_s: float = 0.0
    _interval_detect_s: float = 0.0
    _interval_video_s: float = 0.0
    _interval_json_s: float = 0.0
    _interval_publish_s: float = 0.0
    _interval_loop_s: float = 0.0
    _interval_frames: int = 0
    _interval_wall_started_at_s: float | None = None


@dataclass
class FramePacket:
    frame: Any
    frame_index: int
    source_time_s: float


DetectorRuntimeKey = tuple[str, int, int]


class LatestFrameMailbox:
    def __init__(self, stream_index: int, capacity: int) -> None:
        self.stream_index = int(stream_index)
        self._capacity = max(1, int(capacity))
        self._items: list[Any] = []
        self._closed = False
        self._ready_notified = False
        self._in_flight = False
        self._lock = threading.Lock()

    def push(self, item: Any, ready_queue: queue.Queue[int]) -> int:
        with self._lock:
            if self._closed:
                return 0
            dropped = 0
            while len(self._items) >= self._capacity:
                self._items.pop(0)
                dropped += 1
            self._items.append(item)
            if not self._in_flight and not self._ready_notified:
                ready_queue.put(self.stream_index)
                self._ready_notified = True
            return dropped

    def take_for_processing(self) -> Any | None:
        with self._lock:
            if not self._items:
                self._ready_notified = False
                return None
            item = self._items.pop(0)
            self._in_flight = True
            self._ready_notified = False
            return item

    def complete(self, ready_queue: queue.Queue[int]) -> None:
        with self._lock:
            self._in_flight = False
            if self._items and not self._ready_notified and not self._closed:
                ready_queue.put(self.stream_index)
                self._ready_notified = True

    def close(self) -> None:
        with self._lock:
            self._closed = True

    def drained(self) -> bool:
        with self._lock:
            return self._closed and not self._items and not self._in_flight


@dataclass
class StreamRuntime:
    index: int
    url: str
    family: str
    probe: RtspProbe
    runtime: RuntimeModules
    quant_preproc_state: QuantTessCpuPreprocState
    source_session: Any
    source_run: Any
    video_session: Any
    video_run: Any
    json_sender: Any
    class_labels: list[str]
    metrics: StreamMetrics = field(default_factory=StreamMetrics)
    error: Exception | None = None


@dataclass
class DetectorWorkerRuntime:
    key: DetectorRuntimeKey
    session: Any
    run: Any


@dataclass
class DetectorWorkerContext:
    index: int
    detectors: dict[DetectorRuntimeKey, DetectorWorkerRuntime]


def load_class_labels(path: Path = _DEFAULT_LABELS_PATH) -> list[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def detector_runtime_key(family: str, probe: RtspProbe) -> DetectorRuntimeKey:
    return (str(family).strip().lower(), int(probe.width), int(probe.height))


def collect_detector_runtime_keys(streams: list[Any]) -> list[DetectorRuntimeKey]:
    keys: list[DetectorRuntimeKey] = []
    seen: set[DetectorRuntimeKey] = set()
    for stream in streams:
        key = detector_runtime_key(stream.family, stream.probe)
        if key in seen:
            continue
        seen.add(key)
        keys.append(key)
    return keys


def create_stream_runtime(
    index: int,
    url: str,
    cfg: AppConfig,
    family: str,
    quant_preproc: QuantTessCpuPreproc,
    class_labels: list[str],
) -> StreamRuntime:
    runtime = load_runtime_modules()
    probe = probe_rtsp(url)
    quant_preproc_state = build_cpu_quanttess_preproc_state(runtime, quant_preproc, probe.width, probe.height)
    source_session, source_run = build_source_run(runtime, cfg, url, probe)
    if cfg.video_enabled:
        video_session, video_run = build_optiview_video_run(runtime, cfg, probe, index)
    else:
        video_session, video_run = None, None
    json_sender = build_optiview_json_output(runtime, cfg, index)
    return StreamRuntime(
        index=index,
        url=url,
        family=family,
        probe=probe,
        runtime=runtime,
        quant_preproc_state=quant_preproc_state,
        source_session=source_session,
        source_run=source_run,
        video_session=video_session,
        video_run=video_run,
        json_sender=json_sender,
        class_labels=class_labels,
    )


def build_detector_worker_contexts(
    runtime: RuntimeModules,
    cfg: AppConfig,
    model: Any,
    quant_preproc: QuantTessCpuPreproc,
    worker_count: int,
    detector_keys: list[DetectorRuntimeKey],
) -> list[DetectorWorkerContext]:
    contexts: list[DetectorWorkerContext] = []
    for worker_index in range(worker_count):
        detectors: dict[DetectorRuntimeKey, DetectorWorkerRuntime] = {}
        for key in detector_keys:
            family, width, height = key
            probe = RtspProbe(width=width, height=height, fps=0)
            session, run = build_detection_run(runtime, cfg, model, family, probe, quant_preproc)
            detectors[key] = DetectorWorkerRuntime(key=key, session=session, run=run)
        contexts.append(DetectorWorkerContext(index=worker_index, detectors=detectors))
    return contexts


def close_stream_runtime(stream: StreamRuntime) -> None:
    for run in (stream.video_run, stream.source_run):
        try:
            if run is not None:
                run.close()
        except Exception:
            pass


def close_detector_worker_context(context: DetectorWorkerContext) -> None:
    for detector in context.detectors.values():
        try:
            if detector.run is not None:
                detector.run.close()
        except Exception:
            pass


def start_producer_threads_sequentially(
    producer_threads: list[threading.Thread],
    startup_events: list[threading.Event],
    stop_event: threading.Event,
    startup_timeout_ms: int = _SOURCE_STARTUP_PULL_TIMEOUT_MS,
    startup_stagger_s: float = _SOURCE_STARTUP_STAGGER_S,
) -> list[threading.Thread]:
    started_threads: list[threading.Thread] = []
    timeout_s = startup_timeout_ms / 1000.0

    for index, thread in enumerate(producer_threads):
        if stop_event.is_set():
            break
        thread.start()
        started_threads.append(thread)
        if not startup_events[index].wait(timeout_s):
            stop_event.set()
            break
        if stop_event.is_set():
            break
        if index + 1 < len(producer_threads) and startup_stagger_s > 0:
            time.sleep(startup_stagger_s)

    return started_threads


def producer_thread(
    stream: StreamRuntime,
    cfg: AppConfig,
    mailbox: LatestFrameMailbox,
    ready_queue: queue.Queue[int],
    stop_event: threading.Event,
    startup_ready: threading.Event | None = None,
) -> None:
    frame_index = 0
    empty_pulls = 0
    emit_period_s = 1.0 / cfg.fps if cfg.fps > 0 else 0.0
    next_allowed_emit_s: float | None = None
    try:
        while not stop_event.is_set():
            if cfg.frames > 0 and frame_index >= cfg.frames:
                break
            if emit_period_s > 0.0:
                now = time.perf_counter()
                if next_allowed_emit_s is None:
                    next_allowed_emit_s = now
                if now < next_allowed_emit_s:
                    time.sleep(next_allowed_emit_s - now)
                    continue
                while next_allowed_emit_s <= now:
                    next_allowed_emit_s += emit_period_s
            t0 = time.perf_counter()
            pull_timeout_ms = _SOURCE_STARTUP_PULL_TIMEOUT_MS if frame_index == 0 else _SOURCE_PULL_TIMEOUT_MS
            sample = stream.source_run.pull(timeout_ms=pull_timeout_ms)
            elapsed = time.perf_counter() - t0
            if sample is None:
                empty_pulls += 1
                if cfg.frames > 0 and empty_pulls >= 20:
                    raise RuntimeError(f"stream {stream.index} timed out waiting for RTSP frames")
                continue

            empty_pulls = 0

            frame = tensor_rgb_from_sample(stream.runtime, sample)
            stream.metrics.mailbox_drops += mailbox.push(
                FramePacket(frame=frame, frame_index=frame_index, source_time_s=elapsed),
                ready_queue,
            )
            if startup_ready is not None and frame_index == 0:
                startup_ready.set()
            frame_index += 1
    except Exception as exc:
        stream.error = exc
        stop_event.set()
        if startup_ready is not None:
            startup_ready.set()
    finally:
        mailbox.close()


def render_frame(stream: StreamRuntime, cfg: AppConfig, frame, detections: list[dict]):
    if cfg.video_mode == "clean":
        return frame
    return draw_detection_boxes(stream.runtime, frame.copy(), detections, stream.class_labels)


def process_frame(
    worker_context: DetectorWorkerContext,
    stream: StreamRuntime,
    cfg: AppConfig,
    packet: FramePacket,
) -> None:
    metrics = stream.metrics
    loop_start = time.perf_counter()
    if metrics.wall_started_at_s is None:
        metrics.wall_started_at_s = loop_start
    if metrics._interval_wall_started_at_s is None:
        metrics._interval_wall_started_at_s = loop_start

    preproc_t0 = time.perf_counter()
    quant_input = cpu_quanttess_input(stream.runtime, packet.frame, stream.quant_preproc_state)
    preproc_elapsed = time.perf_counter() - preproc_t0

    detector = worker_context.detectors[detector_runtime_key(stream.family, stream.probe)]
    detect_t0 = time.perf_counter()
    det_sample = detector.run.run(quant_input, timeout_ms=50000)
    detect_elapsed = time.perf_counter() - detect_t0
    if det_sample is None:
        raise RuntimeError(f"stream {stream.index} detect run timed out")

    detections = detections_from_detector_sample(
        stream.runtime.pyneat,
        stream.family,
        det_sample,
        stream.probe.width,
        stream.probe.height,
    )

    publish_t0 = time.perf_counter()
    needs_output_frame = (stream.video_run is not None) or bool(cfg.output_dir and cfg.save_every > 0)
    frame_out = render_frame(stream, cfg, packet.frame, detections) if needs_output_frame else packet.frame
    pyneat = stream.runtime.pyneat
    video_t0 = time.perf_counter()
    if stream.video_run is not None:
        if not stream.video_run.push(frame_out, copy=True, image_format=pyneat.PixelFormat.RGB):
            raise RuntimeError(f"stream {stream.index} OptiView video push failed")
        publish_wall_time_s = time.time()
    else:
        publish_wall_time_s = time.time()
    video_elapsed = time.perf_counter() - video_t0

    objects, labels = make_optiview_detection_payload(
        pyneat,
        detections,
        img_w=stream.probe.width,
        img_h=stream.probe.height,
        class_labels=stream.class_labels,
    )
    json_t0 = time.perf_counter()
    if not stream.json_sender.send_detection(
        optiview_timestamp_ms(publish_wall_time_s, cfg.optiview_json_offset_ms),
        optiview_frame_id(det_sample, packet.frame_index),
        objects,
        labels,
    ):
        raise RuntimeError(f"stream {stream.index} OptiView JSON send failed")
    json_elapsed = time.perf_counter() - json_t0

    output_dir = Path(cfg.output_dir) if cfg.output_dir else None
    if save_debug_frame(stream.runtime, output_dir, stream.index, packet.frame_index, frame_out, cfg.save_every):
        metrics.saved += 1
    publish_elapsed = time.perf_counter() - publish_t0

    metrics.pulled += 1
    metrics.processed += 1
    metrics.detections += len(detections)
    metrics.source_time_s += packet.source_time_s
    metrics._interval_source_s += packet.source_time_s
    metrics.preproc_time_s += preproc_elapsed
    metrics.detect_time_s += detect_elapsed
    metrics.video_push_time_s += video_elapsed
    metrics.json_time_s += json_elapsed
    metrics.publish_time_s += publish_elapsed
    total_elapsed = packet.source_time_s + preproc_elapsed + detect_elapsed + publish_elapsed
    metrics.total_loop_time_s += total_elapsed
    metrics._interval_preproc_s += preproc_elapsed
    metrics._interval_detect_s += detect_elapsed
    metrics._interval_video_s += video_elapsed
    metrics._interval_json_s += json_elapsed
    metrics._interval_publish_s += publish_elapsed
    metrics._interval_loop_s += total_elapsed
    metrics._interval_frames += 1
    metrics.wall_last_processed_at_s = time.perf_counter()


def all_mailboxes_drained(mailboxes: list[LatestFrameMailbox]) -> bool:
    return all(mailbox.drained() for mailbox in mailboxes)


def detector_worker(
    worker_context: DetectorWorkerContext,
    streams: list[StreamRuntime],
    cfg: AppConfig,
    mailboxes: list[LatestFrameMailbox],
    ready_queue: queue.Queue[int],
    stop_event: threading.Event,
) -> None:
    try:
        while True:
            if stop_event.is_set() and all_mailboxes_drained(mailboxes):
                return
            try:
                stream_index = ready_queue.get(timeout=0.1)
            except queue.Empty:
                if all_mailboxes_drained(mailboxes):
                    return
                continue

            stream = streams[stream_index]
            mailbox = mailboxes[stream_index]
            packet = mailbox.take_for_processing()
            if packet is None:
                if all_mailboxes_drained(mailboxes):
                    return
                continue

            try:
                process_frame(worker_context, stream, cfg, packet)
            except Exception as exc:
                stream.error = exc
                stop_event.set()
            finally:
                mailbox.complete(ready_queue)
                if stop_event.is_set() and all_mailboxes_drained(mailboxes):
                    return
    except Exception as exc:
        for stream in streams:
            if stream.error is None:
                stream.error = exc
        stop_event.set()


def wall_clock_fps(frame_count: int, started_at_s: float | None, ended_at_s: float | None) -> float:
    if frame_count <= 0 or started_at_s is None or ended_at_s is None:
        return 0.0
    elapsed_s = ended_at_s - started_at_s
    if elapsed_s <= 0:
        return 0.0
    return frame_count / elapsed_s


def print_interval_profile(stream: StreamRuntime) -> None:
    m = stream.metrics
    n = m._interval_frames
    if n <= 0:
        return
    src_ms = m._interval_source_s * 1000.0 / n
    pre_ms = m._interval_preproc_s * 1000.0 / n
    det_ms = m._interval_detect_s * 1000.0 / n
    vid_ms = m._interval_video_s * 1000.0 / n
    json_ms = m._interval_json_s * 1000.0 / n
    pub_ms = m._interval_publish_s * 1000.0 / n
    loop_ms = m._interval_loop_s * 1000.0 / n
    throughput_fps = wall_clock_fps(n, m._interval_wall_started_at_s, m.wall_last_processed_at_s)
    print(
        f"  [stream {stream.index}] frames {m.processed - n}-{m.processed - 1} | "
        f"source={src_ms:.1f}ms preproc={pre_ms:.1f}ms detect={det_ms:.1f}ms "
        f"video={vid_ms:.1f}ms json={json_ms:.1f}ms publish={pub_ms:.1f}ms "
        f"loop={loop_ms:.1f}ms throughput_fps={throughput_fps:.1f} "
        f"mailbox_drops={m.mailbox_drops}"
    )
    m._interval_source_s = 0.0
    m._interval_preproc_s = 0.0
    m._interval_detect_s = 0.0
    m._interval_video_s = 0.0
    m._interval_json_s = 0.0
    m._interval_publish_s = 0.0
    m._interval_loop_s = 0.0
    m._interval_frames = 0
    m._interval_wall_started_at_s = m.wall_last_processed_at_s


def print_profile_summary(streams: list[StreamRuntime]) -> None:
    print("\nProfile summary (averages per frame):")
    for stream in streams:
        m = stream.metrics
        n = max(m.processed, 1)
        src = m.source_time_s * 1000.0 / n
        pre = m.preproc_time_s * 1000.0 / n
        det = m.detect_time_s * 1000.0 / n
        vid = m.video_push_time_s * 1000.0 / n
        jsn = m.json_time_s * 1000.0 / n
        pub = m.publish_time_s * 1000.0 / n
        loop = m.total_loop_time_s * 1000.0 / n
        throughput_fps = wall_clock_fps(m.processed, m.wall_started_at_s, m.wall_last_processed_at_s)
        print(
            f"  [stream {stream.index}] {m.processed} frames | "
            f"source={src:.1f}ms preproc={pre:.1f}ms detect={det:.1f}ms "
            f"video={vid:.1f}ms json={jsn:.1f}ms publish={pub:.1f}ms "
            f"loop={loop:.1f}ms throughput_fps={throughput_fps:.1f} "
            f"mailbox_drops={m.mailbox_drops} detections={m.detections}"
        )


def run_app(cfg: AppConfig, family: str) -> int:
    if cfg.output_dir:
        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    runtime = load_runtime_modules()
    class_labels = load_class_labels()
    try:
        model = load_detector_model(runtime, cfg)
        quant_preproc = read_preproc_contract(runtime, model)
    except Exception as exc:
        print(f"Error: failed to build model: {exc}", flush=True)
        return 3

    streams: list[StreamRuntime] = []
    try:
        for index, url in enumerate(cfg.rtsp_urls):
            streams.append(create_stream_runtime(index, url, cfg, family, quant_preproc, class_labels))
    except Exception as exc:
        print(f"Error: failed to set up stream runtimes: {exc}", flush=True)
        for stream in streams:
            close_stream_runtime(stream)
        return 4

    try:
        worker_contexts = build_detector_worker_contexts(
            runtime,
            cfg,
            model,
            quant_preproc,
            cfg.worker_count,
            collect_detector_runtime_keys(streams),
        )
    except Exception as exc:
        print(f"Error: failed to build detector workers: {exc}", flush=True)
        for stream in streams:
            close_stream_runtime(stream)
        return 4

    for stream in streams:
        print(
            f"[stream {stream.index}] {stream.probe.width}x{stream.probe.height} "
            f"@{effective_writer_fps(cfg, stream.probe)}fps {stream.url} -> optiview://{cfg.optiview_host} "
            f"video="
            f"{optiview_video_port_for_stream(cfg.optiview_video_port_base, stream.index) if cfg.video_enabled else 'disabled'} "
            f"json={optiview_json_port_for_stream(cfg.optiview_json_port_base, stream.index)}"
        )

    stop_event = threading.Event()
    ready_queue: queue.Queue[int] = queue.Queue()
    mailboxes = [LatestFrameMailbox(stream.index, cfg.mailbox_depth) for stream in streams]
    worker_threads: list[threading.Thread] = []
    producer_threads: list[threading.Thread] = []
    producer_ready_events: list[threading.Event] = []

    for worker_context in worker_contexts:
        worker_threads.append(
            threading.Thread(
                target=detector_worker,
                args=(worker_context, streams, cfg, mailboxes, ready_queue, stop_event),
                name=f"detector-{worker_context.index}",
                daemon=True,
            )
        )

    for stream, mailbox in zip(streams, mailboxes):
        ready = threading.Event()
        producer_ready_events.append(ready)
        producer_threads.append(
            threading.Thread(
                target=producer_thread,
                args=(stream, cfg, mailbox, ready_queue, stop_event, ready),
                name=f"producer-{stream.index}",
                daemon=True,
            )
        )

    try:
        for thread in worker_threads:
            thread.start()
        started_threads = list(worker_threads)
        started_threads.extend(start_producer_threads_sequentially(producer_threads, producer_ready_events, stop_event))
        for thread in started_threads:
            thread.join()
    except KeyboardInterrupt:
        stop_event.set()
        for stream in streams:
            close_stream_runtime(stream)
        for worker_context in worker_contexts:
            close_detector_worker_context(worker_context)
        for mailbox in mailboxes:
            mailbox.close()
        for thread in worker_threads + producer_threads:
            thread.join(timeout=5)
    finally:
        for stream in streams:
            close_stream_runtime(stream)
        for worker_context in worker_contexts:
            close_detector_worker_context(worker_context)

    failed = [stream for stream in streams if stream.error is not None]
    if failed:
        for stream in failed:
            print(f"[stream {stream.index}] error: {stream.error}", flush=True)
        return 5

    if cfg.profile:
        print_profile_summary(streams)
    return 0
