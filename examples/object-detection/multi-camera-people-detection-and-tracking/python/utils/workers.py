"""Worker threads, profiling, and app orchestration for the Python example."""

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
    draw_tracked_people,
    save_overlay_frame,
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
)
from .sample_utils import (
    extract_bbox_payload,
    filter_person_detections,
    make_optiview_tracking_detection,
    parse_bbox_payload,
    tensor_bgr_from_sample,
)
from .tracker import PeopleTracker


_DEFAULT_PROFILE_INTERVAL_FRAMES = 200


@dataclass
class StreamMetrics:
    pulled: int = 0
    processed: int = 0
    detections: int = 0
    saved: int = 0
    frame_q_drops: int = 0
    result_q_drops: int = 0
    frame_q_peak: int = 0
    result_q_peak: int = 0
    source_time_s: float = 0.0
    preproc_time_s: float = 0.0
    pull_wait_s: float = 0.0
    track_time_s: float = 0.0
    overlay_time_s: float = 0.0
    write_time_s: float = 0.0
    total_loop_time_s: float = 0.0
    wall_started_at_s: float | None = None
    wall_last_processed_at_s: float | None = None
    _interval_source_s: float = 0.0
    _interval_preproc_s: float = 0.0
    _interval_pull_s: float = 0.0
    _interval_output_s: float = 0.0
    _interval_loop_s: float = 0.0
    _interval_frames: int = 0
    _interval_frame_q_drops: int = 0
    _interval_result_q_drops: int = 0
    _interval_wall_started_at_s: float | None = None


@dataclass
class FramePacket:
    frame: Any
    frame_index: int
    source_time_s: float


@dataclass
class ResultPacket:
    frame: Any
    frame_index: int
    bbox_payload: bytes | None
    source_time_s: float
    preproc_time_s: float
    pull_wait_s: float


@dataclass
class StreamRuntime:
    index: int
    url: str
    probe: RtspProbe
    runtime: RuntimeModules
    model: Any
    quant_preproc_state: QuantTessCpuPreprocState
    # Keep sessions alive for as long as the runs built from them are in use.
    source_session: Any
    source_run: Any
    detect_session: Any
    detect_run: Any
    video_session: Any
    video_run: Any
    json_sender: Any
    tracker: PeopleTracker
    metrics: StreamMetrics = field(default_factory=StreamMetrics)
    error: Exception | None = None


def create_stream_runtime(
    index: int,
    url: str,
    cfg: AppConfig,
    model: Any,
    quant_preproc: QuantTessCpuPreproc,
) -> StreamRuntime:
    runtime = load_runtime_modules()
    probe = probe_rtsp(url)
    quant_preproc_state = build_cpu_quanttess_preproc_state(
        runtime,
        quant_preproc,
        probe.width,
        probe.height,
    )
    source_session, source_run = build_source_run(runtime, cfg, url, probe)
    detect_session, detect_run = build_detection_run(runtime, cfg, model, probe, quant_preproc)
    video_session, video_run = build_optiview_video_run(runtime, cfg, probe, index)
    json_sender = build_optiview_json_output(runtime, cfg, index)
    tracker = PeopleTracker(
        iou_threshold=cfg.tracker_iou_threshold,
        max_missing_frames=cfg.tracker_max_missing,
    )
    return StreamRuntime(
        index=index,
        url=url,
        probe=probe,
        runtime=runtime,
        model=model,
        quant_preproc_state=quant_preproc_state,
        source_session=source_session,
        source_run=source_run,
        detect_session=detect_session,
        detect_run=detect_run,
        video_session=video_session,
        video_run=video_run,
        json_sender=json_sender,
        tracker=tracker,
    )


def close_stream_runtime(stream: StreamRuntime) -> None:
    for run in (stream.video_run, stream.detect_run, stream.source_run):
        try:
            if run is not None:
                run.close()
        except Exception:
            pass


def record_queue_depth(stream: StreamRuntime, queue_name: str, q: queue.Queue) -> None:
    size = q.qsize()
    if queue_name == "frame":
        stream.metrics.frame_q_peak = max(stream.metrics.frame_q_peak, size)
    else:
        stream.metrics.result_q_peak = max(stream.metrics.result_q_peak, size)


def put_keep_latest(q: queue.Queue, item: Any, stream: StreamRuntime, queue_name: str) -> None:
    record_queue_depth(stream, queue_name, q)
    while True:
        try:
            q.put_nowait(item)
            record_queue_depth(stream, queue_name, q)
            return
        except queue.Full:
            if queue_name == "frame":
                stream.metrics.frame_q_drops += 1
                stream.metrics._interval_frame_q_drops += 1
            else:
                stream.metrics.result_q_drops += 1
                stream.metrics._interval_result_q_drops += 1
            try:
                q.get_nowait()
            except queue.Empty:
                pass


def producer_thread(
    stream: StreamRuntime,
    cfg: AppConfig,
    frame_q: queue.Queue,
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
                return
            t0 = time.perf_counter()
            pull_timeout_ms = (
                _SOURCE_STARTUP_PULL_TIMEOUT_MS if frame_index == 0 else _SOURCE_PULL_TIMEOUT_MS
            )
            sample = stream.source_run.pull(timeout_ms=pull_timeout_ms)
            elapsed = time.perf_counter() - t0
            if sample is None:
                empty_pulls += 1
                if cfg.frames > 0 and empty_pulls >= 20:
                    raise RuntimeError(f"stream {stream.index} timed out waiting for RTSP frames")
                continue
            empty_pulls = 0
            if emit_period_s > 0.0:
                now = time.perf_counter()
                if next_allowed_emit_s is None:
                    next_allowed_emit_s = now
                if now < next_allowed_emit_s:
                    continue
                while next_allowed_emit_s <= now:
                    next_allowed_emit_s += emit_period_s
            frame = tensor_bgr_from_sample(stream.runtime, sample)
            put_keep_latest(
                frame_q,
                FramePacket(frame=frame, frame_index=frame_index, source_time_s=elapsed),
                stream,
                "frame",
            )
            if startup_ready is not None and frame_index == 0:
                startup_ready.set()
            frame_index += 1
    except Exception as exc:
        stream.error = exc
        stop_event.set()
        if startup_ready is not None:
            startup_ready.set()


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


def infer_thread(
    stream: StreamRuntime,
    cfg: AppConfig,
    frame_q: queue.Queue,
    result_q: queue.Queue,
    stop_event: threading.Event,
) -> None:
    runtime = stream.runtime
    detect_run = stream.detect_run

    try:
        while not stop_event.is_set():
            if cfg.frames > 0 and stream.metrics.processed >= cfg.frames:
                return

            try:
                pkt = frame_q.get(timeout=0.1)
            except queue.Empty:
                continue

            preproc_t0 = time.perf_counter()
            quant_input = cpu_quanttess_input(
                runtime,
                pkt.frame,
                stream.quant_preproc_state,
            )
            preproc_elapsed = time.perf_counter() - preproc_t0

            roundtrip_t0 = time.perf_counter()
            det_sample = detect_run.run(quant_input, timeout_ms=50000)
            roundtrip_elapsed = time.perf_counter() - roundtrip_t0

            if det_sample is None:
                raise RuntimeError(f"stream {stream.index} detect run timed out")

            bbox_payload = extract_bbox_payload(runtime.pyneat, det_sample)
            put_keep_latest(
                result_q,
                ResultPacket(
                    frame=pkt.frame,
                    frame_index=pkt.frame_index,
                    bbox_payload=bbox_payload,
                    source_time_s=pkt.source_time_s,
                    preproc_time_s=preproc_elapsed,
                    pull_wait_s=roundtrip_elapsed,
                ),
                stream,
                "result",
            )

    except Exception as exc:
        stream.error = exc
        stop_event.set()


def publish_thread(
    stream: StreamRuntime,
    cfg: AppConfig,
    result_q: queue.Queue,
    stop_event: threading.Event,
) -> None:
    runtime = stream.runtime
    output_dir = Path(cfg.output_dir) if cfg.output_dir else None
    profile_every = cfg.save_every if cfg.save_every > 0 else _DEFAULT_PROFILE_INTERVAL_FRAMES

    try:
        while not stop_event.is_set():
            if cfg.frames > 0 and stream.metrics.processed >= cfg.frames:
                return
            try:
                pkt = result_q.get(timeout=0.1)
            except queue.Empty:
                continue

            loop_start = time.perf_counter()
            if stream.metrics.wall_started_at_s is None:
                stream.metrics.wall_started_at_s = loop_start
            if stream.metrics._interval_wall_started_at_s is None:
                stream.metrics._interval_wall_started_at_s = loop_start

            stream.metrics.source_time_s += pkt.source_time_s
            stream.metrics._interval_source_s += pkt.source_time_s
            stream.metrics.preproc_time_s += pkt.preproc_time_s
            stream.metrics.pull_wait_s += pkt.pull_wait_s
            stream.metrics._interval_preproc_s += pkt.preproc_time_s
            stream.metrics._interval_pull_s += pkt.pull_wait_s
            stream.metrics.pulled += 1

            boxes = parse_bbox_payload(pkt.bbox_payload, pkt.frame.shape[1], pkt.frame.shape[0])
            boxes = filter_person_detections(boxes, cfg.person_class_id)

            track_t0 = time.perf_counter()
            tracked = stream.tracker.update(boxes, frame_index=pkt.frame_index)
            stream.metrics.track_time_s += time.perf_counter() - track_t0
            stream.metrics.detections += len(tracked)

            write_t0 = time.perf_counter()
            pyneat = runtime.pyneat
            if not stream.video_run.push(pkt.frame, copy=True, image_format=pyneat.PixelFormat.RGB):
                raise RuntimeError(f"stream {stream.index} OptiView video push failed")

            frame_id = str(stream.metrics.processed)
            objects, labels = make_optiview_tracking_detection(pyneat, tracked)
            if not stream.json_sender.send_detection(
                int(time.time() * 1000),
                frame_id,
                objects,
                labels,
            ):
                raise RuntimeError(f"stream {stream.index} OptiView JSON send failed")

            if output_dir is not None and cfg.save_every > 0 and pkt.frame_index % cfg.save_every == 0:
                overlay_t0 = time.perf_counter()
                overlay = draw_tracked_people(runtime, pkt.frame.copy(), tracked)
                stream.metrics.overlay_time_s += time.perf_counter() - overlay_t0
                if save_overlay_frame(
                    runtime,
                    output_dir,
                    stream.index,
                    pkt.frame_index,
                    overlay,
                    cfg.save_every,
                ):
                    stream.metrics.saved += 1
            stream.metrics.write_time_s += time.perf_counter() - write_t0

            completed_at = time.perf_counter()
            output_elapsed = completed_at - loop_start
            stream.metrics._interval_output_s += output_elapsed
            stream.metrics.processed += 1
            per_frame = pkt.source_time_s + pkt.preproc_time_s + pkt.pull_wait_s + output_elapsed
            stream.metrics.total_loop_time_s += per_frame
            stream.metrics._interval_loop_s += per_frame
            stream.metrics._interval_frames += 1
            stream.metrics.wall_last_processed_at_s = completed_at

            if cfg.profile and stream.metrics._interval_frames >= profile_every:
                print_interval_profile(stream)

    except Exception as exc:
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
    pull_ms = m._interval_pull_s * 1000.0 / n
    out_ms = m._interval_output_s * 1000.0 / n
    loop_ms = m._interval_loop_s * 1000.0 / n
    throughput_fps = wall_clock_fps(n, m._interval_wall_started_at_s, m.wall_last_processed_at_s)
    print(
        f"  [stream {stream.index}] frames {m.processed - n}-{m.processed - 1} | "
        f"src={src_ms:.1f}ms  preproc={pre_ms:.1f}ms  "
        f"pull_wait={pull_ms:.1f}ms  output={out_ms:.1f}ms  "
        f"loop={loop_ms:.1f}ms  throughput_fps={throughput_fps:.1f}  "
        f"frame_q(drops={m._interval_frame_q_drops},peak={m.frame_q_peak})  "
        f"result_q(drops={m._interval_result_q_drops},peak={m.result_q_peak})"
    )
    m._interval_source_s = 0.0
    m._interval_preproc_s = 0.0
    m._interval_pull_s = 0.0
    m._interval_output_s = 0.0
    m._interval_loop_s = 0.0
    m._interval_frames = 0
    m._interval_frame_q_drops = 0
    m._interval_result_q_drops = 0
    m._interval_wall_started_at_s = m.wall_last_processed_at_s


def print_profile_summary(streams: list[StreamRuntime]) -> None:
    print("\nProfile summary (averages per frame):")
    for stream in streams:
        m = stream.metrics
        n = max(m.processed, 1)
        src = m.source_time_s * 1000.0 / n
        pre = m.preproc_time_s * 1000.0 / n
        pll = m.pull_wait_s * 1000.0 / n
        trk = m.track_time_s * 1000.0 / n
        ovl = m.overlay_time_s * 1000.0 / n
        wrt = m.write_time_s * 1000.0 / n
        out = trk + ovl + wrt
        loop = m.total_loop_time_s * 1000.0 / n
        throughput_fps = wall_clock_fps(m.processed, m.wall_started_at_s, m.wall_last_processed_at_s)
        print(
            f"  [stream {stream.index}] {m.processed} frames | "
            f"src={src:.1f}ms  preproc={pre:.1f}ms  "
            f"pull_wait={pll:.1f}ms  output={out:.1f}ms "
            f"(track={trk:.1f} overlay={ovl:.1f} write={wrt:.1f})  "
            f"loop={loop:.1f}ms  throughput_fps={throughput_fps:.1f}  "
            f"frame_q(total_drops={m.frame_q_drops},peak={m.frame_q_peak})  "
            f"result_q(total_drops={m.result_q_drops},peak={m.result_q_peak})"
        )


def run_app(cfg: AppConfig) -> int:
    if cfg.output_dir:
        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    runtime = load_runtime_modules()
    try:
        model = load_detector_model(runtime, cfg)
        quant_preproc = read_preproc_contract(runtime, model)
    except Exception as exc:
        print(f"Error: failed to build model: {exc}", flush=True)
        return 3

    streams: list[StreamRuntime] = []
    try:
        for index, url in enumerate(cfg.rtsp_urls):
            stream = create_stream_runtime(index, url, cfg, model, quant_preproc)
            streams.append(stream)
    except Exception as exc:
        print(f"Error: failed to set up stream runtimes: {exc}", flush=True)
        for stream in streams:
            close_stream_runtime(stream)
        return 4

    for stream in streams:
        print(
            f"[stream {stream.index}] {stream.probe.width}x{stream.probe.height} "
            f"@{effective_writer_fps(cfg, stream.probe)}fps "
            f"{stream.url} -> optiview://{cfg.optiview_host} "
            f"video={optiview_video_port_for_stream(cfg.optiview_video_port_base, stream.index)} "
            f"json={optiview_json_port_for_stream(cfg.optiview_json_port_base, stream.index)}"
        )

    stop_event = threading.Event()
    worker_threads: list[threading.Thread] = []
    producer_threads: list[threading.Thread] = []
    producer_ready_events: list[threading.Event] = []
    for stream in streams:
        frame_q: queue.Queue[FramePacket] = queue.Queue(maxsize=4)
        result_q: queue.Queue[ResultPacket] = queue.Queue(maxsize=4)
        producer_ready = threading.Event()
        producer_ready_events.append(producer_ready)
        producer_threads.append(
            threading.Thread(
                target=producer_thread,
                args=(stream, cfg, frame_q, stop_event, producer_ready),
                name=f"producer-{stream.index}",
                daemon=True,
            )
        )
        worker_threads.append(
            threading.Thread(
                target=infer_thread,
                args=(stream, cfg, frame_q, result_q, stop_event),
                name=f"infer-{stream.index}",
                daemon=True,
            )
        )
        worker_threads.append(
            threading.Thread(
                target=publish_thread,
                args=(stream, cfg, result_q, stop_event),
                name=f"publish-{stream.index}",
                daemon=True,
            )
        )

    try:
        for thread in worker_threads:
            thread.start()
        started_threads = list(worker_threads)
        started_threads.extend(
            start_producer_threads_sequentially(
                producer_threads,
                producer_ready_events,
                stop_event,
            )
        )
        for thread in started_threads:
            thread.join()
    except KeyboardInterrupt:
        stop_event.set()
        for thread in worker_threads + producer_threads:
            thread.join(timeout=5)
    finally:
        for stream in streams:
            close_stream_runtime(stream)

    failed = [stream for stream in streams if stream.error is not None]
    if failed:
        for stream in failed:
            print(f"[stream {stream.index}] error: {stream.error}", flush=True)
        return 5

    if cfg.profile:
        print_profile_summary(streams)
    return 0
