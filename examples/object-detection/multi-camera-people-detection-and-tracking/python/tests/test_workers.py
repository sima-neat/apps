"""Worker and profiling tests for the multi-camera people tracking example."""

from __future__ import annotations

import queue
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest


EXAMPLE_DIR = Path(__file__).resolve().parent.parent.parent
PYTHON_DIR = EXAMPLE_DIR / "python"

if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))


@pytest.mark.unit
class TestWorkers:
    def test_print_interval_profile_reports_wall_clock_throughput(self, capsys):
        from utils.workers import StreamMetrics, print_interval_profile

        metrics = StreamMetrics(
            processed=10,
            _interval_preproc_s=0.12,
            _interval_pull_s=0.11,
            _interval_output_s=0.03,
            _interval_loop_s=0.66,
            _interval_frames=10,
            _interval_wall_started_at_s=100.0,
            wall_last_processed_at_s=100.33,
        )
        stream = SimpleNamespace(index=2, metrics=metrics)

        print_interval_profile(stream)

        out = capsys.readouterr().out
        assert "throughput_fps=30.3" in out
        assert "push=" not in out

    def test_publish_thread_skips_overlay_when_not_saving_debug_frame(self, monkeypatch):
        from utils.config import AppConfig
        from utils.tracker import TrackedDetection
        from utils.workers import ResultPacket, StreamMetrics, publish_thread

        class FakeVideoRun:
            def __init__(self):
                self.calls = []

            def push(self, frame, copy=False, image_format=None):
                self.calls.append((frame, copy, image_format))
                return True

        class FakeJsonSender:
            def __init__(self):
                self.calls = []

            def send_detection(self, timestamp_ms, frame_id, objects, labels):
                self.calls.append((timestamp_ms, frame_id, objects, labels))
                return True

        cfg = AppConfig(
            model="model.tar.gz",
            rtsp_urls=["rtsp://camera-0"],
            output_dir=None,
            frames=1,
            optiview_host="127.0.0.1",
            optiview_video_port_base=9000,
            optiview_json_port_base=9100,
            fps=0,
            bitrate_kbps=2500,
            save_every=30,
            profile=False,
            person_class_id=0,
            detection_threshold=None,
            nms_iou_threshold=None,
            top_k=None,
            tracker_iou_threshold=0.3,
            tracker_max_missing=15,
            latency_ms=200,
            tcp=False,
        )
        video_run = FakeVideoRun()
        sender = FakeJsonSender()
        frame = SimpleNamespace(shape=(24, 32, 3), label="clean-frame")
        stream = SimpleNamespace(
            index=0,
            runtime=SimpleNamespace(pyneat=SimpleNamespace(PixelFormat=SimpleNamespace(RGB="rgb"))),
            tracker=SimpleNamespace(
                update=lambda boxes, frame_index: [
                    TrackedDetection(
                        track_id=5,
                        x1=1.0,
                        y1=2.0,
                        x2=11.0,
                        y2=22.0,
                        score=0.9,
                        class_id=0,
                    )
                ]
            ),
            video_run=video_run,
            json_sender=sender,
            metrics=StreamMetrics(),
            error=None,
        )
        result_q: queue.Queue = queue.Queue()
        result_q.put(
            ResultPacket(
                frame=frame,
                frame_index=0,
                bbox_payload=b"bbox",
                source_time_s=0.01,
                preproc_time_s=0.02,
                pull_wait_s=0.03,
            )
        )
        stop_event = threading.Event()

        monkeypatch.setattr("utils.workers.parse_bbox_payload", lambda payload, w, h: [{"class_id": 0, "x1": 1.0, "y1": 2.0, "x2": 11.0, "y2": 22.0, "score": 0.9}])
        monkeypatch.setattr("utils.workers.filter_person_detections", lambda boxes, person_class_id: boxes)
        sentinel_objects = ["objects"]
        sentinel_labels = ["Track ID: 5"]
        monkeypatch.setattr(
            "utils.workers.make_optiview_tracking_detection",
            lambda pyneat, tracked: (sentinel_objects, sentinel_labels),
        )
        monkeypatch.setattr("utils.workers.save_overlay_frame", lambda *args, **kwargs: False)
        monkeypatch.setattr("utils.workers.draw_tracked_people", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("overlay should not be rendered")))

        publish_thread(stream, cfg, result_q, stop_event)

        assert video_run.calls == [(frame, True, "rgb")]
        assert len(sender.calls) == 1
        _, frame_id, objects, labels = sender.calls[0]
        assert frame_id == "0"
        assert objects is sentinel_objects
        assert labels is sentinel_labels

    def test_producer_thread_throttles_to_cfg_fps(self, monkeypatch):
        from utils.config import AppConfig
        from utils.workers import StreamMetrics, producer_thread

        class FakeSourceRun:
            def __init__(self):
                self.calls = 0

            def pull(self, timeout_ms=None):
                self.calls += 1
                return f"sample-{self.calls}"

        perf_values = iter([
            0.0,
            0.001,
            0.001,
            0.002,
            0.003,
            0.003,
            0.102,
            0.103,
            0.103,
        ])
        monkeypatch.setattr("utils.workers.time.perf_counter", lambda: next(perf_values))
        monkeypatch.setattr(
            "utils.workers.tensor_bgr_from_sample",
            lambda runtime, sample: {"frame": sample},
        )

        cfg = AppConfig(
            model="model.tar.gz",
            rtsp_urls=["rtsp://camera-0"],
            output_dir=None,
            frames=2,
            optiview_host="127.0.0.1",
            optiview_video_port_base=9000,
            optiview_json_port_base=9100,
            fps=10,
            bitrate_kbps=2500,
            save_every=0,
            profile=False,
            person_class_id=0,
            detection_threshold=None,
            nms_iou_threshold=None,
            top_k=None,
            tracker_iou_threshold=0.3,
            tracker_max_missing=15,
            latency_ms=200,
            tcp=False,
        )
        stream = SimpleNamespace(
            index=0,
            source_run=FakeSourceRun(),
            runtime=SimpleNamespace(),
            metrics=StreamMetrics(),
            error=None,
        )
        frame_q: queue.Queue = queue.Queue(maxsize=4)
        stop_event = threading.Event()

        producer_thread(stream, cfg, frame_q, stop_event)

        assert stream.source_run.calls == 3
        assert frame_q.qsize() == 2
        first = frame_q.get_nowait()
        second = frame_q.get_nowait()
        assert first.frame == {"frame": "sample-1"}
        assert second.frame == {"frame": "sample-3"}
        assert first.frame_index == 0
        assert second.frame_index == 1
