"""Unit tests for multi-camera-people-detection-and-tracking (Python)."""

from __future__ import annotations

import json
import subprocess
import sys
import queue
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest


EXAMPLE_DIR = Path(__file__).resolve().parent.parent.parent
MAIN_PY = EXAMPLE_DIR / "python" / "main.py"
PYTHON_DIR = EXAMPLE_DIR / "python"

if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))


@pytest.mark.unit
class TestArgParsing:
    """Validate CLI argument parsing for the multi-camera tracking example."""

    def test_help_runs(self):
        result = subprocess.run(
            [sys.executable, str(MAIN_PY), "--help"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=str(EXAMPLE_DIR),
        )
        assert result.returncode == 0
        assert "--model" in result.stdout
        assert "--rtsp" in result.stdout

    def test_missing_all_args(self):
        result = subprocess.run(
            [sys.executable, str(MAIN_PY)],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=str(EXAMPLE_DIR),
        )
        assert result.returncode == 2
        assert "error" in result.stderr.lower()

    def test_missing_model(self):
        result = subprocess.run(
            [sys.executable, str(MAIN_PY), "--rtsp", "rtsp://camera-0"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=str(EXAMPLE_DIR),
        )
        assert result.returncode == 2
        assert "--model" in result.stderr or "required" in result.stderr.lower()

    def test_missing_rtsp(self):
        result = subprocess.run(
            [sys.executable, str(MAIN_PY), "--model", "model.tar.gz"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=str(EXAMPLE_DIR),
        )
        assert result.returncode == 2
        assert "--rtsp" in result.stderr or "required" in result.stderr.lower()

    def test_invalid_udp_port_base(self):
        result = subprocess.run(
            [
                sys.executable,
                str(MAIN_PY),
                "--model",
                "model.tar.gz",
                "--rtsp",
                "rtsp://camera-0",
                "--optiview-video-port-base",
                "0",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=str(EXAMPLE_DIR),
        )
        assert result.returncode != 0
        assert "optiview" in result.stderr.lower() or "port" in result.stderr.lower()

    def test_invalid_optiview_json_port_base(self):
        result = subprocess.run(
            [
                sys.executable,
                str(MAIN_PY),
                "--model",
                "model.tar.gz",
                "--rtsp",
                "rtsp://camera-0",
                "--optiview-json-port-base",
                "0",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=str(EXAMPLE_DIR),
        )
        assert result.returncode != 0
        assert "optiview" in result.stderr.lower() or "port" in result.stderr.lower()

    def test_invalid_save_every(self):
        result = subprocess.run(
            [
                sys.executable,
                str(MAIN_PY),
                "--model",
                "model.tar.gz",
                "--rtsp",
                "rtsp://camera-0",
                "--output",
                "/tmp/tracking-out",
                "--save-every",
                "0",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=str(EXAMPLE_DIR),
        )
        assert result.returncode != 0
        assert "save" in result.stderr.lower()


@pytest.mark.unit
class TestTracker:
    def test_reuses_track_id_for_nearby_detection(self):
        from tracker import PeopleTracker

        tracker = PeopleTracker(iou_threshold=0.3, max_missing_frames=2)

        first = tracker.update(
            [{"x1": 10.0, "y1": 10.0, "x2": 50.0, "y2": 80.0, "score": 0.9, "class_id": 0}],
            frame_index=0,
        )
        second = tracker.update(
            [{"x1": 12.0, "y1": 12.0, "x2": 52.0, "y2": 82.0, "score": 0.88, "class_id": 0}],
            frame_index=1,
        )

        assert len(first) == 1
        assert len(second) == 1
        assert second[0].track_id == first[0].track_id

    def test_creates_new_track_for_non_overlapping_detection(self):
        from tracker import PeopleTracker

        tracker = PeopleTracker(iou_threshold=0.3, max_missing_frames=2)

        first = tracker.update(
            [{"x1": 10.0, "y1": 10.0, "x2": 50.0, "y2": 80.0, "score": 0.9, "class_id": 0}],
            frame_index=0,
        )
        second = tracker.update(
            [{"x1": 110.0, "y1": 10.0, "x2": 150.0, "y2": 80.0, "score": 0.85, "class_id": 0}],
            frame_index=1,
        )

        assert len(first) == 1
        assert len(second) == 1
        assert second[0].track_id != first[0].track_id

    def test_drops_track_after_missing_budget(self):
        from tracker import PeopleTracker

        tracker = PeopleTracker(iou_threshold=0.3, max_missing_frames=1)

        tracker.update(
            [{"x1": 10.0, "y1": 10.0, "x2": 50.0, "y2": 80.0, "score": 0.9, "class_id": 0}],
            frame_index=0,
        )
        tracker.update([], frame_index=1)
        tracker.update([], frame_index=2)

        assert tracker.active_track_count() == 0


@pytest.mark.unit
class TestHelpers:
    def test_optiview_video_port_for_stream_offsets_from_base(self):
        import main as app_main

        assert app_main.optiview_video_port_for_stream(9000, 0) == 9000
        assert app_main.optiview_video_port_for_stream(9000, 3) == 9003

    def test_optiview_json_port_for_stream_offsets_from_base(self):
        import main as app_main

        assert app_main.optiview_json_port_for_stream(9100, 0) == 9100
        assert app_main.optiview_json_port_for_stream(9100, 3) == 9103

    def test_filter_person_detections_keeps_only_requested_class(self):
        import main as app_main

        boxes = [
            {"x1": 0.0, "y1": 0.0, "x2": 10.0, "y2": 20.0, "score": 0.9, "class_id": 0},
            {"x1": 1.0, "y1": 1.0, "x2": 12.0, "y2": 22.0, "score": 0.8, "class_id": 2},
        ]

        kept = app_main.filter_person_detections(boxes, person_class_id=0)
        assert kept == [boxes[0]]

    def test_sample_output_path_uses_per_stream_directory(self, tmp_path: Path):
        import main as app_main

        out = app_main.sample_output_path(tmp_path, stream_index=2, frame_index=17)
        assert out == tmp_path / "stream_2" / "frame_000017.jpg"

    def test_print_interval_profile_reports_latency_and_wall_clock_throughput(self, capsys):
        import main as app_main

        metrics = app_main.StreamMetrics(
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

        app_main.print_interval_profile(stream, profile_every=10)

        out = capsys.readouterr().out
        assert "throughput_fps=30.3" in out
        assert "latency_fps=" not in out

    def test_print_profile_summary_reports_latency_and_wall_clock_throughput(self, capsys):
        import main as app_main

        metrics = app_main.StreamMetrics(
            processed=30,
            source_time_s=0.99,
            preproc_time_s=0.42,
            pull_wait_s=0.39,
            track_time_s=0.03,
            overlay_time_s=0.03,
            write_time_s=0.03,
            total_loop_time_s=1.98,
            wall_started_at_s=200.0,
            wall_last_processed_at_s=200.99,
        )
        stream = SimpleNamespace(index=1, metrics=metrics)

        app_main.print_profile_summary([stream])

        out = capsys.readouterr().out
        assert "throughput_fps=30.3" in out
        assert "latency_fps=" not in out

    def test_producer_thread_uses_longer_timeout_for_first_frame(self, monkeypatch):
        import main as app_main

        class FakeRun:
            def __init__(self):
                self.timeouts = []

            def pull(self, timeout_ms):
                self.timeouts.append(timeout_ms)
                return "sample"

        cfg = app_main.AppConfig(
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
        stream = SimpleNamespace(
            source_run=FakeRun(),
            runtime=SimpleNamespace(),
            metrics=app_main.StreamMetrics(),
            error=None,
            index=0,
        )
        frame_q: queue.Queue = queue.Queue(maxsize=1)
        stop_event = threading.Event()
        startup_ready = threading.Event()

        monkeypatch.setattr(app_main, "tensor_bgr_from_sample", lambda runtime, sample: "frame")

        app_main.producer_thread(stream, cfg, frame_q, stop_event, startup_ready)

        assert stream.source_run.timeouts == [app_main._SOURCE_STARTUP_PULL_TIMEOUT_MS]
        assert startup_ready.is_set()
        pkt = frame_q.get_nowait()
        assert pkt.frame == "frame"
        assert pkt.frame_index == 0

    def test_start_producer_threads_sequentially_waits_for_ready_signal(self, monkeypatch):
        import main as app_main

        started = []
        slept = []

        class FakeThread:
            def __init__(self, name):
                self.name = name

            def start(self):
                started.append(self.name)

        class FakeEvent:
            def __init__(self):
                self.wait_calls = []

            def wait(self, timeout):
                self.wait_calls.append(timeout)
                return True

        stop_event = threading.Event()
        startup_events = [FakeEvent(), FakeEvent()]
        producer_threads = [FakeThread("producer-0"), FakeThread("producer-1")]

        monkeypatch.setattr(app_main.time, "sleep", lambda seconds: slept.append(seconds))

        started_threads = app_main.start_producer_threads_sequentially(
            producer_threads,
            startup_events,
            stop_event,
            startup_timeout_ms=1234,
            startup_stagger_s=0.25,
        )

        assert started_threads == producer_threads
        assert started == ["producer-0", "producer-1"]
        assert startup_events[0].wait_calls == [1.234]
        assert startup_events[1].wait_calls == [1.234]
        assert slept == [0.25]

    def test_load_detector_model_uses_tensor_input_contract(self):
        import main as app_main

        class FakeModelOptions:
            def __init__(self):
                self.media_type = ""
                self.format = "BGR"
                self.input_max_width = 0
                self.input_max_height = 0
                self.input_max_depth = 0

        class FakePyneat:
            def __init__(self):
                self.last_model_options = None

            def ModelOptions(self):
                return FakeModelOptions()

            def Model(self, path, options):
                self.last_model_options = options
                return ("model", path, options)

        runtime = app_main.RuntimeModules(cv2=None, np=None, pyneat=FakePyneat())
        cfg = app_main.AppConfig(
            model="model.tar.gz",
            rtsp_urls=["rtsp://camera-0"],
            output_dir=None,
            frames=0,
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

        model = app_main.load_detector_model(runtime, cfg)

        assert model[0] == "model"
        opt = runtime.pyneat.last_model_options
        assert opt.media_type == "application/vnd.simaai.tensor"
        assert opt.format == ""

    def test_cpu_quanttess_preprocess_letterboxes_bgr_frame_into_rgb_fp32_tensor(self):
        import main as app_main

        class FakeImage:
            def __init__(
                self,
                *,
                shape,
                channels="bgr",
                dtype="uint8",
                source=None,
                scale=None,
            ):
                self.shape = shape
                self.channels = channels
                self.dtype = dtype
                self.source = source
                self.scale = scale

            def __getitem__(self, item):
                if item == (Ellipsis, slice(None, None, -1)):
                    return FakeImage(
                        shape=self.shape,
                        channels="rgb",
                        dtype=self.dtype,
                        source=self.source or self,
                    )
                raise AssertionError(f"unexpected image slice: {item}")

            def astype(self, dtype):
                return FakeImage(
                    shape=self.shape,
                    channels=self.channels,
                    dtype=dtype,
                    source=self.source or self,
                    scale=self.scale,
                )

            def __truediv__(self, value):
                return FakeImage(
                    shape=self.shape,
                    channels=self.channels,
                    dtype=self.dtype,
                    source=self.source or self,
                    scale=value,
                )

        class FakeCanvas:
            def __init__(self, shape, dtype):
                self.shape = shape
                self.dtype = dtype
                self.assignments = []

            def __setitem__(self, key, value):
                self.assignments.append((key, value))

        class FakeCV2:
            COLOR_BGR2RGB = "bgr2rgb"
            INTER_LINEAR = "linear"

            @staticmethod
            def cvtColor(frame, code):
                assert code == FakeCV2.COLOR_BGR2RGB
                return frame[..., ::-1]

            @staticmethod
            def resize(image, size, interpolation=None):
                assert interpolation == FakeCV2.INTER_LINEAR
                dst_w, dst_h = size
                return FakeImage(
                    shape=(dst_h, dst_w, 3),
                    channels=image.channels,
                    dtype=image.dtype,
                    source=image.source or image,
                )

        class FakeNp:
            float32 = "float32"

            @staticmethod
            def zeros(shape, dtype=None):
                return FakeCanvas(shape, dtype)

            @staticmethod
            def ascontiguousarray(value):
                return value

        runtime = app_main.RuntimeModules(cv2=FakeCV2(), np=FakeNp(), pyneat=None)
        contract = app_main.QuantTessCpuPreproc(
            width=8,
            height=8,
            aspect_ratio=True,
            padding_type="CENTER",
        )
        frame_bgr = FakeImage(shape=(2, 4, 3), channels="bgr")

        quant_input = app_main.cpu_quanttess_input(runtime, frame_bgr, contract)

        assert quant_input.shape == (8, 8, 3)
        assert quant_input.dtype == runtime.np.float32
        assert len(quant_input.assignments) == 1
        key, value = quant_input.assignments[0]
        assert key == (slice(2, 6, None), slice(0, 8, None))
        assert value.shape == (4, 8, 3)
        assert value.channels == "rgb"
        assert value.dtype == runtime.np.float32
        assert value.scale == 255.0

    def test_build_detect_run_uses_quanttess_mla_and_boxdecode_tensor_contract(self):
        import main as app_main

        class FakeSession:
            def __init__(self):
                self.added = []
                self.build_calls = []

            def add(self, node):
                self.added.append(node)

            def build(self, *args, **kwargs):
                self.build_calls.append((args, kwargs))
                return "fake-run"

        class FakeQuantTessOptions:
            def __init__(self, model):
                self.model = model

        class FakeRunOptions:
            def __init__(self):
                self.queue_depth = None
                self.overflow_policy = None

        class FakePyneat:
            class OverflowPolicy:
                Block = "block"
                KeepLatest = "keep-latest"

            class RunMode:
                Async = "async"

            def __init__(self):
                self.last_session = None
                self.last_seed_tensor = None
                self.nodes = SimpleNamespace(
                    input=lambda opt=None: ("input", opt),
                    quant_tess=lambda opt: ("quant_tess", opt),
                    sima_box_decode=lambda model, **kwargs: ("boxdecode", model, kwargs),
                    output=lambda: ("output",),
                )
                self.groups = SimpleNamespace(mla=lambda model: ("mla", model))
                self.Tensor = SimpleNamespace(from_numpy=self._from_numpy)

            def Session(self):
                self.last_session = FakeSession()
                return self.last_session

            def _from_numpy(self, arr, copy=False, image_format=None):
                self.last_seed_tensor = {
                    "array": arr,
                    "copy": copy,
                    "image_format": image_format,
                }
                return ("tensor_from_numpy", arr, copy, image_format)

            def QuantTessOptions(self, model):
                return FakeQuantTessOptions(model)

            def RunOptions(self):
                return FakeRunOptions()

        class FakeModel:
            def input_appsrc_options(self, copy):
                return SimpleNamespace(
                    copy=copy,
                    media_type="application/vnd.simaai.tensor",
                    format="FP32",
                )

        class FakeNp:
            float32 = "float32"

            @staticmethod
            def zeros(shape, dtype=None):
                return {"shape": shape, "dtype": dtype}

            @staticmethod
            def ascontiguousarray(value):
                return value

        cfg = app_main.AppConfig(
            model="model.tar.gz",
            rtsp_urls=["rtsp://camera-0"],
            output_dir=None,
            frames=0,
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
        probe = app_main.RtspProbe(width=1280, height=720, fps=16)
        preproc = app_main.QuantTessCpuPreproc(
            width=640,
            height=640,
            aspect_ratio=True,
            padding_type="CENTER",
        )
        runtime = app_main.RuntimeModules(cv2=None, np=FakeNp(), pyneat=FakePyneat())

        _, run = app_main.build_detection_run(runtime, cfg, FakeModel(), probe, preproc)

        added_kinds = [node[0] for node in runtime.pyneat.last_session.added]
        assert run == "fake-run"
        assert added_kinds == ["input", "quant_tess", "mla", "boxdecode", "output"]

        input_opt = runtime.pyneat.last_session.added[0][1]
        assert input_opt.media_type == "application/vnd.simaai.tensor"
        assert input_opt.format == "FP32"

        boxdecode_kwargs = runtime.pyneat.last_session.added[3][2]
        assert boxdecode_kwargs["original_width"] == probe.width
        assert boxdecode_kwargs["original_height"] == probe.height
        assert boxdecode_kwargs["detection_threshold"] == app_main._YOLOV8_BOXDECODE_DEFAULTS["detection_threshold"]
        assert boxdecode_kwargs["nms_iou_threshold"] == app_main._YOLOV8_BOXDECODE_DEFAULTS["nms_iou_threshold"]
        assert boxdecode_kwargs["top_k"] == app_main._YOLOV8_BOXDECODE_DEFAULTS["topk"]

        (build_args, build_kwargs) = runtime.pyneat.last_session.build_calls[0]
        assert build_kwargs == {}
        assert build_args[0] == {"shape": (640, 640, 3), "dtype": runtime.np.float32}

    def test_build_source_run_uses_probed_resolution_and_explicit_run_options(self):
        import main as app_main

        class FakeCaps:
            def __init__(self):
                self.enable = False
                self.format = ""
                self.width = 0
                self.height = 0
                self.fps = 0
                self.memory = None

        class FakeRtspDecodedInputOptions:
            def __init__(self):
                self.url = ""
                self.latency_ms = 0
                self.tcp = False
                self.payload_type = 0
                self.insert_queue = False
                self.out_format = ""
                self.decoder_raw_output = True
                self.auto_caps_from_stream = False
                self.use_videoconvert = True
                self.use_videoscale = False
                self.fallback_h264_width = 0
                self.fallback_h264_height = 0
                self.fallback_h264_fps = 0
                self.output_caps = FakeCaps()

        class FakeRunOptions:
            def __init__(self):
                self.queue_depth = None
                self.overflow_policy = None
                self.output_memory = None

        class FakeSession:
            def __init__(self):
                self.added = []
                self.build_calls = []

            def add(self, node):
                self.added.append(node)

            def build(self, *args, **kwargs):
                self.build_calls.append((args, kwargs))
                return "fake-source-run"

        class FakePyneat:
            class CapsMemory:
                SystemMemory = "system"

            class OverflowPolicy:
                KeepLatest = "keep-latest"

            class OutputMemory:
                Owned = "owned"

            def __init__(self):
                self.last_session = None
                self.last_rtsp_options = None
                self.nodes = SimpleNamespace(
                    output=lambda opt=None: ("output", opt),
                    input=lambda opt=None: ("input", opt),
                )
                self.groups = SimpleNamespace(
                    rtsp_decoded_input=self._rtsp_decoded_input,
                )
                self.OutputOptions = SimpleNamespace(
                    every_frame=lambda every: ("every_frame", every)
                )

            def _rtsp_decoded_input(self, opt):
                self.last_rtsp_options = opt
                return ("rtsp_decoded_input", opt)

            def RtspDecodedInputOptions(self):
                return FakeRtspDecodedInputOptions()

            def Session(self):
                self.last_session = FakeSession()
                return self.last_session

            def RunOptions(self):
                return FakeRunOptions()

        cfg = app_main.AppConfig(
            model="model.tar.gz",
            rtsp_urls=["rtsp://camera-0"],
            output_dir=None,
            frames=0,
            optiview_host="127.0.0.1",
            optiview_video_port_base=9000,
            optiview_json_port_base=9100,
            fps=0,
            bitrate_kbps=2500,
            save_every=30,
            profile=False,
            person_class_id=0,
            detection_threshold=0.25,
            nms_iou_threshold=0.55,
            top_k=120,
            tracker_iou_threshold=0.3,
            tracker_max_missing=15,
            latency_ms=120,
            tcp=True,
        )
        probe = app_main.RtspProbe(width=1920, height=1080, fps=25)
        runtime = app_main.RuntimeModules(cv2=None, np=None, pyneat=FakePyneat())

        _, run = app_main.build_source_run(runtime, cfg, "rtsp://camera-0", probe)

        assert run == "fake-source-run"
        assert runtime.pyneat.last_rtsp_options.output_caps.width == probe.width
        assert runtime.pyneat.last_rtsp_options.output_caps.height == probe.height
        assert runtime.pyneat.last_rtsp_options.output_caps.fps == probe.fps
        assert runtime.pyneat.last_rtsp_options.output_caps.memory == runtime.pyneat.CapsMemory.SystemMemory

        assert len(runtime.pyneat.last_session.build_calls) == 1
        (build_args, build_kwargs) = runtime.pyneat.last_session.build_calls[0]
        assert build_kwargs == {}
        (run_opt,) = build_args
        assert run_opt.queue_depth == 4
        assert run_opt.overflow_policy == runtime.pyneat.OverflowPolicy.KeepLatest
        assert run_opt.output_memory == runtime.pyneat.OutputMemory.Owned

    def test_make_optiview_tracking_json_uses_track_ids(self):
        import main as app_main

        tracked = [
            app_main.TrackedDetection(
                track_id=7,
                x1=10.0,
                y1=20.0,
                x2=40.0,
                y2=60.0,
                score=0.95,
                class_id=0,
            )
        ]

        payload = app_main.make_optiview_tracking_json(1234, "frame-9", tracked)
        parsed = json.loads(payload)

        assert parsed["type"] == "object-detection"
        assert parsed["timestamp"] == 1234
        assert parsed["frame_id"] == "frame-9"
        assert parsed["data"]["objects"][0]["id"] == "track_7"
        assert parsed["data"]["objects"][0]["bbox"] == [10.0, 20.0, 30.0, 40.0]

    def test_build_optiview_video_run_inserts_video_convert_before_encoder(self):
        import main as app_main

        class FakeSession:
            def __init__(self):
                self.added = []
                self.build_calls = []

            def add(self, node):
                self.added.append(node)

            def build(self, *args, **kwargs):
                self.build_calls.append((args, kwargs))
                return "fake-video-run"

        class FakeRunOptions:
            def __init__(self):
                self.queue_depth = None
                self.overflow_policy = None

        class FakeUdpOptions:
            def __init__(self):
                self.udp_host = ""
                self.udp_port = 0
                self.config_interval = 0
                self.payload_type = 0
                self.udp_sync = True
                self.udp_async = True

        class FakePyneat:
            class OverflowPolicy:
                KeepLatest = "keep-latest"

            class RunMode:
                Async = "async"

            def __init__(self):
                self.last_session = None
                self.last_input_options = None
                self.last_seed_tensor = None
                self.nodes = SimpleNamespace(
                    input=self._input,
                    video_convert=lambda: ("video_convert",),
                    h264_encode_sima=lambda *args, **kwargs: ("h264_encode_sima", args, kwargs),
                )
                self.groups = SimpleNamespace(
                    udp_h264_output_group=lambda opt: ("udp_h264_output_group", opt)
                )
                self.PixelFormat = SimpleNamespace(BGR="bgr")
                self.Tensor = SimpleNamespace(from_numpy=self._from_numpy)

            def Session(self):
                self.last_session = FakeSession()
                return self.last_session

            def _input(self, opt=None):
                self.last_input_options = opt
                return ("input", opt)

            def _from_numpy(self, arr, copy=False, image_format=None):
                self.last_seed_tensor = {
                    "array": arr,
                    "copy": copy,
                    "image_format": image_format,
                }
                return ("tensor_from_numpy", arr, copy, image_format)

            def InputOptions(self):
                return SimpleNamespace(media_type="", format="", use_simaai_pool=True)

            def UdpH264OutputGroupOptions(self):
                return FakeUdpOptions()

            def RunOptions(self):
                return FakeRunOptions()

        class FakeNp:
            uint8 = "uint8"

            @staticmethod
            def zeros(shape, dtype=None):
                return {"shape": shape, "dtype": dtype}

        cfg = app_main.AppConfig(
            model="model.tar.gz",
            rtsp_urls=["rtsp://camera-0"],
            output_dir=None,
            frames=0,
            optiview_host="192.168.0.107",
            optiview_video_port_base=9000,
            optiview_json_port_base=9100,
            fps=0,
            bitrate_kbps=2500,
            save_every=30,
            profile=False,
            person_class_id=0,
            detection_threshold=0.25,
            nms_iou_threshold=0.55,
            top_k=120,
            tracker_iou_threshold=0.3,
            tracker_max_missing=15,
            latency_ms=120,
            tcp=True,
        )
        probe = app_main.RtspProbe(width=1920, height=1080, fps=25)
        runtime = app_main.RuntimeModules(cv2=None, np=FakeNp(), pyneat=FakePyneat())

        _, run = app_main.build_optiview_video_run(runtime, cfg, probe, stream_index=1)

        assert run == "fake-video-run"
        added_kinds = [node[0] for node in runtime.pyneat.last_session.added]
        assert added_kinds == ["input", "video_convert", "h264_encode_sima", "udp_h264_output_group"]
        assert runtime.pyneat.last_input_options.media_type == "video/x-raw"
        assert runtime.pyneat.last_input_options.format == "BGR"
        assert runtime.pyneat.last_input_options.use_simaai_pool is False

        udp_opt = runtime.pyneat.last_session.added[3][1]
        assert udp_opt.udp_host == cfg.optiview_host
        assert udp_opt.udp_port == 9001

        (build_args, build_kwargs) = runtime.pyneat.last_session.build_calls[0]
        assert build_kwargs == {}
        assert build_args[0][0] == "tensor_from_numpy"
        assert build_args[0][3] == runtime.pyneat.PixelFormat.BGR

    def test_build_optiview_json_output_uses_per_stream_ports(self):
        import main as app_main

        class FakeChannelOptions:
            def __init__(self):
                self.host = ""
                self.channel = 0
                self.video_port_base = 0
                self.json_port_base = 0

        class FakeJsonOutput:
            def __init__(self, opt):
                self.opt = opt

        class FakePyneat:
            def OptiViewChannelOptions(self):
                return FakeChannelOptions()

            def OptiViewJsonOutput(self, opt):
                return FakeJsonOutput(opt)

        cfg = app_main.AppConfig(
            model="model.tar.gz",
            rtsp_urls=["rtsp://camera-0"],
            output_dir=None,
            frames=0,
            optiview_host="192.168.0.107",
            optiview_video_port_base=9000,
            optiview_json_port_base=9100,
            fps=0,
            bitrate_kbps=2500,
            save_every=30,
            profile=False,
            person_class_id=0,
            detection_threshold=0.25,
            nms_iou_threshold=0.55,
            top_k=120,
            tracker_iou_threshold=0.3,
            tracker_max_missing=15,
            latency_ms=120,
            tcp=True,
        )
        runtime = app_main.RuntimeModules(cv2=None, np=None, pyneat=FakePyneat())

        sender = app_main.build_optiview_json_output(runtime, cfg, stream_index=2)

        assert sender.opt.host == cfg.optiview_host
        assert sender.opt.channel == 2
        assert sender.opt.video_port_base == cfg.optiview_video_port_base
        assert sender.opt.json_port_base == cfg.optiview_json_port_base

    def test_publish_thread_skips_overlay_when_not_saving_debug_frame(self, monkeypatch):
        import main as app_main

        class FakeVideoRun:
            def __init__(self):
                self.calls = []

            def push(self, frame, copy=False, image_format=None):
                self.calls.append((frame, copy, image_format))
                return True

        class FakeJsonSender:
            def __init__(self):
                self.payloads = []

            def send_json(self, payload):
                self.payloads.append(payload)
                return True

        cfg = app_main.AppConfig(
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
            runtime=SimpleNamespace(pyneat=SimpleNamespace(PixelFormat=SimpleNamespace(BGR="bgr"))),
            tracker=SimpleNamespace(
                update=lambda boxes, frame_index: [
                    app_main.TrackedDetection(
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
            metrics=app_main.StreamMetrics(),
            error=None,
        )
        result_q: queue.Queue = queue.Queue()
        result_q.put(
            app_main.ResultPacket(
                frame=frame,
                frame_index=0,
                bbox_payload=b"bbox",
                source_time_s=0.01,
                preproc_time_s=0.02,
                pull_wait_s=0.03,
            )
        )
        stop_event = threading.Event()

        monkeypatch.setattr(app_main, "parse_bbox_payload", lambda payload, w, h: [{"class_id": 0, "x1": 1.0, "y1": 2.0, "x2": 11.0, "y2": 22.0, "score": 0.9}])
        monkeypatch.setattr(app_main, "make_optiview_tracking_json", lambda timestamp_ms, frame_id, tracked: "json-payload")
        monkeypatch.setattr(app_main, "save_overlay_frame", lambda *args, **kwargs: False)
        monkeypatch.setattr(app_main, "draw_tracked_people", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("overlay should not be rendered")))

        app_main.publish_thread(stream, cfg, result_q, stop_event)

        assert video_run.calls == [(frame, True, "bgr")]
        assert sender.payloads == ["json-payload"]
