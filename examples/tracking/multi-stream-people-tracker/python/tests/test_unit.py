"""Unit tests for the multi-camera people tracking example."""

from __future__ import annotations

import queue
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest


EXAMPLE_DIR = Path(__file__).resolve().parent.parent.parent
PYTHON_DIR = EXAMPLE_DIR / "python"
MAIN_PY = EXAMPLE_DIR / "python" / "main.py"

if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

pytestmark = pytest.mark.unit


class TestUnitSuiteEntrypoint:
    def test_support_modules_import(self):
        import utils.config  # noqa: F401
        import utils.image_utils  # noqa: F401
        import utils.pipeline  # noqa: F401
        import utils.sample_utils  # noqa: F401
        import utils.tracker  # noqa: F401
        import utils.workers  # noqa: F401


class TestConfigLoading:
    def test_load_app_config_parses_dynamic_stream_list(self, tmp_path: Path):
        from utils.config import VideoMode, load_app_config, metadata_output_enabled

        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            """
model: assets/models/yolo_v8m_mpk.tar.gz
input:
  tcp: true
  latency_ms: 200
inference:
  frames: 0
  fps: 0
  bitrate_kbps: 2500
  profile: true
  person_class_id: 0
  detection_threshold: 0.25
  nms_iou_threshold: 0.5
  top_k: 24
tracking:
  iou_threshold: 0.3
  max_missing_frames: 15
output:
  insight:
    host: 192.168.0.107
    video_port_base: 9000
    metadata_port_base: 9100
  video_mode: annotated
  debug_dir: null
  save_every: 0
streams:
  - rtsp://192.168.0.235:8554/src1
  - rtsp://192.168.0.235:8554/src2
  - rtsp://192.168.0.235:8554/src3
""".strip(),
            encoding="utf-8",
        )

        cfg = load_app_config(config_path)

        assert cfg.model == "assets/models/yolo_v8m_mpk.tar.gz"
        assert cfg.insight_host == "192.168.0.107"
        assert cfg.insight_video_port_base == 9000
        assert cfg.insight_metadata_port_base == 9100
        assert cfg.video_mode is VideoMode.ANNOTATED
        assert metadata_output_enabled(cfg) is False
        assert cfg.tcp is True
        assert cfg.save_every == 0
        assert cfg.detection_threshold == pytest.approx(0.25)
        assert cfg.nms_iou_threshold == pytest.approx(0.5)
        assert cfg.top_k == 24
        assert cfg.rtsp_urls == [
            "rtsp://192.168.0.235:8554/src1",
            "rtsp://192.168.0.235:8554/src2",
            "rtsp://192.168.0.235:8554/src3",
        ]

    def test_load_app_config_rejects_missing_streams(self, tmp_path: Path):
        from utils.config import load_app_config

        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            """
model: assets/models/yolo_v8m_mpk.tar.gz
input: {}
inference:
  detection_threshold: 0.25
  nms_iou_threshold: 0.5
  top_k: 24
tracking: {}
output:
  insight:
    host: 127.0.0.1
    video_port_base: 9000
    metadata_port_base: 9100
  debug_dir: null
  save_every: 0
""".strip(),
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="streams"):
            load_app_config(config_path)

    def test_load_app_config_defaults_to_clean_video_mode(self, tmp_path: Path):
        from utils.config import VideoMode, load_app_config, metadata_output_enabled

        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            """
model: assets/models/yolo_v8m_mpk.tar.gz
input: {}
inference:
  detection_threshold: 0.25
  nms_iou_threshold: 0.5
  top_k: 24
tracking: {}
output:
  insight:
    host: 127.0.0.1
    video_port_base: 9000
    metadata_port_base: 9100
  debug_dir: null
  save_every: 0
streams:
  - rtsp://127.0.0.1:8554/src1
""".strip(),
            encoding="utf-8",
        )

        cfg = load_app_config(config_path)

        assert cfg.video_mode is VideoMode.CLEAN
        assert metadata_output_enabled(cfg) is True

    def test_load_app_config_rejects_invalid_video_mode(self, tmp_path: Path):
        from utils.config import load_app_config

        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            """
model: assets/models/yolo_v8m_mpk.tar.gz
input: {}
inference: {}
tracking: {}
output:
  insight:
    host: 127.0.0.1
    video_port_base: 9000
    metadata_port_base: 9100
  video_mode: invalid
  debug_dir: null
  save_every: 0
streams:
  - rtsp://127.0.0.1:8554/src1
""".strip(),
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="video_mode"):
            load_app_config(config_path)

    def test_common_config_yaml_uses_supported_shape(self):
        from utils.config import VideoMode, load_app_config

        config_path = EXAMPLE_DIR / "common" / "config.yaml"
        cfg = load_app_config(config_path)

        assert cfg.model
        assert cfg.insight_video_port_base > 0
        assert cfg.insight_metadata_port_base > 0
        assert cfg.save_every >= 0
        assert cfg.video_mode in {VideoMode.CLEAN, VideoMode.ANNOTATED}
        assert isinstance(cfg.rtsp_urls, list)


class TestImageUtils:
    def test_cpu_quanttess_input_letterboxes_rgb_frame_into_rgb_fp32_tensor(self):
        from utils.image_utils import build_cpu_quanttess_preproc_state, cpu_quanttess_input
        from utils.pipeline import QuantTessCpuPreproc, RuntimeModules

        class FakeImage:
            def __init__(
                self,
                *,
                shape,
                channels="rgb",
                dtype="uint8",
                source=None,
                scale=None,
            ):
                self.shape = shape
                self.channels = channels
                self.dtype = dtype
                self.source = source
                self.scale = scale

            def astype(self, dtype):
                return FakeImage(
                    shape=self.shape,
                    channels=self.channels,
                    dtype=dtype,
                    source=self.source or self,
                    scale=self.scale,
                )

        class FakeCanvasRegion:
            def __init__(self, canvas, key):
                self.canvas = canvas
                self.key = key

        class FakeCanvas:
            def __init__(self, shape, dtype):
                self.shape = shape
                self.dtype = dtype
                self.assignments = []
                self.fill_calls = []

            def __getitem__(self, key):
                return FakeCanvasRegion(self, key)

            def fill(self, value):
                self.fill_calls.append(value)

        class FakeCV2:
            INTER_LINEAR = "linear"

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
            def multiply(image, scale, out=None, casting=None):
                assert casting == "unsafe"
                assert isinstance(out, FakeCanvasRegion)
                out.canvas.assignments.append(
                    (
                        out.key,
                        FakeImage(
                            shape=image.shape,
                            channels=image.channels,
                            dtype=FakeNp.float32,
                            source=image.source or image,
                            scale=scale,
                        ),
                    )
                )
                return out.canvas

        runtime = RuntimeModules(cv2=FakeCV2(), np=FakeNp(), pyneat=None)
        contract = QuantTessCpuPreproc(width=8, height=8, aspect_ratio=True, padding_type="CENTER")
        frame_rgb = FakeImage(shape=(2, 4, 3), channels="rgb")
        state = build_cpu_quanttess_preproc_state(runtime, contract, src_width=4, src_height=2)

        quant_input = cpu_quanttess_input(runtime, frame_rgb, state)

        assert quant_input.shape == (8, 8, 3)
        assert quant_input.dtype == runtime.np.float32
        key, value = quant_input.assignments[0]
        assert key == (slice(2, 6, None), slice(0, 8, None))
        assert value.shape == (4, 8, 3)
        assert value.channels == "rgb"
        assert value.dtype == runtime.np.float32
        assert value.scale == pytest.approx(1.0 / 255.0)
        assert quant_input.fill_calls == [0.0]

    def test_cpu_quanttess_input_reuses_preallocated_output_buffer_when_state_is_supplied(self):
        from utils.image_utils import build_cpu_quanttess_preproc_state, cpu_quanttess_input
        from utils.pipeline import QuantTessCpuPreproc, RuntimeModules

        class FakeImage:
            def __init__(self, *, shape, channels="rgb", dtype="uint8", source=None, scale=None):
                self.shape = shape
                self.channels = channels
                self.dtype = dtype
                self.source = source
                self.scale = scale

        class FakeCanvasRegion:
            def __init__(self, canvas, key):
                self.canvas = canvas
                self.key = key

        class FakeCanvas:
            def __init__(self, shape, dtype):
                self.shape = shape
                self.dtype = dtype
                self.assignments = []
                self.fill_calls = []

            def __getitem__(self, key):
                return FakeCanvasRegion(self, key)

            def fill(self, value):
                self.fill_calls.append(value)

        class FakeCV2:
            INTER_LINEAR = "linear"

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
            def multiply(image, scale, out=None, casting=None):
                assert casting == "unsafe"
                assert isinstance(out, FakeCanvasRegion)
                out.canvas.assignments.append(
                    (
                        out.key,
                        FakeImage(
                            shape=image.shape,
                            channels=image.channels,
                            dtype=FakeNp.float32,
                            source=image.source or image,
                            scale=scale,
                        ),
                    )
                )
                return out.canvas

        runtime = RuntimeModules(cv2=FakeCV2(), np=FakeNp(), pyneat=None)
        contract = QuantTessCpuPreproc(width=8, height=8, aspect_ratio=True, padding_type="CENTER")
        state = build_cpu_quanttess_preproc_state(runtime, contract, src_width=4, src_height=2)

        first = cpu_quanttess_input(
            runtime,
            FakeImage(shape=(2, 4, 3), channels="rgb"),
            state,
        )
        second = cpu_quanttess_input(
            runtime,
            FakeImage(shape=(2, 4, 3), channels="rgb"),
            state,
        )

        assert first is state.quant_input
        assert second is state.quant_input
        assert state.quant_input.fill_calls == [0.0, 0.0]
        assert len(state.quant_input.assignments) == 2

    def test_save_overlay_frame_converts_rgb_frame_to_bgr_before_imwrite(self, tmp_path: Path):
        from utils.image_utils import save_overlay_frame

        class FakeFrame:
            def __init__(self, channels):
                self.channels = channels

        class FakeCV2:
            COLOR_RGB2BGR = "rgb2bgr"

            def __init__(self):
                self.calls = []

            def cvtColor(self, frame, code):
                assert code == self.COLOR_RGB2BGR
                converted = FakeFrame("bgr")
                self.calls.append(("cvtColor", frame.channels, code, converted.channels))
                return converted

            def imwrite(self, path, frame):
                self.calls.append(("imwrite", Path(path).name, frame.channels))
                return True

        cv2 = FakeCV2()
        runtime = SimpleNamespace(cv2=cv2)

        saved = save_overlay_frame(
            runtime,
            tmp_path / "debug",
            stream_index=1,
            frame_index=2,
            frame=FakeFrame("rgb"),
            save_every=1,
        )

        assert saved is True
        assert cv2.calls == [
            ("cvtColor", "rgb", "rgb2bgr", "bgr"),
            ("imwrite", "frame_000002.jpg", "bgr"),
        ]


class TestMainEntrypoint:
    def test_help_runs(self):
        result = subprocess.run(
            [sys.executable, str(MAIN_PY), "--help"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=str(EXAMPLE_DIR),
        )
        assert result.returncode == 0
        assert "--config" in result.stdout

    def test_missing_config_file_fails_cleanly(self):
        result = subprocess.run(
            [sys.executable, str(MAIN_PY), "--config", "does-not-exist.yaml"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=str(EXAMPLE_DIR),
        )
        assert result.returncode != 0
        assert "config" in result.stderr.lower()


class TestPipelineBuilders:
    def test_build_detection_run_uses_model_preprocess_mla_and_boxdecode_contract(self):
        from utils.config import AppConfig
        from utils.pipeline import RtspProbe, RuntimeModules, build_detection_run

        class FakeGraph:
            def __init__(self):
                self.added = []
                self.build_calls = []

            def add(self, node):
                self.added.append(node)

            def build(self, *args, **kwargs):
                self.build_calls.append((args, kwargs))
                return "fake-run"

        class FakeModelOptions:
            def __init__(self):
                self.preprocess = SimpleNamespace(color_convert=SimpleNamespace())

        class FakeRunOptions:
            pass

        class FakeTensor:
            @staticmethod
            def from_numpy(array, **kwargs):
                return ("tensor", array, kwargs)

        class FakeModel:
            def __init__(self, path, options):
                self.path = path
                self.options = options

            def input_appsrc_options(self, copy):
                return SimpleNamespace(copy=copy)

            def preprocess(self):
                return ("preprocess", self)

        class FakePyneat:
            class InputKind:
                Image = "image"

            class PreprocessColorFormat:
                RGB = "rgb"

            class BoxDecodeType:
                YoloV8 = "yolov8"

            class PixelFormat:
                RGB = "rgb"

            class TensorMemory:
                EV74 = "ev74"

            class RunPreset:
                Realtime = "realtime"

            class OverflowPolicy:
                KeepLatest = "keep-latest"

            class OutputMemory:
                Owned = "owned"

            class RunMode:
                Async = "async"

            class PayloadType:
                Image = "image"

            Tensor = FakeTensor

            def __init__(self):
                self.last_graph = None
                self.nodes = SimpleNamespace(
                    input=lambda opt=None: ("input", opt),
                    sima_box_decode=lambda model, **kwargs: ("boxdecode", model, kwargs),
                    output=lambda: ("output",),
                )
                self.groups = SimpleNamespace(mla=lambda model: ("mla", model))
                self.models = []

            def Graph(self, name):
                self.last_graph = FakeGraph()
                self.last_graph.name = name
                return self.last_graph

            def ModelOptions(self):
                return FakeModelOptions()

            def Model(self, path, options):
                model = FakeModel(path, options)
                self.models.append(model)
                return model

            def RunOptions(self):
                return FakeRunOptions()

        class FakeNp:
            uint8 = "uint8"

            @staticmethod
            def zeros(shape, dtype=None):
                return {"shape": shape, "dtype": dtype}

        cfg = AppConfig(
            model="model.tar.gz",
            rtsp_urls=["rtsp://camera-0"],
            output_dir=None,
            frames=0,
            insight_host="127.0.0.1",
            insight_video_port_base=9000,
            insight_metadata_port_base=9100,
            fps=0,
            bitrate_kbps=2500,
            save_every=30,
            profile=False,
            person_class_id=0,
            detection_threshold=0.25,
            nms_iou_threshold=0.5,
            top_k=24,
            tracker_iou_threshold=0.3,
            tracker_max_missing=15,
            latency_ms=200,
            tcp=False,
        )
        probe = RtspProbe(width=1280, height=720, fps=16)
        runtime = RuntimeModules(cv2=None, np=FakeNp(), pyneat=FakePyneat())

        built = build_detection_run(runtime, cfg, probe)

        added_kinds = [node[0] for node in runtime.pyneat.last_graph.added]
        assert built.run == "fake-run"
        assert added_kinds == ["input", "preprocess", "mla", "boxdecode", "output"]
        assert runtime.pyneat.models[0].options.preprocess.kind == runtime.pyneat.InputKind.Image
        assert runtime.pyneat.last_graph.build_calls

    def test_build_insight_metadata_output_uses_per_stream_ports(self):
        from utils.config import AppConfig
        from utils.pipeline import RuntimeModules, build_insight_metadata_output

        class FakeMetadataSenderOptions:
            def __init__(self):
                self.host = ""
                self.channel = 0
                self.metadata_port_base = 0

        class FakeMetadataSender:
            def __init__(self, opt):
                self.opt = opt

        class FakePyneat:
            def MetadataSenderOptions(self):
                return FakeMetadataSenderOptions()

            def MetadataSender(self, opt):
                return FakeMetadataSender(opt)

        cfg = AppConfig(
            model="model.tar.gz",
            rtsp_urls=["rtsp://camera-0"],
            output_dir=None,
            frames=0,
            insight_host="192.168.0.107",
            insight_video_port_base=9000,
            insight_metadata_port_base=9100,
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
        runtime = RuntimeModules(cv2=None, np=None, pyneat=FakePyneat())

        sender = build_insight_metadata_output(runtime, cfg, stream_index=2)

        assert sender.opt.host == cfg.insight_host
        assert sender.opt.channel == 2
        assert sender.opt.metadata_port_base == cfg.insight_metadata_port_base


class TestSampleUtils:
    def test_make_insight_tracking_detection_uses_track_id(self):
        from utils.sample_utils import make_insight_tracking_detection
        from utils.tracker import TrackedDetection

        tracks = make_insight_tracking_detection(
            [
                TrackedDetection(
                    track_id=42,
                    x1=1.0,
                    y1=2.0,
                    x2=11.0,
                    y2=22.0,
                    score=0.9,
                    class_id=0,
                )
            ],
        )

        assert len(tracks) == 1
        assert tracks[0]["id"] == "42"
        assert tracks[0]["label"] == "person"
        assert tracks[0]["confidence"] == pytest.approx(0.9)
        assert tracks[0]["bbox"] == [1.0, 2.0, 10.0, 20.0]


class TestTracker:
    def test_reuses_track_id_for_nearby_detection(self):
        from utils.tracker import PeopleTracker

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

    def test_drops_track_after_missing_budget(self):
        from utils.tracker import PeopleTracker

        tracker = PeopleTracker(iou_threshold=0.3, max_missing_frames=1)

        tracker.update(
            [{"x1": 10.0, "y1": 10.0, "x2": 50.0, "y2": 80.0, "score": 0.9, "class_id": 0}],
            frame_index=0,
        )
        tracker.update([], frame_index=1)
        tracker.update([], frame_index=2)

        assert tracker.active_track_count() == 0


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

            def push_tensors(self, tensors):
                self.calls.append(tensors)
                return True

        class FakeTensor:
            @staticmethod
            def from_numpy(frame, **kwargs):
                return SimpleNamespace(frame=frame, kwargs=kwargs)

        class FakeMetadataSender:
            def __init__(self):
                self.calls = []

            def send_metadata(self, metadata_type, data_json, timestamp_ms, frame_id):
                self.calls.append((metadata_type, data_json, timestamp_ms, frame_id))
                return True

        cfg = AppConfig(
            model="model.tar.gz",
            rtsp_urls=["rtsp://camera-0"],
            output_dir=None,
            frames=1,
            insight_host="127.0.0.1",
            insight_video_port_base=9000,
            insight_metadata_port_base=9100,
            fps=0,
            bitrate_kbps=2500,
            save_every=30,
            profile=False,
            person_class_id=0,
            detection_threshold=0.25,
            nms_iou_threshold=0.5,
            top_k=24,
            tracker_iou_threshold=0.3,
            tracker_max_missing=15,
            latency_ms=200,
            tcp=False,
        )
        video_run = FakeVideoRun()
        sender = FakeMetadataSender()
        frame = SimpleNamespace(shape=(24, 32, 3), label="clean-frame")
        stream = SimpleNamespace(
            index=0,
            runtime=SimpleNamespace(
                pyneat=SimpleNamespace(
                    PixelFormat=SimpleNamespace(RGB="rgb"),
                    TensorMemory=SimpleNamespace(EV74="ev74"),
                    Tensor=FakeTensor,
                )
            ),
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
            video=SimpleNamespace(run=video_run),
            metadata_sender=sender,
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

        monkeypatch.setattr(
            "utils.workers.parse_bbox_payload",
            lambda payload, w, h: [{"class_id": 0, "x1": 1.0, "y1": 2.0, "x2": 11.0, "y2": 22.0, "score": 0.9}],
        )
        monkeypatch.setattr("utils.workers.filter_person_detections", lambda boxes, person_class_id: boxes)
        sentinel_tracks = [{"id": "5"}]
        monkeypatch.setattr(
            "utils.workers.make_insight_tracking_detection",
            lambda tracked: sentinel_tracks,
        )
        monkeypatch.setattr("utils.workers.save_overlay_frame", lambda *args, **kwargs: False)
        monkeypatch.setattr(
            "utils.workers.draw_tracked_people",
            lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("overlay should not be rendered")),
        )

        publish_thread(stream, cfg, result_q, stop_event)

        assert len(video_run.calls) == 1
        video_tensor = video_run.calls[0][0]
        assert video_tensor.frame is frame
        assert video_tensor.kwargs == {
            "copy": True,
            "image_format": "rgb",
            "memory": "ev74",
        }
        assert len(sender.calls) == 1
        metadata_type, data_json, _, frame_id = sender.calls[0]
        assert metadata_type == "tracking"
        assert frame_id == "0"
        assert data_json == '{"tracks":[{"id":"5"}]}'

    def test_publish_thread_annotated_video_suppresses_metadata(self, monkeypatch):
        from utils.config import AppConfig, VideoMode
        from utils.tracker import TrackedDetection
        from utils.workers import ResultPacket, StreamMetrics, publish_thread

        class FakeVideoRun:
            def __init__(self):
                self.calls = []

            def push_tensors(self, tensors):
                self.calls.append(tensors)
                return True

        class FakeTensor:
            @staticmethod
            def from_numpy(frame, **kwargs):
                return SimpleNamespace(frame=frame, kwargs=kwargs)

        class FakeMetadataSender:
            def __init__(self):
                self.calls = []

            def send_metadata(self, *args):
                self.calls.append(args)
                return True

        cfg = AppConfig(
            model="model.tar.gz",
            rtsp_urls=["rtsp://camera-0"],
            output_dir=None,
            frames=1,
            insight_host="127.0.0.1",
            insight_video_port_base=9000,
            insight_metadata_port_base=9100,
            fps=0,
            bitrate_kbps=2500,
            save_every=0,
            profile=False,
            person_class_id=0,
            detection_threshold=0.25,
            nms_iou_threshold=0.5,
            top_k=24,
            tracker_iou_threshold=0.3,
            tracker_max_missing=15,
            latency_ms=200,
            tcp=False,
            video_mode=VideoMode.ANNOTATED,
        )
        video_run = FakeVideoRun()
        sender = FakeMetadataSender()
        frame = SimpleNamespace(
            shape=(24, 32, 3),
            label="clean-frame",
            copy=lambda: SimpleNamespace(shape=(24, 32, 3), label="copy-frame"),
        )
        stream = SimpleNamespace(
            index=0,
            runtime=SimpleNamespace(
                pyneat=SimpleNamespace(
                    PixelFormat=SimpleNamespace(RGB="rgb"),
                    TensorMemory=SimpleNamespace(EV74="ev74"),
                    Tensor=FakeTensor,
                )
            ),
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
            video=SimpleNamespace(run=video_run),
            metadata_sender=sender,
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

        monkeypatch.setattr(
            "utils.workers.parse_bbox_payload",
            lambda payload, w, h: [{"class_id": 0, "x1": 1.0, "y1": 2.0, "x2": 11.0, "y2": 22.0, "score": 0.9}],
        )
        monkeypatch.setattr("utils.workers.filter_person_detections", lambda boxes, person_class_id: boxes)
        monkeypatch.setattr("utils.workers.save_overlay_frame", lambda *args, **kwargs: False)
        monkeypatch.setattr(
            "utils.workers.draw_tracked_people",
            lambda runtime, copied_frame, tracked: SimpleNamespace(shape=copied_frame.shape, label="overlay-frame"),
        )

        publish_thread(stream, cfg, result_q, stop_event)

        assert video_run.calls[0][0].frame.label == "overlay-frame"
        assert video_run.calls[0][0].kwargs == {
            "copy": True,
            "image_format": "rgb",
            "memory": "ev74",
        }
        assert sender.calls == []

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
            insight_host="127.0.0.1",
            insight_video_port_base=9000,
            insight_metadata_port_base=9100,
            fps=10,
            bitrate_kbps=2500,
            save_every=0,
            profile=False,
            person_class_id=0,
            detection_threshold=0.25,
            nms_iou_threshold=0.5,
            top_k=24,
            tracker_iou_threshold=0.3,
            tracker_max_missing=15,
            latency_ms=200,
            tcp=False,
        )
        stream = SimpleNamespace(
            index=0,
            source=SimpleNamespace(run=FakeSourceRun()),
            runtime=SimpleNamespace(),
            metrics=StreamMetrics(),
            error=None,
        )
        frame_q: queue.Queue = queue.Queue(maxsize=4)
        stop_event = threading.Event()

        producer_thread(stream, cfg, frame_q, stop_event)

        assert stream.source.run.calls == 3
        assert frame_q.qsize() == 2
        first = frame_q.get_nowait()
        second = frame_q.get_nowait()
        assert first.frame == {"frame": "sample-1"}
        assert second.frame == {"frame": "sample-3"}
        assert first.frame_index == 0
        assert second.frame_index == 1

    def test_infer_thread_exits_when_frame_queue_is_closed_and_drained(self):
        from utils.config import AppConfig
        from utils.workers import KeepLatestQueue, StreamMetrics, infer_thread

        cfg = AppConfig(
            model="model.tar.gz",
            rtsp_urls=["rtsp://camera-0"],
            output_dir=None,
            frames=10,
            insight_host="127.0.0.1",
            insight_video_port_base=9000,
            insight_metadata_port_base=9100,
            fps=0,
            bitrate_kbps=2500,
            save_every=0,
            profile=False,
            person_class_id=0,
            detection_threshold=0.25,
            nms_iou_threshold=0.5,
            top_k=24,
            tracker_iou_threshold=0.3,
            tracker_max_missing=15,
            latency_ms=200,
            tcp=False,
        )
        frame_q = KeepLatestQueue(maxsize=4)
        result_q = KeepLatestQueue(maxsize=4)
        frame_q.close()
        stream = SimpleNamespace(
            index=0,
            runtime=SimpleNamespace(),
            detect=SimpleNamespace(run=SimpleNamespace()),
            quant_preproc_state=SimpleNamespace(),
            metrics=StreamMetrics(),
            error=None,
        )
        stop_event = threading.Event()

        infer_thread(stream, cfg, frame_q, result_q, stop_event)

        assert stream.error is None
        assert result_q.is_closed()
        assert result_q.empty()

    def test_publish_thread_exits_when_result_queue_is_closed_and_drained(self):
        from utils.config import AppConfig
        from utils.workers import KeepLatestQueue, StreamMetrics, publish_thread

        cfg = AppConfig(
            model="model.tar.gz",
            rtsp_urls=["rtsp://camera-0"],
            output_dir=None,
            frames=10,
            insight_host="127.0.0.1",
            insight_video_port_base=9000,
            insight_metadata_port_base=9100,
            fps=0,
            bitrate_kbps=2500,
            save_every=0,
            profile=False,
            person_class_id=0,
            detection_threshold=0.25,
            nms_iou_threshold=0.5,
            top_k=24,
            tracker_iou_threshold=0.3,
            tracker_max_missing=15,
            latency_ms=200,
            tcp=False,
        )
        result_q = KeepLatestQueue(maxsize=4)
        result_q.close()
        stream = SimpleNamespace(
            index=0,
            runtime=SimpleNamespace(),
            tracker=SimpleNamespace(),
            video=SimpleNamespace(run=SimpleNamespace()),
            metadata_sender=SimpleNamespace(),
            metrics=StreamMetrics(),
            error=None,
        )
        stop_event = threading.Event()

        publish_thread(stream, cfg, result_q, stop_event)

        assert stream.error is None
