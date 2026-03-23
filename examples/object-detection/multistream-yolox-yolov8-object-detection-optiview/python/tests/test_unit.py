"""Unit tests for multistream-yolox-yolov8-object-detection-optiview (Python)."""

from __future__ import annotations

import queue
import subprocess
import sys
from pathlib import Path
import struct
from types import SimpleNamespace

import pytest


EXAMPLE_DIR = Path(__file__).resolve().parent.parent.parent
PYTHON_DIR = EXAMPLE_DIR / "python"
MAIN_PY = EXAMPLE_DIR / "python" / "main.py"
DEFAULT_CONFIG = EXAMPLE_DIR / "common" / "config.yaml"

if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))


@pytest.mark.unit
class TestArgParsing:
    def test_help_mentions_config(self) -> None:
        result = subprocess.run(
            [sys.executable, str(MAIN_PY), "--help"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=str(EXAMPLE_DIR),
        )
        assert result.returncode == 0
        assert "--config" in result.stdout

    def test_missing_config_path_fails(self, tmp_path: Path) -> None:
        missing = tmp_path / "missing.yaml"
        result = subprocess.run(
            [sys.executable, str(MAIN_PY), "--config", str(missing)],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=str(EXAMPLE_DIR),
        )
        assert result.returncode == 2
        assert "config file not found" in result.stderr.lower()

    def test_default_config_smoke_runs(self) -> None:
        assert DEFAULT_CONFIG.exists()
        result = subprocess.run(
            [sys.executable, str(MAIN_PY), "--validate-config-only"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=str(EXAMPLE_DIR),
        )
        assert result.returncode == 0
        assert "config validated" in result.stdout.lower()


@pytest.mark.unit
class TestConfigLoading:
    def test_load_app_config_parses_runtime_worker_count(self, tmp_path: Path) -> None:
        from utils.config import load_app_config

        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            """
model:
  path: assets/models/yolox_s_mpk.tar.gz
  family: auto
streams:
  - rtsp://127.0.0.1:8554/src1
  - rtsp://127.0.0.1:8554/src2
input:
  tcp: true
  latency_ms: 120
runtime:
  worker_count: 4
  mailbox_depth: 1
  profile: true
inference:
  frames: 0
  fps: 0
  min_score: 0.25
  nms_iou: 0.45
  max_detections: 100
output:
  optiview:
    host: 127.0.0.1
    video_port_base: 9000
    json_port_base: 9100
    json_offset_ms: 12.5
  video_mode: clean
  debug_dir: null
  save_every: 0
""".strip(),
            encoding="utf-8",
        )

        cfg = load_app_config(config_path)

        assert cfg.model.path == "assets/models/yolox_s_mpk.tar.gz"
        assert cfg.model.family == "auto"
        assert cfg.worker_count == 4
        assert cfg.mailbox_depth == 1
        assert cfg.profile is True
        assert cfg.optiview_json_offset_ms == pytest.approx(12.5)
        assert cfg.rtsp_urls == [
            "rtsp://127.0.0.1:8554/src1",
            "rtsp://127.0.0.1:8554/src2",
        ]
        assert cfg.video_mode == "clean"

    def test_load_app_config_rejects_invalid_worker_count(self, tmp_path: Path) -> None:
        from utils.config import load_app_config

        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            """
model:
  path: assets/models/yolo_v8m_mpk.tar.gz
  family: auto
streams:
  - rtsp://127.0.0.1:8554/src1
runtime:
  worker_count: 0
output:
  optiview:
    host: 127.0.0.1
""".strip(),
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="worker_count"):
            load_app_config(config_path)

    def test_load_app_config_rejects_invalid_video_mode(self, tmp_path: Path) -> None:
        from utils.config import load_app_config

        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            """
model:
  path: assets/models/yolo_v8m_mpk.tar.gz
  family: auto
streams:
  - rtsp://127.0.0.1:8554/src1
runtime:
  worker_count: 2
output:
  optiview:
    host: 127.0.0.1
  video_mode: purple
""".strip(),
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="video_mode"):
            load_app_config(config_path)

    def test_load_app_config_rejects_empty_streams(self, tmp_path: Path) -> None:
        from utils.config import load_app_config

        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            """
model:
  path: assets/models/yolo_v8m_mpk.tar.gz
  family: auto
streams: []
runtime:
  worker_count: 2
output:
  optiview:
    host: 127.0.0.1
""".strip(),
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="streams"):
            load_app_config(config_path)


@pytest.mark.unit
class TestModelFamily:
    def test_resolve_model_family_auto_for_yolox(self) -> None:
        from utils.model_family import resolve_model_family

        assert resolve_model_family("assets/models/yolox_s_mpk.tar.gz", family_hint="auto") == "yolox"

    def test_resolve_model_family_auto_for_yolov8(self) -> None:
        from utils.model_family import resolve_model_family

        assert (
            resolve_model_family("assets/models/yolo_v8m_mpk.tar.gz", family_hint="auto")
            == "yolov8"
        )


@pytest.mark.unit
class TestSampleUtils:
    def test_parse_bbox_payload_normalizes_yolov8_boxes(self) -> None:
        from utils.sample_utils import parse_bbox_payload

        payload = struct.pack(
            "<Iiiiifi",
            1,
            -10,
            20,
            140,
            50,
            0.9,
            3,
        )

        boxes = parse_bbox_payload(payload, img_w=100, img_h=80)

        assert len(boxes) == 1
        assert boxes[0]["x1"] == 0.0
        assert boxes[0]["y1"] == 20.0
        assert boxes[0]["x2"] == 100.0
        assert boxes[0]["y2"] == 70.0
        assert boxes[0]["score"] == pytest.approx(0.9)
        assert boxes[0]["class_id"] == 3

    def test_require_detector_output_kind_accepts_yolox_detessdequant_bundle(self) -> None:
        from utils.sample_utils import require_detector_output_kind

        class FakeTensor:
            def copy_payload_bytes(self) -> bytes:
                return b""

        class FakeSample:
            def __init__(
                self,
                *,
                kind: str,
                tensor=None,
                fields=None,
                payload_tag: str = "",
                format: str = "",
            ) -> None:
                self.kind = kind
                self.tensor = tensor
                self.fields = list(fields or [])
                self.payload_tag = payload_tag
                self.format = format

        pyneat = SimpleNamespace(
            SampleKind=SimpleNamespace(
                Tensor="tensor",
                Bundle="bundle",
                Unknown="unknown",
            )
        )
        sample = FakeSample(
            kind=pyneat.SampleKind.Bundle,
            payload_tag="DETESSDEQUANT",
            fields=[
                FakeSample(
                    kind=pyneat.SampleKind.Tensor,
                    tensor=FakeTensor(),
                    payload_tag="DETESSDEQUANT",
                )
            ],
        )

        assert require_detector_output_kind(pyneat, "yolox", sample) == "DETESSDEQUANT"

    def test_require_detector_output_kind_rejects_unsupported_sample_kind(self) -> None:
        from utils.sample_utils import require_detector_output_kind

        class FakeSample:
            def __init__(self, *, kind: str, payload_tag: str = "", fields=None) -> None:
                self.kind = kind
                self.tensor = None
                self.fields = list(fields or [])
                self.payload_tag = payload_tag
                self.format = ""

        pyneat = SimpleNamespace(
            SampleKind=SimpleNamespace(
                Tensor="tensor",
                Bundle="bundle",
                Unknown="unknown",
            )
        )
        sample = FakeSample(kind=pyneat.SampleKind.Unknown)

        with pytest.raises(RuntimeError, match="unsupported detector output"):
            require_detector_output_kind(pyneat, "yolov8", sample)

    def test_require_detector_output_kind_rejects_unsupported_family(self) -> None:
        from utils.sample_utils import require_detector_output_kind

        class FakeSample:
            def __init__(self, *, kind: str, payload_tag: str = "") -> None:
                self.kind = kind
                self.tensor = None
                self.fields = []
                self.payload_tag = payload_tag
                self.format = payload_tag

        pyneat = SimpleNamespace(
            SampleKind=SimpleNamespace(
                Tensor="tensor",
                Bundle="bundle",
                Unknown="unknown",
            )
        )
        sample = FakeSample(kind=pyneat.SampleKind.Tensor, payload_tag="BBOX")

        with pytest.raises(ValueError, match="unsupported model family"):
            require_detector_output_kind(pyneat, "yolo11", sample)

    def test_make_optiview_detection_payload_builds_objects_and_labels(self) -> None:
        from utils.sample_utils import make_optiview_detection_payload

        class FakeOptiViewObject:
            def __init__(self) -> None:
                self.x = 0
                self.y = 0
                self.w = 0
                self.h = 0
                self.score = 0.0
                self.class_id = -1

        objects, labels = make_optiview_detection_payload(
            SimpleNamespace(OptiViewObject=FakeOptiViewObject),
            [
                {
                    "x1": -5.0,
                    "y1": 10.0,
                    "x2": 35.0,
                    "y2": 50.0,
                    "score": 0.8,
                    "class_id": 2,
                }
            ],
            img_w=30,
            img_h=40,
            class_labels=["person", "bicycle", "car"],
        )

        assert len(objects) == 1
        assert objects[0].x == 0
        assert objects[0].y == 10
        assert objects[0].w == 30
        assert objects[0].h == 30
        assert objects[0].score == pytest.approx(0.8)
        assert objects[0].class_id == 2
        assert labels == ["person", "bicycle", "car"]

    def test_optiview_frame_id_prefers_detector_sample_frame_id(self) -> None:
        from utils.sample_utils import optiview_frame_id

        assert optiview_frame_id(SimpleNamespace(frame_id=42), fallback_frame_index=7) == "42"

    def test_optiview_frame_id_falls_back_to_packet_index(self) -> None:
        from utils.sample_utils import optiview_frame_id

        assert optiview_frame_id(SimpleNamespace(frame_id=-1), fallback_frame_index=7) == "7"

    def test_optiview_timestamp_ms_applies_publish_offset(self) -> None:
        from utils.sample_utils import optiview_timestamp_ms

        assert optiview_timestamp_ms(publish_time_s=1.234, offset_ms=25.0) == 1259


@pytest.mark.unit
class TestPipelineBuilders:
    def test_detector_stage_names_for_yolov8(self) -> None:
        from utils.pipeline import detector_stage_names

        assert detector_stage_names("yolov8") == (
            "input",
            "quant_tess",
            "mla",
            "sima_box_decode",
            "output",
        )

    def test_detector_stage_names_for_yolox(self) -> None:
        from utils.pipeline import detector_stage_names

        assert detector_stage_names("yolox") == (
            "input",
            "quant_tess",
            "mla",
            "detess_dequant",
            "output",
        )


@pytest.mark.unit
class TestWorkers:
    def test_collect_detector_runtime_keys_deduplicates_same_geometry(self) -> None:
        from utils.workers import collect_detector_runtime_keys

        streams = [
            SimpleNamespace(family="yolov8", probe=SimpleNamespace(width=640, height=480)),
            SimpleNamespace(family="yolov8", probe=SimpleNamespace(width=640, height=480)),
            SimpleNamespace(family="yolov8", probe=SimpleNamespace(width=1280, height=720)),
        ]

        assert collect_detector_runtime_keys(streams) == [
            ("yolov8", 640, 480),
            ("yolov8", 1280, 720),
        ]

    def test_latest_frame_mailbox_deduplicates_ready_notifications_and_requeues_after_completion(
        self,
    ) -> None:
        from utils.workers import LatestFrameMailbox

        ready_q: queue.Queue[int] = queue.Queue()
        mailbox = LatestFrameMailbox(stream_index=7, capacity=1)

        assert mailbox.push("frame-0", ready_q) == 0
        assert mailbox.push("frame-1", ready_q) == 1
        assert ready_q.get_nowait() == 7
        with pytest.raises(queue.Empty):
            ready_q.get_nowait()

        assert mailbox.take_for_processing() == "frame-1"
        assert mailbox.push("frame-2", ready_q) == 0
        with pytest.raises(queue.Empty):
            ready_q.get_nowait()

        mailbox.complete(ready_q)
        assert ready_q.get_nowait() == 7
