"""E2E tests for multi-stream-obb-detector (Python)."""

from __future__ import annotations

import importlib.util
import json
import math
import os
from pathlib import Path
import sys

import pytest

from tests.utils.metadata_json_listener import MetadataJsonListener
from tests.utils.output_assertions import assert_streamed_frames_are_usable
from tests.utils.process_assertions import assert_exited_cleanly


EXAMPLE_DIR = Path(__file__).resolve().parent.parent.parent
MAIN_PY = EXAMPLE_DIR / "src" / "python" / "main.py"
LABELS_FILE = EXAMPLE_DIR / "src" / "common" / "dota_label.txt"
E2E_INSIGHT_HOST = "127.0.0.1"


def _runtime_deps_ready() -> bool:
    return all(importlib.util.find_spec(name) is not None for name in ("cv2", "numpy", "pyneat"))


def _env_int_or_default(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw else default


def assert_oriented_objects(metadata, labels: set[str]) -> None:
    """Fail unless every published object is a well-formed oriented box.

    Whether any object is published depends on the stream: the model detects
    top-down aerial objects, and the CI streams are street scenes. An empty run
    is therefore not a failure here; the unit tests cover the object values.
    """
    for message in metadata.messages:
        # Insight wraps the array: {"data": {"objects": [...]}, "frame_id": ..., "type": ...}
        objects = (json.loads(message.payload).get("data") or {}).get("objects")
        where = f"port {message.port} frame {message.frame_id}"
        assert isinstance(objects, list), f"{where}: 'data.objects' must be a list"
        for obj in objects:
            assert set(obj) == {"id", "label", "confidence", "bbox", "oriented_bbox"}, (
                f"{where}: unexpected keys {sorted(obj)}"
            )
            assert obj["label"] in labels, f"{where}: label {obj['label']!r} is not a DOTA class"
            assert 0.0 <= obj["confidence"] <= 1.0, f"{where}: confidence out of range"
            cx, cy, width, height, angle = obj["oriented_bbox"]
            assert width > 0 and height > 0 and math.isfinite(angle), (
                f"{where}: degenerate oriented box {obj['oriented_bbox']}"
            )
            x, y, extent_w, extent_h = obj["bbox"]
            # Each published value is rounded to 0.1 px, so the centers can differ by 0.125 px.
            assert math.isclose(x + extent_w / 2, cx, abs_tol=0.15) and math.isclose(
                y + extent_h / 2, cy, abs_tol=0.15
            ), f"{where}: bbox {obj['bbox']} is not centered on the oriented box"


@pytest.mark.e2e
class TestE2E:
    @pytest.mark.parametrize(
        ("codec", "urls_fixture"),
        [("h264", "rtsp_h264_urls"), ("h265", "rtsp_h265_urls")],
    )
    def test_multi_stream_insight_and_save_pipeline(
        self,
        request,
        codec,
        urls_fixture,
        e2e_model_path,
        tmp_output_dir,
        test_timeout_ms,
        skip_unless_e2e_ready,
        e2e_config_writer,
        e2e_config_section,
        run_until_output_files,
    ):
        rtsp_urls = request.getfixturevalue(urls_fixture)
        skip_unless_e2e_ready(
            _runtime_deps_ready(),
            "python runtime dependencies (cv2, numpy, pyneat) are not available",
        )
        skip_unless_e2e_ready(
            len(rtsp_urls) >= 2, f"need at least two RTSP {codec.upper()} URLs for multistream e2e"
        )
        output_cfg = e2e_config_section("multi-stream-obb-detector", "testing.e2e.output")
        total_saved_frames = int(output_cfg["total_saved_frames"])
        metadata_port_base = _env_int_or_default("SIMANEAT_APPS_TEST_INSIGHT_METADATA_PORT", 9100)

        config_path = e2e_config_writer(
            {
                "streams": rtsp_urls[:2],
                "input": {"codec": codec},
                "output": {
                    "insight": {
                        "host": E2E_INSIGHT_HOST,
                        "video_port_base": _env_int_or_default(
                            "SIMANEAT_APPS_TEST_INSIGHT_VIDEO_PORT", 9000
                        ),
                        "metadata_port_base": metadata_port_base,
                    },
                    "debug_dir": str(tmp_output_dir),
                },
                "inference": {"frames": 140},
            }
        )

        cmd = [sys.executable, str(MAIN_PY), "--config", str(config_path)]
        with MetadataJsonListener(
            E2E_INSIGHT_HOST,
            metadata_port_base,
            num_ports=2,
            require_all_ports=True,
        ) as metadata_listener:
            result = run_until_output_files(
                cmd,
                tmp_output_dir,
                total_saved_frames,
                test_timeout_ms / 1000,
                cwd=str(EXAMPLE_DIR),
            )
            metadata = metadata_listener.wait_for_messages(5.0)

        assert_exited_cleanly(result)
        assert metadata.success, (
            f"object-detection metadata was not received on all streams: {metadata.error}"
        )
        labels = {line.strip() for line in LABELS_FILE.read_text().splitlines() if line.strip()}
        assert_oriented_objects(metadata, labels)
        assert_streamed_frames_are_usable(tmp_output_dir, total_saved_frames, streams=2)
