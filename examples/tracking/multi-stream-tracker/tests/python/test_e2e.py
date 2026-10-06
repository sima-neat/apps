"""E2E tests for multi-stream-tracker (Python)."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest

from tests.utils.metadata_json_listener import MetadataJsonListener
from tests.utils.output_assertions import assert_streamed_frames_are_usable
from tests.utils.process_assertions import assert_exited_cleanly


EXAMPLE_DIR = Path(__file__).resolve().parent.parent.parent
PYTHON_DIR = EXAMPLE_DIR / "src" / "python"
MAIN_PY = PYTHON_DIR / "main.py"
E2E_INSIGHT_HOST = "127.0.0.1"

if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))


def _runtime_deps_ready() -> bool:
    return all(importlib.util.find_spec(name) is not None for name in ("cv2", "numpy", "pyneat"))


def _env_int_or_default(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw else default


def assert_tracking_metadata(metadata, expected_labels: set[str]) -> None:
    """Fail unless the published tracks carry valid classes, ids and boxes.

    Every published track must satisfy the metadata contract: a configured class
    label, a positive integer id, and a box inside the frame. Whether any track
    is published depends on the stream content, so an empty run is not treated
    as a failure here; that tracking produces tracks at all is covered by the
    unit tests and the golden parity cases.
    """
    checked = 0
    for message in metadata.messages:
        payload = json.loads(message.payload)
        # Insight wraps the array: {"data": {"tracks": [...]}, "frame_id": ..., "type": ...}
        tracks = (payload.get("data") or {}).get("tracks")
        assert isinstance(tracks, list), (
            f"port {message.port}: 'data.tracks' must be a list, got {type(tracks).__name__}"
        )
        for track in tracks:
            assert set(track) == {"id", "label", "confidence", "bbox"}, (
                f"port {message.port} frame {message.frame_id}: "
                f"unexpected metadata keys {sorted(track)}"
            )
            assert track["label"] in expected_labels, (
                f"port {message.port} frame {message.frame_id}: label {track['label']!r} "
                f"is not a configured class {sorted(expected_labels)}"
            )
            assert str(track["id"]).isdigit() and int(track["id"]) > 0, (
                f"port {message.port} frame {message.frame_id}: invalid id {track['id']!r}"
            )
            assert 0.0 <= float(track["confidence"]) <= 1.0, (
                f"port {message.port} frame {message.frame_id}: confidence out of range"
            )
            x, y, w, h = (float(value) for value in track["bbox"])
            assert x >= 0 and y >= 0 and w > 0 and h > 0, (
                f"port {message.port} frame {message.frame_id}: degenerate bbox {track['bbox']}"
            )
            checked += 1

    print(f"[e2e] validated {checked} published tracks")


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
        output_cfg = e2e_config_section("multi-stream-tracker", "testing.e2e.output")
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
                "inference": {
                    "frames": 140,
                },
            }
        )

        cmd = [
            sys.executable,
            str(MAIN_PY),
            "--config",
            str(config_path),
        ]
        with MetadataJsonListener(
            E2E_INSIGHT_HOST,
            metadata_port_base,
            num_ports=2,
            metadata_type="tracking",
            data_array_key="tracks",
            require_all_ports=True,
            # min_object_count stays at 0 on purpose. Requiring a track here
            # would make the suite depend on the CI stream carrying a
            # configured class: the H.265 fixture does, the H.264 fixture
            # published 140 empty frames on both streams. require_all_ports
            # still fails if an application stops publishing tracking
            # metadata, and every track that is published is checked below.
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
            f"tracking metadata was not received on all streams: {metadata.error}"
        )
        from main import load_app_config

        assert_tracking_metadata(
            metadata,
            expected_labels={entry.label for entry in load_app_config(config_path).tracker_classes},
        )

        assert_streamed_frames_are_usable(
            tmp_output_dir, total_saved_frames, streams=len(rtsp_urls[:2])
        )
