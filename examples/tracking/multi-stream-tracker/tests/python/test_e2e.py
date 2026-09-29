"""E2E tests for multi-stream-tracker (Python)."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest

from tests.utils.metadata_json_listener import MetadataJsonListener


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


def assert_tracking_metadata(metadata, expected_labels: set[str], stream_count: int) -> None:
    """Fail unless every stream published real tracks with valid labels and stable IDs.

    Without this the suite passes on `{"tracks": []}`: the listener only checks
    that some valid JSON arrived on each port.
    """
    per_port: dict[int, list[tuple[str, list[dict]]]] = {}
    for message in metadata.messages:
        payload = json.loads(message.payload)
        tracks = payload.get("tracks")
        assert isinstance(tracks, list), f"port {message.port}: 'tracks' must be a list"
        per_port.setdefault(message.port, []).append((message.frame_id, tracks))

    assert len(per_port) == stream_count, (
        f"expected tracking metadata from {stream_count} streams, got {sorted(per_port)}"
    )

    for port, frames in sorted(per_port.items()):
        ids_per_frame = []
        for frame_id, tracks in frames:
            for track in tracks:
                assert set(track) == {"id", "label", "confidence", "bbox"}, (
                    f"port {port} frame {frame_id}: unexpected metadata keys {sorted(track)}"
                )
                assert track["label"] in expected_labels, (
                    f"port {port} frame {frame_id}: label {track['label']!r} "
                    f"is not a configured class {sorted(expected_labels)}"
                )
                assert str(track["id"]).isdigit() and int(track["id"]) > 0, (
                    f"port {port} frame {frame_id}: invalid track id {track['id']!r}"
                )
                assert 0.0 <= float(track["confidence"]) <= 1.0
                x, y, w, h = (float(value) for value in track["bbox"])
                assert x >= 0 and y >= 0 and w > 0 and h > 0, (
                    f"port {port} frame {frame_id}: degenerate bbox {track['bbox']}"
                )
            if tracks:
                ids_per_frame.append({str(track["id"]) for track in tracks})

        assert ids_per_frame, f"port {port}: every frame published an empty track list"

        all_ids = set().union(*ids_per_frame)
        assert len(all_ids) >= 2, (
            f"port {port}: expected multiple tracks over the run, saw ids {sorted(all_ids)}"
        )

        # A tracker that re-numbers every frame would never repeat an id.
        carried = any(
            previous & current for previous, current in zip(ids_per_frame, ids_per_frame[1:])
        )
        assert carried, (
            f"port {port}: no track id persisted between consecutive published frames; "
            "ids are not stable"
        )


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
        ) as metadata_listener:
            result = run_until_output_files(
                cmd,
                tmp_output_dir,
                total_saved_frames,
                test_timeout_ms / 1000,
                cwd=str(EXAMPLE_DIR),
            )
            metadata = metadata_listener.wait_for_messages(5.0)

        assert result.returncode == 0, (
            f"main.py exited with code {result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
        assert metadata.success, (
            f"tracking metadata was not received on all streams: {metadata.error}"
        )
        from main import load_app_config

        assert_tracking_metadata(
            metadata,
            expected_labels={entry.label for entry in load_app_config(config_path).tracker_classes},
            stream_count=2,
        )

        files = [
            path
            for path in tmp_output_dir.rglob("*")
            if path.is_file() and path.name != "config.yaml"
        ]
        assert len(files) >= total_saved_frames, (
            f"Expected at least {total_saved_frames} sampled output files, got {len(files)}"
        )
        assert all(path.stat().st_size > 0 for path in files)
