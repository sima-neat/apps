"""Unit tests for detection-to-vlm-assistant (Python)."""

from __future__ import annotations

import ast
import base64
import json
from pathlib import Path
from queue import Empty, Queue
import socket
import sys
import threading
from types import SimpleNamespace
from urllib import error, request

import pytest
import yaml

EXAMPLE_DIR = Path(__file__).resolve().parent.parent.parent
CONFIG_YAML = EXAMPLE_DIR / "src" / "common" / "config.yaml"
DETECTOR_APP_PY = EXAMPLE_DIR / "src" / "python" / "detector_app.py"
GENAI_SERVER_PY = EXAMPLE_DIR / "src" / "python" / "genai_server.py"
DASHBOARD_PY = EXAMPLE_DIR / "src" / "python" / "dashboard.py"
WEB_DIR = EXAMPLE_DIR / "src" / "python" / "web"

sys.path.insert(0, str(DASHBOARD_PY.parent))
from dashboard import DashboardServer, DashboardState  # noqa: E402
from async_pipeline import AsyncModelPipeline  # noqa: E402
from semantic_describer import AppearanceDescriber, normalize_appearance  # noqa: E402
from tracker import ObjectTracker, TrackerConfig  # noqa: E402
from visibility_gate import FullPersonVisibilityGate, overlapping_bboxes  # noqa: E402


@pytest.mark.unit
def test_input_options_format_uses_pyneat_enum() -> None:
    tree = ast.parse(DETECTOR_APP_PY.read_text(encoding="utf-8"))
    bad_lines: list[int] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not isinstance(node.value, ast.Constant) or not isinstance(node.value.value, str):
            continue
        for target in node.targets:
            if isinstance(target, ast.Attribute) and target.attr == "format":
                bad_lines.append(node.lineno)

    assert bad_lines == []


@pytest.mark.unit
def test_genai_config_and_scripts_use_genai_names() -> None:
    raw = yaml.safe_load(CONFIG_YAML.read_text(encoding="utf-8"))

    assert raw["genai_server"]["host"]
    assert raw["genai_server"]["port"] > 0
    assert raw["genai_server"]["model"]["name"] == "Gemma-4-E2B-it-GPTQ-a16w4"
    assert raw["genai_server"]["model"]["path"].endswith(
        "/gemma-4-E2B-it-GPTQ-a16w4"
    )
    assert "TextOnly" not in raw["genai_server"]["model"]["path"]
    assert raw["genai"]["host"]
    assert raw["genai"]["port"] > 0
    assert raw["genai"]["max_tokens"] == 64
    assert raw["genai"]["max_pending_requests"] == 8
    assert "interval_seconds" not in raw["genai"]


@pytest.mark.unit
def test_browser_dashboard_config_and_assets() -> None:
    raw = yaml.safe_load(CONFIG_YAML.read_text(encoding="utf-8"))

    assert raw["web"]["enabled"] is True
    assert raw["web"]["port"] == 5000
    assert raw["web"]["title"] == "Real-Time Semantic People Tracker"
    assert raw["web"]["tls"]["enabled"] is True
    assert raw["web"]["tls"]["cert"]
    assert raw["web"]["tls"]["key"]
    assert raw["web"]["tls"]["ca_cert"]
    assert raw["webcam"]["target_fps"] == 30
    assert raw["webcam"]["upload_max_width"] == 640
    assert raw["webcam"]["max_frame_bytes"] >= 1024 * 1024
    assert raw["inference"]["queue_depth"] >= 2
    assert raw["inference"]["internal_queue_depth"] >= 1
    assert raw["inference"]["min_score"] == 0.25
    assert raw["tracking"] == {
        "high_score_threshold": 0.55,
        "new_track_threshold": 0.65,
        "match_iou_threshold": 0.20,
        "center_distance_enabled": True,
        "max_center_distance": 1.5,
        "velocity_momentum": 0.80,
        "max_missing_frames": 30,
        "min_confirmed_hits": 3,
        "max_active_tracks": 32,
        "description_edge_margin_ratio": 0.03,
        "description_min_clear_frames": 5,
    }
    assert "source" not in raw
    assert "insight" not in raw
    for asset in ("index.html", "app.css", "app.js"):
        assert (WEB_DIR / asset).is_file()


@pytest.mark.unit
def test_dashboard_state_and_http_contract() -> None:
    state = DashboardState(
        title="Visual AI",
        model_name="test-vlm",
        target_fps=30,
    )
    detections = [
        {
            "track_id": 7,
            "label": "person",
            "score": 0.93,
            "bbox": [4, 5, 20, 30],
            "semantic_status": "pending",
            "semantic_description": None,
        }
    ]
    state.record_frame(detections, width=640, height=360, tracker_latency_ms=0.2)
    detection_revision, detection_update, closed = state.wait_for_detection_update(
        0, timeout=0
    )
    assert detection_revision == 1
    assert closed is False
    assert detection_update is not None
    assert detection_update["source"] == {"width": 640, "height": 360}
    assert detection_update["detections"] == detections

    state.update_track_semantic(
        7,
        status="ready",
        description="man, white shirt, backpack",
        latency_ms=123.4,
    )
    detection_revision, semantic_update, closed = state.wait_for_detection_update(
        detection_revision, timeout=0
    )
    assert detection_revision == 2
    assert closed is False
    assert semantic_update is not None
    assert semantic_update["detections"][0]["semantic_status"] == "ready"
    assert semantic_update["detections"][0]["semantic_description"] == (
        "man, white shirt, backpack"
    )
    snapshot = state.snapshot()
    assert snapshot["metrics"]["current_tracks"] == 1
    assert snapshot["metrics"]["tracks_created"] == 1
    assert snapshot["metrics"]["tracks_described"] == 1
    assert snapshot["metrics"]["vlm_latency_ms"] == 123
    assert snapshot["metrics"]["tracker_latency_ms"] == 0.2
    assert "events" not in snapshot
    assert "settings" not in snapshot

    state.update_track_semantic(
        7, status="unavailable", description=None, latency_ms=None
    )
    unavailable_revision, unavailable_update, _closed = (
        state.wait_for_detection_update(detection_revision, timeout=0)
    )
    assert unavailable_revision == 3
    assert unavailable_update["detections"][0]["semantic_status"] == "unavailable"

    server = DashboardServer(state, "127.0.0.1", 0)
    server.start()
    base_url = f"http://127.0.0.1:{server.port}"
    try:
        with request.urlopen(f"{base_url}/api/health", timeout=2) as response:
            health = json.load(response)
        assert health["status"] == "ok"

        frame_request = request.Request(
            f"{base_url}/api/frame",
            data=b"jpeg-frame-payload",
            headers={"Content-Type": "image/jpeg"},
            method="POST",
        )
        with request.urlopen(frame_request, timeout=2) as response:
            accepted = json.load(response)
        assert accepted == {"accepted": True, "sequence": 1}
        assert state.wait_for_frame(0, timeout=0) == (1, b"jpeg-frame-payload")
        assert state.submit_frame(
            b"newer-frame", source_epoch=100, source_sequence=2
        ) == (2, True)
        assert state.submit_frame(
            b"stale-frame", source_epoch=100, source_sequence=1
        ) == (2, False)
        assert state.wait_for_frame(1, timeout=0) == (2, b"newer-frame")

        req = request.Request(
            f"{base_url}/api/settings",
            data=b"{}",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(error.HTTPError) as rejected:
            request.urlopen(req, timeout=2)
        assert rejected.value.code == 404

        with request.urlopen(f"{base_url}/", timeout=2) as response:
            page = response.read().decode()
        assert "Real-Time Semantic People Tracker" in page
        assert "webcam-video" in page
        assert "modalix-webcam-ca.crt" in page
        assert "Latest insight" not in page
        assert "Analysis policy" not in page
        assert "Recent visual events" not in page
        with request.urlopen(f"{base_url}/assets/app.js", timeout=2) as response:
            app_js = response.read().decode()
        assert 'EventSource("/api/detections")' in app_js
        assert "waiting for clear view…" in app_js
        assert 'font = "700 24px Inter' in app_js
        assert "wrapOverlayText" in app_js
        assert "topLabelX" in app_js
        assert "semanticX" in app_js
    finally:
        server.stop()


def _person(x1: float, score: float = 0.9, class_id: int = 0) -> dict:
    return {
        "x1": x1,
        "y1": 10.0,
        "x2": x1 + 30.0,
        "y2": 90.0,
        "score": score,
        "class_id": class_id,
    }


@pytest.mark.unit
def test_tracker_confirms_reuses_recovers_and_expires_tracks() -> None:
    tracker = ObjectTracker(
        TrackerConfig(
            high_score_threshold=0.55,
            new_track_threshold=0.65,
            min_confirmed_hits=3,
            max_missing_frames=2,
        )
    )
    assert tracker.update([_person(10)], 0) == []
    assert tracker.update([_person(12)], 1) == []
    confirmed = tracker.update([_person(14)], 2)
    assert len(confirmed) == 1
    track_id = confirmed[0].track_id
    recovered = tracker.update([_person(16, score=0.4)], 3)
    assert [track.track_id for track in recovered] == [track_id]
    tracker.update([], 4)
    tracker.update([], 5)
    tracker.update([], 6)
    assert tracker.active_track_count() == 0
    assert tracker.update([_person(18)], 7) == []
    assert tracker.update([_person(19)], 8) == []
    replacement = tracker.update([_person(20)], 9)
    assert replacement[0].track_id != track_id


@pytest.mark.unit
def test_tracker_assigns_multiple_people_and_bounds_state() -> None:
    tracker = ObjectTracker(TrackerConfig(min_confirmed_hits=1, max_active_tracks=2))
    tracks = tracker.update([_person(0), _person(100), _person(200)], 0)
    assert len(tracks) == 2
    assert len({track.track_id for track in tracks}) == 2
    assert tracker.active_track_ids() == {1, 2}


@pytest.mark.unit
def test_full_person_visibility_gate_requires_stable_edge_clearance() -> None:
    gate = FullPersonVisibilityGate(edge_margin_ratio=0.03, min_clear_frames=5)
    clipped = (0.0, 30.0, 180.0, 330.0)
    full = (40.0, 30.0, 180.0, 330.0)

    assert not gate.observe(1, clipped, 640, 360)
    for _ in range(4):
        assert not gate.observe(1, full, 640, 360)
    assert gate.observe(1, full, 640, 360)

    bottom_edge = (40.0, 30.0, 180.0, 360.0)
    for _ in range(4):
        assert not gate.observe(3, bottom_edge, 640, 360)
    assert gate.observe(3, bottom_edge, 640, 360)

    assert not gate.observe(2, full, 640, 360)
    assert not gate.observe(2, full, 640, 360)
    assert not gate.observe(2, (40.0, 30.0, 635.0, 330.0), 640, 360)
    for _ in range(4):
        assert not gate.observe(2, full, 640, 360)
    assert gate.observe(2, full, 640, 360)

    for _ in range(4):
        assert not gate.observe(4, full, 640, 360)
    gate.reset(4)
    for _ in range(4):
        assert not gate.observe(4, full, 640, 360)
    assert gate.observe(4, full, 640, 360)

    gate.retain_tracks({2})
    for _ in range(4):
        assert not gate.observe(1, full, 640, 360)


@pytest.mark.unit
def test_overlapping_people_are_marked_unsafe_for_vlm_crops() -> None:
    first = (10.0, 10.0, 80.0, 160.0)
    second = (60.0, 40.0, 130.0, 170.0)
    separate = (200.0, 20.0, 270.0, 160.0)
    touching = (270.0, 20.0, 330.0, 160.0)

    assert overlapping_bboxes([first, second, separate, touching]) == {
        first,
        second,
    }


@pytest.mark.unit
def test_appearance_normalization_preserves_complete_attribute_line() -> None:
    assert normalize_appearance('Description: "man, white shirt, black backpack."') == (
        "man, white shirt, black backpack"
    )
    assert normalize_appearance(
        "male, white, long-sleeved shirt, black-framed glasses, silver necklace"
    ) == "male, white, long-sleeved shirt, black-framed glasses, silver necklace"
    with pytest.raises(ValueError, match="comma-separated"):
        normalize_appearance("The person is wearing a bright red winter jacket")
    with pytest.raises(ValueError, match="comma-separated"):
        normalize_appearance("The image shows a person.")
    with pytest.raises(ValueError, match="empty"):
        normalize_appearance("  \n")


@pytest.mark.unit
def test_appearance_describer_runs_once_for_each_track() -> None:
    calls: list[str] = []
    updates: list[tuple[int, str]] = []
    complete = threading.Event()

    def describe(crop: str) -> str:
        calls.append(crop)
        return f"person, {crop} shirt"

    def updated(track_id, state, _latency) -> None:
        updates.append((track_id, state.status))
        if len(updates) == 2:
            complete.set()

    describer = AppearanceDescriber(describe, updated, queue_size=8)
    describer.retain_tracks({1, 2})
    assert describer.schedule(1, "white")
    assert not describer.schedule(1, "duplicate")
    assert describer.schedule(2, "blue")
    describer.start()
    try:
        assert complete.wait(timeout=2)
        assert calls == ["white", "blue"]
        assert updates == [(1, "ready"), (2, "ready")]
        assert describer.state_for(1).description == "person, white shirt"
    finally:
        describer.close()


@pytest.mark.unit
def test_appearance_describer_retries_once_then_stops() -> None:
    attempts = 0
    complete = threading.Event()

    def fail(_crop) -> str:
        nonlocal attempts
        attempts += 1
        raise OSError("temporary failure")

    observed = []

    def updated(_track_id, state, _latency) -> None:
        observed.append(state)
        complete.set()

    describer = AppearanceDescriber(fail, updated, max_attempts=2)
    describer.retain_tracks({3})
    assert describer.schedule(3, "crop")
    describer.start()
    try:
        assert complete.wait(timeout=2)
        assert attempts == 2
        assert observed[0].status == "unavailable"
        assert not describer.schedule(3, "again")
    finally:
        describer.close()


@pytest.mark.unit
def test_appearance_describer_second_attempt_can_succeed() -> None:
    attempts = 0
    complete = threading.Event()

    def describe(_crop) -> str:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise TimeoutError("first attempt timed out")
        return "woman, blue jacket"

    updates = []

    def updated(_track_id, state, _latency) -> None:
        updates.append(state)
        complete.set()

    describer = AppearanceDescriber(describe, updated, max_attempts=2)
    describer.retain_tracks({8})
    assert describer.schedule(8, "crop")
    describer.start()
    try:
        assert complete.wait(timeout=2)
        assert attempts == 2
        assert updates[0].status == "ready"
        assert updates[0].description == "woman, blue jacket"
    finally:
        describer.close()


@pytest.mark.unit
def test_appearance_describer_skips_expired_queued_track() -> None:
    calls = 0

    def describe(_crop) -> str:
        nonlocal calls
        calls += 1
        return "white shirt"

    describer = AppearanceDescriber(describe, lambda *_args: None)
    describer.retain_tracks({4})
    assert describer.schedule(4, "crop")
    describer.retain_tracks(set())
    describer.start()
    try:
        threading.Event().wait(0.1)
        assert calls == 0
        assert describer.state_for(4) is None
    finally:
        describer.close()


@pytest.mark.unit
def test_async_model_pipeline_correlates_frames_and_closes() -> None:
    class FakeRunner:
        def __init__(self) -> None:
            self.outputs: Queue = Queue()
            self.closed = False
            self.input_closed = False

        def push(self, samples) -> bool:
            source = samples[0]
            self.outputs.put(
                SimpleNamespace(frame_id=source.frame_id, tensors=[source.frame_id])
            )
            return True

        def pull(self, timeout_ms: int):
            try:
                return self.outputs.get(timeout=timeout_ms / 1000)
            except Empty:
                return None

        def close_input(self) -> None:
            self.input_closed = True

        def close(self) -> None:
            self.closed = True

    runner = FakeRunner()
    received: list[tuple[str, int]] = []
    complete = threading.Event()

    def on_result(context, result) -> None:
        received.append((context, result.frame_id))
        if len(received) == 3:
            complete.set()

    pipeline = AsyncModelPipeline(runner, on_result, pull_timeout_ms=20)
    pipeline.start()
    try:
        for frame_id in (10, 11, 12):
            sample = SimpleNamespace(frame_id=frame_id)
            assert pipeline.submit(frame_id, sample, f"frame-{frame_id}")
        assert complete.wait(timeout=1)
        assert received == [
            ("frame-10", 10),
            ("frame-11", 11),
            ("frame-12", 12),
        ]
        assert pipeline.submitted == 3
        assert pipeline.completed == 3
        assert pipeline.dropped == 0
        pipeline.raise_if_failed()
    finally:
        pipeline.close()

    assert runner.input_closed is True
    assert runner.closed is True


@pytest.mark.unit
def test_binary_websocket_frame_ingest() -> None:
    state = DashboardState(
        title="Visual AI",
        model_name="test-vlm",
        target_fps=30,
    )
    server = DashboardServer(state, "127.0.0.1", 0)
    server.start()
    client = socket.create_connection(("127.0.0.1", server.port), timeout=2)
    try:
        key = base64.b64encode(b"0123456789abcdef").decode()
        client.sendall(
            (
                "GET /api/frames HTTP/1.1\r\n"
                f"Host: 127.0.0.1:{server.port}\r\n"
                "Upgrade: websocket\r\n"
                "Connection: Upgrade\r\n"
                f"Sec-WebSocket-Key: {key}\r\n"
                "Sec-WebSocket-Version: 13\r\n\r\n"
            ).encode()
        )
        response = b""
        while b"\r\n\r\n" not in response:
            response += client.recv(4096)
        assert b"101 Switching Protocols" in response

        payload = b"binary-jpeg-frame"
        mask = b"\x01\x02\x03\x04"
        encoded = bytes(value ^ mask[index & 3] for index, value in enumerate(payload))
        client.sendall(bytes([0x82, 0x80 | len(payload)]) + mask + encoded)
        assert state.wait_for_frame(0, timeout=1) == (1, payload)

        client.sendall(b"\x88\x80" + mask)
    finally:
        client.close()
        server.stop()
