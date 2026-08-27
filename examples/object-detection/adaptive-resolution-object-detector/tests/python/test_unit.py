"""Unit tests for the adaptive-resolution-object-detector example."""

from __future__ import annotations

import json
from pathlib import Path
import dataclasses
import importlib.util
import io
import os
import signal
import subprocess
import threading
import sys
import textwrap
import time
from types import SimpleNamespace

import pytest

EXAMPLE_DIR = Path(__file__).resolve().parent.parent.parent
PYTHON_DIR = EXAMPLE_DIR / "src" / "python"
MAIN_PY = PYTHON_DIR / "main.py"          # entry point: --mode dispatcher
ADAPTIVE_PY = PYTHON_DIR / "adaptive_app.py"
FUSED_PY = PYTHON_DIR / "fused_app.py"

if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

pytestmark = pytest.mark.unit


def write_config(tmp_path: Path, streams_block: str) -> Path:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "model:",
                "  path: assets/models/yolo26m-det-int8-b1.tar.gz",
                streams_block,
                "output:",
                "  insight:",
                "    host: 127.0.0.1",
            ]
        ),
        encoding="utf-8",
    )
    return config_path


RICH_TWO = textwrap.dedent(
    """\
    streams:
      max_streams: 8
      sources:
        - id: cam-1
          rtsp_url: rtsp://127.0.0.1:8554/src1
        - id: cam-2
          rtsp_url: rtsp://127.0.0.1:8554/src2"""
)

BARE_THREE = textwrap.dedent(
    """\
    streams:
      - rtsp://127.0.0.1:8554/src1
      - rtsp://127.0.0.1:8554/src2
      - rtsp://127.0.0.1:8554/src3"""
)


class TestMainEntrypoint:
    def test_help_runs(self):
        result = subprocess.run(
            [sys.executable, str(MAIN_PY), "--help"],
            capture_output=True,
            text=True,
            cwd=str(EXAMPLE_DIR),
            timeout=20,
        )
        assert result.returncode == 0
        assert "--config" in result.stdout
        assert "--validate-config-only" in result.stdout

    def test_missing_config_file_fails_cleanly(self):
        result = subprocess.run(
            [sys.executable, str(MAIN_PY), "--config", "does-not-exist.yaml"],
            capture_output=True,
            text=True,
            cwd=str(EXAMPLE_DIR),
            timeout=20,
        )
        assert result.returncode == 2
        assert "config file not found" in result.stderr

    def test_validate_config_only_reports_stream_count(self, tmp_path: Path):
        config_path = write_config(tmp_path, RICH_TWO)
        result = subprocess.run(
            [sys.executable, str(MAIN_PY), "--config", str(config_path), "--validate-config-only"],
            capture_output=True,
            text=True,
            cwd=str(EXAMPLE_DIR),
            timeout=20,
        )
        assert result.returncode == 0
        assert "streams=2" in result.stdout


class TestConfigLoading:
    def test_rich_sources(self, tmp_path: Path):
        from adaptive_app import load_app_config

        cfg = load_app_config(write_config(tmp_path, RICH_TWO))
        assert cfg.model_path == "assets/models/yolo26m-det-int8-b1.tar.gz"
        assert [s.id for s in cfg.sources] == ["cam-1", "cam-2"]
        assert cfg.max_streams == 8

    def test_bare_list_autonames(self, tmp_path: Path):
        from adaptive_app import load_app_config

        cfg = load_app_config(write_config(tmp_path, BARE_THREE))
        assert [s.id for s in cfg.sources] == ["cam-1", "cam-2", "cam-3"]
        assert [s.rtsp_url for s in cfg.sources][0] == "rtsp://127.0.0.1:8554/src1"

    def test_rejects_over_max_streams(self, tmp_path: Path):
        from adaptive_app import load_app_config

        block = RICH_TWO.replace("max_streams: 8", "max_streams: 1")
        with pytest.raises(ValueError, match="max_streams"):
            load_app_config(write_config(tmp_path, block))

    def test_rejects_empty_streams(self, tmp_path: Path):
        from adaptive_app import load_app_config

        with pytest.raises(ValueError, match="streams"):
            load_app_config(write_config(tmp_path, "streams: []"))

    def test_reload_sources_matches_load(self, tmp_path: Path):
        from adaptive_app import load_app_config, reload_sources

        config_path = write_config(tmp_path, RICH_TWO)
        cfg = load_app_config(config_path)
        reloaded = reload_sources(config_path)
        assert [s.id for s in reloaded] == [s.id for s in cfg.sources]



class TestOutputPolicy:
    HEIGHTS = [2160, 1080, 720, 480]






class FakeMetadataSender:
    def __init__(self):
        self.calls = []
        self.raw = []

    def send_metadata(self, metadata_type, data_json, timestamp_ms, frame_id):
        self.calls.append((metadata_type, data_json, timestamp_ms, frame_id))
        return True

    # The adaptive app publishes here, not through send_metadata: Insight matches
    # a held frame by an exact rtp_timestamp key that the convenience API cannot
    # carry. send_metadata() swallows exceptions so a dropped datagram never
    # stalls detection - which also means a fake missing this method would make
    # the test silently assert nothing.
    def send_raw_json(self, payload_json):
        self.raw.append(payload_json)
        return True


class FakeSample:
    frame_id = 42
    pts_ns = 1_234_000_000


class TestMetadata:

    def test_send_metadata_uses_object_detection_contract(self):
        from adaptive_app import ProfileWindow, StreamRuntime, send_metadata

        sender = FakeMetadataSender()
        runtime = StreamRuntime(
            channel=0,
            id="cam-1",
            url="rtsp://127.0.0.1:8554/src1",
            labels=["person"],
            frame_w=100,
            frame_h=100,
            metadata_sender=sender,
            profile=ProfileWindow(False, "cam-1"),
        )
        boxes = [{"x1": 10.0, "y1": 20.0, "x2": 40.0, "y2": 60.0, "score": 0.75, "class_id": 0}]

        send_metadata(runtime, FakeSample(), boxes, stream_count=2)

        assert len(sender.raw) == 1, "detections must be published via send_raw_json"
        envelope = json.loads(sender.raw[0])
        assert envelope["type"] == "object-detection"
        assert envelope["stream_id"] == "cam-1"
        assert envelope["stream_index"] == 0
        assert envelope["frame_id"] == "42"
        assert envelope["pts_ns"] == FakeSample.pts_ns
        assert envelope["timestamp"] == 1234
        # `data` is already an object in the envelope, not a JSON string.
        data = envelope["data"]
        assert data["stream_count"] == 2
        assert data["objects"][0]["label"] == "person"


class TestModeDispatch:
    """`main.py` is an entry point, not an implementation.

    It selects a topology with --mode and forwards everything else unchanged.
    The C++ entry point (src/cpp/main.cpp) takes the same flag, which is what
    lets the pipelines chooser switch language without changing anything else.
    """

    def test_help_lists_both_modes(self):
        result = subprocess.run(
            [sys.executable, str(MAIN_PY), "--help"],
            capture_output=True, text=True, timeout=60,
        )
        assert result.returncode == 0
        assert "--mode" in result.stdout
        assert "adaptive" in result.stdout and "fused" in result.stdout

    def test_rejects_unknown_mode(self):
        result = subprocess.run(
            [sys.executable, str(MAIN_PY), "--mode", "nonsense",
             "--config", str(EXAMPLE_DIR / "src" / "common" / "config.yaml"),
             "--validate-config-only"],
            capture_output=True, text=True, timeout=60,
        )
        assert result.returncode != 0

    def test_both_implementations_are_importable(self):
        """Each mode's module must load on its own - main.py only dispatches."""
        import importlib

        for module_name, entry in (("adaptive_app", "load_app_config"),
                                   ("fused_app", "load_app_config")):
            module = importlib.import_module(module_name)
            assert hasattr(module, entry), f"{module_name}.{entry} missing"
            assert callable(module.main)

    def test_modes_do_not_share_a_config_schema(self, tmp_path):
        """Handing a mode the other's config must fail, not half-run.

        `adaptive` reads streams.sources plus the adaptive: policy sections;
        `fused` reads a bare streams: list. Silently accepting the wrong shape
        would start a pipeline with settings the user never asked for.
        """
        import adaptive_app
        import fused_app

        adaptive_cfg = tmp_path / "adaptive.yaml"
        adaptive_cfg.write_text(textwrap.dedent("""
            model:
              path: /tmp/model.tar.gz
              labels: /tmp/labels.txt
            adaptive:
              resolutions: [640]
            streams:
              max_streams: 4
              sources:
                - id: cam-1
                  rtsp_url: rtsp://127.0.0.1:8554/src1
            output:
              insight:
                host: 127.0.0.1
        """).strip() + "\n", encoding="utf-8")

        # The adaptive schema is not a valid fused config.
        with pytest.raises(Exception):
            fused_app.load_app_config(adaptive_cfg)

        fused_cfg = tmp_path / "fused.yaml"
        fused_cfg.write_text(textwrap.dedent("""
            model:
              path: /tmp/model.tar.gz
              labels: /tmp/labels.txt
            streams:
              - rtsp://127.0.0.1:8554/src1
            output:
              insight:
                host: 127.0.0.1
        """).strip() + "\n", encoding="utf-8")

        # The reverse is NOT symmetric, and that is deliberate: the adaptive
        # loader also accepts a bare list and auto-names the streams (see
        # TestConfigLoading::test_bare_list_autonames). So a fused config loads
        # under adaptive - what it will not do is silently pick up fused-only
        # settings, since there are none in that shape.
        adaptive_cfg_from_fused = adaptive_app.load_app_config(fused_cfg)
        assert len(adaptive_cfg_from_fused.sources) == 1
        assert adaptive_cfg_from_fused.sources[0].rtsp_url.endswith("/src1")

    def test_fused_accepts_more_than_four_streams(self, tmp_path):
        """The old "up to four streams" guard was a development-phase limit.

        This topology is the one the high-density example uses for 16/24/48
        streams, so the ceiling is measured capacity, not a placeholder. The C++
        side carries the same limit.
        """
        import fused_app

        urls = "\n".join(f"  - rtsp://127.0.0.1:8554/src{i}" for i in range(1, 9))
        cfg_path = tmp_path / "fused8.yaml"
        cfg_path.write_text(
            "model:\n  path: /tmp/model.tar.gz\n  labels: /tmp/labels.txt\n"
            f"streams:\n{urls}\n"
            "output:\n  insight:\n    host: 127.0.0.1\n",
            encoding="utf-8",
        )
        cfg = fused_app.load_app_config(cfg_path)
        assert len(cfg.rtsp_urls) == 8


def _wait_for(predicate, timeout: float = 10.0) -> bool:
    """Poll a manager predicate; stream workers settle on their own threads."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


class _IdleRun:
    """A runtime that stays up and never yields a sample."""

    def pull(self, output_name, timeout_ms):
        time.sleep(0.01)
        return None

    def running(self):
        return True

    def last_error(self):
        return ""

    def close(self):
        pass


def _fake_init_stream_runtime(cfg, channel, source, labels):
    """Builds for any stream except the ones named "bad-*"."""
    if source.id.startswith("bad"):
        raise RuntimeError(f"cannot build {source.id}")
    return SimpleNamespace(
        channel=channel,
        processed=0,
        run=_IdleRun(),
        profile=None,
        output_name="out",
        frame_w=640,
        frame_h=640,
    )


class TestStreamFailureIsolation:
    """A stream that fails must not take the healthy streams with it.

    The live panel adds streams into a running app, so an unreachable camera or a
    typo'd URL is an ordinary event, not a reason to stop the world. These tests
    pin that: a failing worker retires its own slot and leaves stop_event alone.
    """

    @staticmethod
    def _manager(tmp_path, monkeypatch):
        import adaptive_app

        monkeypatch.setattr(adaptive_app, "init_stream_runtime", _fake_init_stream_runtime)
        cfg = adaptive_app.load_app_config(write_config(tmp_path, RICH_TWO))
        return adaptive_app, adaptive_app.StreamManager(cfg, ["person"])

    def test_failed_add_leaves_healthy_streams_running(self, tmp_path, monkeypatch):
        adaptive_app, manager = self._manager(tmp_path, monkeypatch)
        try:
            manager.add(adaptive_app.StreamSource("good-1", "rtsp://127.0.0.1:8554/src1"))
            assert _wait_for(lambda: manager.active_count() == 1)

            manager.add(adaptive_app.StreamSource("bad-1", "rtsp://127.0.0.1:8554/nope"))
            assert _wait_for(lambda: "bad-1" in manager.failed)

            assert not manager.stop_event.is_set(), "a stream failure set the app-wide stop event"
            assert manager.active_count() == 1, "the healthy stream was torn down"
            assert not manager.all_failed()
        finally:
            manager.shutdown()

    def test_failed_stream_releases_its_channel(self, tmp_path, monkeypatch):
        adaptive_app, manager = self._manager(tmp_path, monkeypatch)
        try:
            manager.add(adaptive_app.StreamSource("good-1", "rtsp://127.0.0.1:8554/src1"))
            assert _wait_for(lambda: manager.active_count() == 1)
            manager.add(adaptive_app.StreamSource("bad-1", "rtsp://127.0.0.1:8554/nope"))
            assert _wait_for(lambda: "bad-1" in manager.failed)

            # Channel 0 belongs to good-1; bad-1 took 1 and must have given it back,
            # or repeated bad adds would exhaust max_streams.
            assert _wait_for(lambda: 1 in manager.free_channels)
            assert sorted(manager.free_channels) == [1, 2, 3, 4, 5, 6, 7]
        finally:
            manager.shutdown()

    def test_dropping_a_bad_source_clears_its_failed_state(self, tmp_path, monkeypatch):
        adaptive_app, manager = self._manager(tmp_path, monkeypatch)
        try:
            good = adaptive_app.StreamSource("good-1", "rtsp://127.0.0.1:8554/src1")
            manager.add(good)
            manager.add(adaptive_app.StreamSource("bad-1", "rtsp://127.0.0.1:8554/nope"))
            assert _wait_for(lambda: "bad-1" in manager.failed)

            manager.apply_sources([good])
            assert "bad-1" not in manager.failed
        finally:
            manager.shutdown()

    def test_all_failed_reports_a_total_failure(self, tmp_path, monkeypatch):
        adaptive_app, manager = self._manager(tmp_path, monkeypatch)
        try:
            manager.add(adaptive_app.StreamSource("bad-1", "rtsp://127.0.0.1:8554/nope"))
            assert _wait_for(manager.all_failed)
            # all_failed() is what ends a batch run; it still is not stop_event.
            assert not manager.stop_event.is_set()
        finally:
            manager.shutdown()


PIPELINES_DIR = EXAMPLE_DIR / "pipelines"
TOOLS_DIR = EXAMPLE_DIR / "tools"

# pipelines/ and tools/ are development and demonstration tooling. The CI "Test
# Apps Runtime" job runs this suite against the INSTALLED tree, which does not
# carry them, so tests that read those sources are meaningful from a source
# checkout and skip elsewhere.
#
# Probe the FILES, not the directories. The install tree turned out to contain a
# pipelines/ directory without the panel sources in it, so an is_dir() guard
# passed and every one of these tests still failed with FileNotFoundError.
def _all_present(*paths: Path) -> bool:
    return all(path.is_file() for path in paths)


_PIPELINE_SOURCES = (
    PIPELINES_DIR / "launcher.py",
    *(PIPELINES_DIR / f"pipeline-{name}" / part
      for name in ("scale", "live", "group")
      for part in ("ui_server.py", "pipeline.py")),
)

requires_pipelines = pytest.mark.skipif(
    not _all_present(*_PIPELINE_SOURCES),
    reason="pipelines/ panel sources are dev tooling and are not in the packaged runtime",
)
requires_tools = pytest.mark.skipif(
    not _all_present(TOOLS_DIR / "gen_test_config.sh"),
    reason="tools/ is dev tooling and is not in the packaged runtime",
)


class TestFusedRunTermination:
    """A fused run must end when its shared output closes.

    all_streams_done() used to return False outright for frame_limit <= 0, so a
    continuous run could never satisfy it. Once the detection output closed, the
    loop spun on a dead output at full CPU - a process reported as running that
    could never emit another detection.
    """

    @staticmethod
    def _streams(*specs):
        return [SimpleNamespace(processed=p, closed=c) for p, c in specs]

    def test_closed_streams_end_a_continuous_run(self):
        import fused_app

        streams = self._streams((123, True), (98, True))
        assert fused_app.all_streams_done(streams, 0) is True

    def test_open_streams_keep_a_continuous_run_going(self):
        import fused_app

        streams = self._streams((123, False), (98, True))
        assert fused_app.all_streams_done(streams, 0) is False

    def test_frame_limit_still_ends_a_batch_run(self):
        import fused_app

        assert fused_app.all_streams_done(self._streams((10, False), (10, False)), 10) is True
        assert fused_app.all_streams_done(self._streams((10, False), (9, False)), 10) is False
        # A stream that closed early does not hold up the rest.
        assert fused_app.all_streams_done(self._streams((10, False), (2, True)), 10) is True


def _load_ui_server(name: str):
    """Import one pipeline's ui_server.py under its own module name."""
    directory = PIPELINES_DIR / f"pipeline-{name}"
    # Every pipeline directory has its own pipeline.py. Leaving one cached under
    # the bare name would silently bind the next panel server to the wrong one,
    # making these tests order-dependent.
    cached = sys.modules.pop("pipeline", None)
    sys.path.insert(0, str(directory))
    try:
        spec = importlib.util.spec_from_file_location(
            f"ui_server_{name}", directory / "ui_server.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(directory))
        sys.modules.pop("pipeline", None)
        if cached is not None:
            sys.modules["pipeline"] = cached


def _multipart(payload: bytes, filename: str = "clip.mp4", boundary: str = "----bnd42",
               quote_boundary: bool = False):
    raw = boundary.encode()
    body = (
        b"--" + raw + b"\r\n"
        b'Content-Disposition: form-data; name="file"; filename="' + filename.encode() + b'"\r\n'
        b"Content-Type: video/mp4\r\n\r\n" + payload + b"\r\n"
        b"--" + raw + b"--\r\n"
    )
    declared = f'"{boundary}"' if quote_boundary else boundary
    return body, f"multipart/form-data; boundary={declared}"


@requires_pipelines
class TestUploadBounds:
    """/api/upload is unauthenticated on 0.0.0.0 and runs on a memory-tight board.

    Reading the whole declared body into memory - and splitting it, which copied
    it again - meant one large or concurrent upload could OOM the DevKit before a
    byte reached disk. The body now streams to the tempfile a chunk at a time
    behind a declared-size ceiling.
    """

    def test_upload_streams_to_disk_without_buffering_the_body(self):
        ui = _load_ui_server("live")
        payload = os.urandom(4 * 1024 * 1024)
        body, ctype = _multipart(payload)
        dest = io.BytesIO()
        name = ui.stream_multipart_file(io.BytesIO(body), len(body), ctype, dest)
        assert name == "clip.mp4"
        assert dest.getvalue() == payload

    def test_payload_containing_boundary_like_bytes_survives(self):
        ui = _load_ui_server("live")
        payload = b"--not-the-boundary--\r\n" * 500
        body, ctype = _multipart(payload)
        dest = io.BytesIO()
        assert ui.stream_multipart_file(io.BytesIO(body), len(body), ctype, dest) == "clip.mp4"
        assert dest.getvalue() == payload

    def test_boundary_split_across_reads_is_still_found(self):
        ui = _load_ui_server("live")

        class Trickle(io.RawIOBase):
            """Hands back 7 bytes at a time, straddling every marker."""

            def __init__(self, data):
                self.data, self.pos = data, 0

            def read(self, size=-1):
                take = 7 if size < 0 else min(7, size)
                chunk = self.data[self.pos:self.pos + take]
                self.pos += len(chunk)
                return chunk

        payload = os.urandom(60_000)
        body, ctype = _multipart(payload)
        dest = io.BytesIO()
        assert ui.stream_multipart_file(Trickle(body), len(body), ctype, dest) == "clip.mp4"
        assert dest.getvalue() == payload

    def test_quoted_boundary_parameter_is_understood(self):
        """RFC 2045 allows boundary="...". Keeping the quotes means the closing
        marker is never matched, which used to read as a successful upload."""
        ui = _load_ui_server("live")
        payload = os.urandom(50_000)
        body, ctype = _multipart(payload, quote_boundary=True)
        dest = io.BytesIO()
        assert ui.stream_multipart_file(io.BytesIO(body), len(body), ctype, dest) == "clip.mp4"
        assert dest.getvalue() == payload

    def test_upload_without_a_closing_boundary_is_refused(self):
        """A partial file, or one carrying the multipart trailer, must not be
        forwarded to Insight and reported complete."""
        ui = _load_ui_server("live")
        body, ctype = _multipart(os.urandom(20_000))
        truncated = body[: len(body) - 40]
        assert ui.stream_multipart_file(
            io.BytesIO(truncated), len(truncated), ctype, io.BytesIO()
        ) is None

        # A boundary that does not match the body is the same failure: nothing
        # ever terminates the part.
        assert ui.stream_multipart_file(
            io.BytesIO(body), len(body), "multipart/form-data; boundary=wrong", io.BytesIO()
        ) is None

    def test_malformed_uploads_are_refused(self):
        ui = _load_ui_server("live")
        assert ui.stream_multipart_file(io.BytesIO(b"x"), 1, "application/json", io.BytesIO()) is None

        body, ctype = _multipart(b"data", filename="")
        assert ui.stream_multipart_file(io.BytesIO(body), len(body), ctype, io.BytesIO()) is None

        # Part headers that never terminate must not be buffered without limit.
        runaway = b"--B\r\n" + b"X" * (ui.MAX_PART_HEADER_BYTES + 1024)
        assert ui.stream_multipart_file(
            io.BytesIO(runaway), len(runaway), "multipart/form-data; boundary=B", io.BytesIO()
        ) is None

    @pytest.mark.parametrize("name", ["live", "scale", "group"])
    def test_every_panel_server_carries_the_ceilings(self, name):
        """All three copies are maintained together; none may drift back."""
        source = (PIPELINES_DIR / f"pipeline-{name}" / "ui_server.py").read_text(encoding="utf-8")
        assert "MAX_UPLOAD_BYTES" in source
        assert "MAX_JSON_BYTES" in source
        assert "def stream_multipart_file" in source
        assert "def parse_multipart_file" not in source, "the unbounded parser came back"


@requires_pipelines
class TestGroupStaging:
    """Grouped mode owns a fixed slot range per group and must leave it clean."""

    def test_compaction_stops_every_position_in_the_range(self, monkeypatch):
        """An external stream holds a position without staging a slot.

        Stopping only positions past the new stream count therefore left a
        dropped Insight source playing: [pinned, external] losing the pinned
        stream leaves one stream, so position 0 - now backed by the external -
        was never stopped even though its media source was still running.
        """
        ui = _load_ui_server("group")
        stopped = []
        monkeypatch.setattr(ui, "_stop_slot", stopped.append)

        streams = [{"kind": "external", "url": "rtsp://cam/1"}]
        urls, staging = ui.plan_group(0, streams)
        assert staging == [], "an external stream must not claim an Insight slot"

        ui.stage_group(0, staging)
        assert stopped == [ui.insight_slot(0, pos) for pos in range(ui.GROUP_SIZE)]

    def test_staging_stays_inside_the_group_range(self, monkeypatch):
        """Whatever it stops, it must never touch a sibling group's slots."""
        ui = _load_ui_server("group")
        stopped = []
        monkeypatch.setattr(ui, "_stop_slot", stopped.append)
        monkeypatch.setattr(ui.pipeline, "api", lambda *args, **kwargs: None)
        # Staging a slot waits for the RTSP mount; nothing here is really coming up.
        monkeypatch.setattr(ui.time, "sleep", lambda _s: None)

        _urls, staging = ui.plan_group(1, [{"kind": "pinned", "video": "clip.mp4"}])
        ui.stage_group(1, staging)

        own = {ui.insight_slot(1, pos) for pos in range(ui.GROUP_SIZE)}
        assert set(stopped) <= own

    def test_activity_panel_reads_the_per_group_logs(self):
        """pipeline.LOG is the single-instance path and is never written here."""
        source = (PIPELINES_DIR / "pipeline-group" / "ui_server.py").read_text(encoding="utf-8")
        assert "pipeline.log_path(g)" in source
        assert "tail -n 40 {pipeline.LOG}" not in source


class TestFusedDebugFramePairing:
    """Saved debug images must show the frame their boxes came from.

    The decoded and detector branches are queued independently, so keeping only
    the newest decoded frame paired every image with whatever had arrived by
    then. Geometry was right, the moment was not - which reads as drift on
    anything moving.
    """

    @staticmethod
    def _stream(frames):
        return SimpleNamespace(index=0, debug_frames=dict(frames), debug_pairing_warned=False)

    def test_detection_gets_its_own_frame_not_the_newest(self):
        import fused_app

        stream = self._stream({10: "frame-10", 11: "frame-11", 12: "frame-12"})
        assert fused_app.take_debug_frame(stream, 11) == "frame-11"

    def test_matching_clears_the_frames_behind_it(self):
        import fused_app

        stream = self._stream({10: "frame-10", 11: "frame-11", 12: "frame-12"})
        fused_app.take_debug_frame(stream, 11)
        # 10's detection has already gone past; only 12 can still be claimed.
        assert list(stream.debug_frames) == [12]

    def test_no_frames_held_yields_nothing(self):
        import fused_app

        assert fused_app.take_debug_frame(self._stream({}), 7) is None

    def test_unmatched_id_falls_back_to_newest_and_warns_once(self, capsys):
        """Saving must keep working even if ids never line up - but say so."""
        import fused_app

        stream = self._stream({10: "frame-10", 12: "frame-12"})
        assert fused_app.take_debug_frame(stream, 99) == "frame-12"
        assert stream.debug_pairing_warned is True
        assert "no id matching" in capsys.readouterr().err

        stream.debug_frames = {13: "frame-13"}
        fused_app.take_debug_frame(stream, 99)
        assert capsys.readouterr().err == "", "the warning must not repeat every frame"


@requires_pipelines
class TestLauncherBounds:
    """The chooser on :8080 is the fourth unauthenticated 0.0.0.0 server."""

    def test_launcher_bounds_request_bodies(self):
        source = (PIPELINES_DIR / "launcher.py").read_text(encoding="utf-8")
        assert "MAX_BODY_BYTES" in source
        # The read must be gated, not reached with a client-declared length.
        gate = source.index("MAX_BODY_BYTES:")
        read = source.index("self.rfile.read(n)")
        assert gate < read, "the ceiling must be checked before the body is read"


class TestMetadataRtpTimestampParity:
    """The panel writes metadata_rtp_timestamp into the config both languages read.

    It is a SIBLING of `insight:` under `output:`, not nested inside it - the
    C++ parser first read it from output.insight.metadata_rtp_timestamp, a key
    the generated config never has, so it silently kept the "auto" default
    every time regardless of what the panel actually requested.
    """

    def test_python_and_cpp_read_the_same_key_path(self):
        py = (EXAMPLE_DIR / "src" / "python" / "adaptive_app.py").read_text(encoding="utf-8")
        cpp = (EXAMPLE_DIR / "src" / "cpp" / "adaptive_app.h").read_text(encoding="utf-8")
        assert 'string_or(output, "metadata_rtp_timestamp"' in py
        assert 'raw.string_or("output.metadata_rtp_timestamp"' in cpp
        assert "output.insight.metadata_rtp_timestamp" not in cpp, (
            "regression: reads a key the generated config never sets"
        )
        assert "send_raw_json" in cpp, "the convenience API cannot carry rtp_timestamp"
        assert "do-timestamp=true" in cpp, "auto must read what Core actually lowered"

    @requires_pipelines
    def test_live_config_places_it_as_a_sibling_of_insight(self):
        live = (PIPELINES_DIR / "pipeline-live" / "pipeline.py").read_text(encoding="utf-8")
        assert 'metadata_rtp_timestamp: "on"\n  insight:' in live


@requires_pipelines
class TestScaleAddRebuildsInstead(object):
    """The fused app has no config watch, so scale mode has no live add.

    config_streams() used to search for "- id:" lines, a key that only exists
    in the adaptive pipeline's rich schema - _config_scale() writes a bare list
    with no id at all. It always read back zero streams, so cmd_add() always
    reused slot 1 and appended a rich {id, rtsp_url} entry into a bare-list
    config the fused loader cannot parse next run.
    """

    @staticmethod
    def _pipeline():
        return _load_ui_server("scale").pipeline

    def test_config_streams_reads_the_bare_list_schema(self, tmp_path, monkeypatch):
        mod = self._pipeline()
        config = tmp_path / "scale-run.yaml"
        config.write_text(
            "model:\n  path: /tmp/model.tar.gz\n\n"
            "streams:\n  - rtsp://127.0.0.1:8554/src1\n  - rtsp://127.0.0.1:8554/src2\n\n"
            "input:\n  tcp: true\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(mod, "CONFIG", config)
        assert mod.config_streams() == [
            "rtsp://127.0.0.1:8554/src1",
            "rtsp://127.0.0.1:8554/src2",
        ]

    def test_config_streams_empty_when_no_config_yet(self, tmp_path, monkeypatch):
        mod = self._pipeline()
        monkeypatch.setattr(mod, "CONFIG", tmp_path / "missing.yaml")
        assert mod.config_streams() == []

    def test_add_counts_the_real_total_and_rebuilds(self, monkeypatch):
        """cmd_add() must see 2 existing streams as 2, not 0, and must go
        through a full rebuild - there is no live-append path any more."""
        mod = self._pipeline()
        monkeypatch.setattr(mod, "config_streams", lambda: ["u1", "u2"])
        calls = []
        monkeypatch.setattr(mod, "cmd_up", calls.append)
        mod.cmd_add()
        assert calls == [3]


@requires_pipelines
class TestGroupPlan:
    """Grouped mode's CLI must actually partition streams into groups.

    cmd_up()/cmd_add() previously called the single-instance write_config()/
    start_app(), which wrote group-run.yaml (explicitly unused in grouped mode)
    and ran ONE fused process for every stream - the plain `scale` topology,
    silently losing per-group failure and rebuild isolation.
    """

    @staticmethod
    def _pipeline():
        return _load_ui_server("group").pipeline

    def test_plan_fills_groups_in_order(self):
        mod = self._pipeline()
        assert mod.group_plan(0) == [0, 0, 0, 0]
        assert mod.group_plan(1) == [1, 0, 0, 0]
        assert mod.group_plan(4) == [4, 0, 0, 0]
        assert mod.group_plan(10) == [4, 4, 2, 0]
        assert mod.group_plan(16) == [4, 4, 4, 4]

    def test_plan_caps_at_the_group_ceiling(self):
        """group_plan() lays out at most GROUP_SIZE * MAX_GROUPS. It stays a
        pure layout function; refusing an impossible request is cmd_up()'s job
        (see test_up_rejects_counts_above_the_ceiling)."""
        mod = self._pipeline()
        ceiling = mod.GROUP_SIZE * mod.MAX_GROUPS
        assert sum(mod.group_plan(ceiling + 5)) == ceiling

    def test_up_rejects_counts_above_the_ceiling(self, monkeypatch):
        """Asking for more than fits must fail, not quietly run a smaller
        experiment and report that as success."""
        mod = self._pipeline()
        touched = []
        monkeypatch.setattr(mod, "stop_group", lambda g: touched.append(g))
        monkeypatch.setattr(mod, "start_group", lambda g: touched.append(g))
        monkeypatch.setattr(mod, "stage_group_sources", lambda *a: touched.append(a))

        ceiling = mod.GROUP_SIZE * mod.MAX_GROUPS
        with pytest.raises(SystemExit) as excinfo:
            mod.cmd_up(ceiling + 1)
        assert str(ceiling) in str(excinfo.value)
        assert touched == [], "nothing may be started or stopped on a refused request"

        # The ceiling itself is still accepted.
        assert sum(mod.group_plan(ceiling)) == ceiling

    def test_total_streams_counts_only_running_groups(self, tmp_path, monkeypatch):
        mod = self._pipeline()
        monkeypatch.setattr(mod, "HERE", tmp_path)
        monkeypatch.setattr(mod, "running_groups", lambda: [0, 2])

        (tmp_path / "group0-run.yaml").write_text(
            "streams:\n  - rtsp://a\n  - rtsp://b\n\ninput:\n  tcp: true\n",
            encoding="utf-8",
        )
        # group 1 has a leftover config from a previous, now-stopped run - it
        # must not be counted since running_groups() excludes it.
        (tmp_path / "group1-run.yaml").write_text(
            "streams:\n  - rtsp://stale\n\ninput:\n  tcp: true\n", encoding="utf-8"
        )
        (tmp_path / "group2-run.yaml").write_text(
            "streams:\n  - rtsp://c\n\ninput:\n  tcp: true\n", encoding="utf-8"
        )

        assert mod.total_streams() == 3

    def test_add_counts_the_real_total_and_rebuilds(self, monkeypatch):
        mod = self._pipeline()
        monkeypatch.setattr(mod, "total_streams", lambda: 5)
        calls = []
        monkeypatch.setattr(mod, "cmd_up", calls.append)
        mod.cmd_add()
        assert calls == [6]

    def test_stage_group_sources_uses_this_groups_own_channel_range(self, monkeypatch):
        """Slots must be channel_for(group, pos) + 1 - the group's own fixed
        range - never the group-unaware 1..count slots the old stage_sources()
        used, which collided with every other group's sources."""
        mod = self._pipeline()
        monkeypatch.setattr(mod, "media_files", lambda prefix: ["a.mp4", "b.mp4"])
        calls = []
        monkeypatch.setattr(mod, "api", lambda path, payload=None, **kw: calls.append((path, payload)))
        monkeypatch.setattr(mod.time, "sleep", lambda _s: None)

        tier = mod.Tier("720p", 1280, 720, "video", 8)
        mod.stage_group_sources(2, tier, 3)

        expected_slots = [mod.channel_for(2, pos) + 1 for pos in range(3)]
        assigned = [p["index"] for path, p in calls if path == "/api/mediasrc/assign"]
        assert assigned == expected_slots
        assert not any(path == "/api/mediasrc/stop-all" for path, _p in calls), (
            "grouped staging must never touch every other group's sources"
        )

    def test_shrinking_a_group_releases_the_positions_it_no_longer_uses(self, monkeypatch):
        """4 streams -> 2 must stop positions 2 and 3, not just leave them be.

        stage_group_sources() used to stop only range(count) - the NEW, smaller
        count - so the vacated higher positions kept playing in Insight with no
        detector referencing them: a resource leak on every shrink, not just the
        drop-to-zero case cmd_up() handles separately.
        """
        mod = self._pipeline()
        monkeypatch.setattr(mod, "media_files", lambda prefix: ["a.mp4"])
        stopped = []
        monkeypatch.setattr(mod, "_stop_insight_slot", stopped.append)
        monkeypatch.setattr(mod, "api", lambda *a, **kw: None)
        monkeypatch.setattr(mod.time, "sleep", lambda _s: None)

        tier = mod.Tier("720p", 1280, 720, "video", 8)
        mod.stage_group_sources(1, tier, 2)  # group 1 shrinking to 2 streams

        expected = {mod.channel_for(1, pos) + 1 for pos in range(mod.GROUP_SIZE)}
        assert set(stopped) == expected

    def test_dropping_a_group_to_zero_also_releases_its_slots(self, monkeypatch):
        """count == 0 skips stage_group_sources() entirely in cmd_up() - it must
        release the group's slots on that path too, not only stop the process."""
        mod = self._pipeline()
        monkeypatch.setattr(mod, "media_files", lambda prefix: sys.exit("must not be called"))
        monkeypatch.setattr(mod, "stop_group", lambda _g: True)
        monkeypatch.setattr(mod, "start_group", lambda _g: None)
        monkeypatch.setattr(mod, "wait_for_group", lambda _g, _n: True)
        monkeypatch.setattr(mod, "delivered_group", lambda _g: {})
        monkeypatch.setattr(mod, "api", lambda *a, **kw: None)
        stopped = []
        monkeypatch.setattr(mod, "_stop_insight_slot", stopped.append)

        mod.cmd_up(0)  # every group's count is 0

        for group in range(mod.MAX_GROUPS):
            expected = {mod.channel_for(group, pos) + 1 for pos in range(mod.GROUP_SIZE)}
            got = {s for s in stopped if s in expected}
            assert got == expected, f"group {group} slots not fully released"

    def test_up_never_calls_the_removed_single_instance_path(self):
        """The scale-topology leftovers this bug traced back to must be gone,
        not just unreachable - so nothing can wire the CLI back to them."""
        source = (PIPELINES_DIR / "pipeline-group" / "pipeline.py").read_text(encoding="utf-8")
        for name in ("def stage_sources", "def write_config(", "def start_app",
                     "def stop_app", "def app_running", "def config_streams",
                     "def append_stream", "def append_source"):
            assert name not in source, f"{name} should have been removed, not left dead"
        assert "def write_config_group" in source
        assert "def stage_group_sources" in source


class TestFusedFpsCap:
    """inference.fps must actually pace processing, not just label the banner.

    output_fps was set from cfg.fps at build time and never consulted again, so
    a configured cap changed nothing about the rate detection actually ran at -
    experiments relying on it measured the uncapped workload.
    """

    @staticmethod
    def _cfg(fps):
        return SimpleNamespace(fps=fps)

    @staticmethod
    def _stream(last_process_ms):
        return SimpleNamespace(last_process_ms=last_process_ms)

    def test_uncapped_never_throttles(self):
        import fused_app

        cfg = self._cfg(0)
        stream = self._stream(fused_app.time_ms())
        assert fused_app.should_throttle_fps(cfg, stream, fused_app.time_ms()) is False

    def test_throttles_before_the_target_interval_elapses(self):
        import fused_app

        cfg = self._cfg(10)  # 100 ms between frames
        now = fused_app.time_ms()
        stream = self._stream(now)
        assert fused_app.should_throttle_fps(cfg, stream, now + 50.0) is True

    def test_admits_once_the_target_interval_has_elapsed(self):
        import fused_app

        cfg = self._cfg(10)
        now = fused_app.time_ms()
        stream = self._stream(now)
        assert fused_app.should_throttle_fps(cfg, stream, now + 150.0) is False

    def test_a_throttled_sample_never_reaches_bbox_extraction(self, monkeypatch):
        """The whole point is skipping the parse/send work, not just the send."""
        import fused_app

        def boom(*_a, **_kw):
            raise AssertionError("extract_bbox_payload must not run while throttled")

        monkeypatch.setattr(fused_app, "extract_bbox_payload", boom)
        cfg = SimpleNamespace(fps=10, frames=0)
        stream = SimpleNamespace(last_process_ms=fused_app.time_ms(), processed=0)
        fused_app.process_output_sample(stream, cfg, sample=object(), detection_pull_ms=0.0)
        assert stream.processed == 0


class TestFusedDecoderFpsCap:
    """A high-fps source must not be declared to decoder admission at its literal
    rate - that gets the whole graph rejected before it starts. See
    src/python/fused_app.py's decoder_fps_cap for the mechanism this mirrors in
    C++ (src/cpp/fused_app.h), which previously had no equivalent at all."""

    def test_cpp_gates_admission_the_same_way_python_does(self):
        cpp = (EXAMPLE_DIR / "src" / "cpp" / "fused_app.h").read_text(encoding="utf-8")
        assert "decoder_fps_cap" in cpp
        assert "dec.dec_fps = capped ? -1 : opt.source_fps;" in cpp
        assert "opt.output_caps.fps = capped ? 0 : fps_out;" in cpp


class TestFusedDecoderRateIsNeverPinnedBelowTheSource:
    """inference.fps must NOT reach dec_fps / output_caps.fps.

    Pinning a decoded rate below the real stream fails caps negotiation
    ("framerate mismatch") - the modules say so themselves. A previous attempt
    to throttle pre-inference fed the requested rate into both fields, which
    would have made a 30-fps source with inference.fps=10 fail to negotiate
    instead of rate-limiting: worse than the original bug, where the setting
    merely did nothing. Only the source's true rate or an unpinned value are
    safe, so the throttle lives after the pull (see should_throttle_fps).
    """

    @pytest.mark.parametrize(
        "path", ["src/python/fused_app.py", "src/cpp/fused_app.h"]
    )
    def test_only_the_admission_cap_chooses_the_decoder_rate(self, path):
        source = (EXAMPLE_DIR / path).read_text(encoding="utf-8")
        assert "effective_decode_fps" not in source, (
            "regression: inference.fps is feeding a pinned decoder rate again"
        )
        # The admission cap may still swap in an UNPINNED value; that is safe.
        assert "decoder_fps_cap" in source

    def test_python_decoder_rate_comes_from_the_probed_source_rate(self):
        source = (EXAMPLE_DIR / "src" / "python" / "fused_app.py").read_text(encoding="utf-8")
        assert "dec.dec_fps = -1 if (cap > 0 and opt.source_fps > cap) else opt.source_fps" in source
        assert "opt.output_caps.fps = 0 if capped else fps" in source

    def test_cpp_decoder_rate_comes_from_the_probed_source_rate(self):
        source = (EXAMPLE_DIR / "src" / "cpp" / "fused_app.h").read_text(encoding="utf-8")
        assert "dec.dec_fps = capped ? -1 : opt.source_fps;" in source
        assert "opt.output_caps.fps = capped ? 0 : fps_out;" in source


class TestE2eCodecFixtures:
    """The adaptive e2e tests must not silently mix H.264 and H.265 sources.

    adaptive_app.py/adaptive_app.h hardcode H.264 decode. Concatenating
    rtsp_h264_urls with rtsp_h265_urls to reach the "at least two" threshold
    meant an environment with, say, one URL of each would feed an H.265 stream
    into an H.264-only decoder and fail on a fixture mismatch, not a real defect.
    """

    def test_python_e2e_requires_h264_only(self):
        source = (EXAMPLE_DIR / "tests" / "python" / "test_e2e.py").read_text(encoding="utf-8")
        assert "rtsp_h265_urls" not in source
        assert "rtsp_urls = rtsp_h264_urls" in source

    def test_cpp_e2e_requires_h264_only(self):
        source = (EXAMPLE_DIR / "tests" / "cpp" / "test_e2e.cpp").read_text(encoding="utf-8")
        assert "rtsp_h265_urls_from_env" not in source
        assert "rtsp_h264_urls_from_env()" in source


@requires_pipelines
class TestMemoryEstimateMatchesConfiguredBuffers:
    """The panel's decoder-memory estimate must charge the pool size the
    pipeline actually configures, not an unrelated guess.

    All three panels hardcoded n_visible=18 with a comment claiming "scale app
    runs decoder num_buffers=18" - but _config_scale() writes decoder_buffers: 8
    for scale/group, and the live pipeline never sets the key at all, so it
    runs on adaptive_app.py's own default of 4. The estimate overcharged scale/
    group by 10 buffers per stream and live by 14, and would have made
    ENFORCE_LIMITS (an intentional, switchable safety guard) reject
    configurations the app can actually run, had anyone flipped it on.
    """

    @pytest.mark.parametrize(
        "name,expected", [("scale", 8), ("group", 8), ("live", 4)]
    )
    def test_ui_server_uses_the_pipeline_s_own_decoder_buffers(self, name, expected):
        ui = _load_ui_server(name)
        assert ui.pipeline.DECODER_BUFFERS == expected

        # Recompute stream_pool_bytes' "visible" term independently, using the
        # EXPECTED buffer count, and confirm it is exactly what got charged -
        # not just that the constant has the right value in isolation.
        w, h = 1280, 720
        n_hidden = 24 if h <= 720 else 20
        hidden = n_hidden * (ui._align(w, 64) * ui._align(h, 64) * 3 // 2)
        visible = expected * (ui._align(w, 256) * ui._align(h, 64) * 3 // 2)
        inp = 2 * (w * h * 3 // 4)
        assert ui.stream_pool_bytes(w, h) == inp + hidden + visible

    @pytest.mark.parametrize("name", ["scale", "group", "live"])
    def test_no_pipeline_still_hardcodes_the_stale_estimate(self, name):
        source = (PIPELINES_DIR / f"pipeline-{name}" / "ui_server.py").read_text(encoding="utf-8")
        assert "n_visible = 18" not in source
        assert "pipeline.DECODER_BUFFERS" in source

    def test_scale_and_group_configs_still_render_the_same_value(self):
        """The constant must be read by the YAML template too, not just named
        for the estimate - otherwise the two could still drift independently."""
        for name in ("scale", "group"):
            mod = _load_ui_server(name).pipeline
            rendered = mod._config_scale(["rtsp://x/1"], "test")
            assert f"decoder_buffers: {mod.DECODER_BUFFERS}" in rendered


@requires_tools
class TestGenTestConfigModelDir:
    """A fresh install has no assets/models/ at all - models/ is where
    download_models.sh and the README put packs (see pipeline-scale/pipeline.py's
    _MODEL_DIRS). Every TESTING.md command invokes this generator without
    MODEL_DIR set, so defaulting to the old path produced a config pointing at
    a model that does not exist on a clean checkout.
    """

    @staticmethod
    def _run_resolution(tmp_path, apps_root_has, model_dir_env=None):
        """Extract gen_test_config.sh's own MODEL_DIR-resolution block and run
        it for real in a subprocess, against a fake APPS_ROOT - not a
        hand-copied duplicate of the logic, so a future edit that breaks it
        here fails this test.
        """
        source = (EXAMPLE_DIR / "tools" / "gen_test_config.sh").read_text(encoding="utf-8")
        start = source.index('APPS_ROOT="$(cd "$(dirname "$0")')
        end = source.index("\n\n{", start)
        block = source[start:end]
        assert "MODEL_DIR" in block

        apps_root = tmp_path / "apps"
        for name in apps_root_has:
            (apps_root / name).mkdir(parents=True)

        script = f'APPS_ROOT="{apps_root}"\n' + block.split("\n", 1)[1] + "\necho \"$MODEL_DIR\"\n"
        env = {"PATH": "/usr/bin:/bin"}
        if model_dir_env is not None:
            env["MODEL_DIR"] = model_dir_env
        result = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=env)
        assert result.returncode == 0, result.stderr
        return result.stdout.strip()

    def test_fresh_install_uses_models(self, tmp_path):
        resolved = self._run_resolution(tmp_path, apps_root_has=["models"])
        assert resolved == str(tmp_path / "apps" / "models")

    def test_old_checkout_falls_back_to_assets_models(self, tmp_path):
        resolved = self._run_resolution(tmp_path, apps_root_has=["assets/models"])
        assert resolved == str(tmp_path / "apps" / "assets" / "models")

    def test_neither_present_still_falls_back_without_erroring(self, tmp_path):
        resolved = self._run_resolution(tmp_path, apps_root_has=[])
        assert resolved == str(tmp_path / "apps" / "assets" / "models")

    def test_explicit_model_dir_overrides_both(self, tmp_path):
        resolved = self._run_resolution(
            tmp_path, apps_root_has=["models"], model_dir_env="/custom/path"
        )
        assert resolved == "/custom/path"


class TestGracefulSigterm:
    """SIGTERM is the normal stop signal from every panel and CLI `down`.

    Unhandled, it terminates the process outright: manager.shutdown() /
    run.close() never run, the decoder and CVU pools they would have released
    stay allocated in the reserved region, and the caller sees the PID vanish
    and reports a clean stop - the exact failure stop_app()'s own docstring
    warns about.
    """

    def test_a_real_sigterm_reaches_the_handler_instead_of_killing(self, tmp_path):
        """Spawns a process, signals it for real, and checks it exits 0 through
        the handler rather than 143 (killed). Nothing about this is mocked."""
        probe = tmp_path / "probe.py"
        probe.write_text(
            "import signal, sys, time\n"
            f"sys.path.insert(0, {str(PYTHON_DIR)!r})\n"
            "import fused_app\n"
            "fused_app._stop_requested = False\n"
            "signal.signal(signal.SIGTERM, fused_app._request_stop)\n"
            "print('READY', flush=True)\n"
            "for _ in range(200):\n"
            "    if fused_app._stop_requested:\n"
            "        print('GRACEFUL', flush=True); sys.exit(0)\n"
            "    time.sleep(0.05)\n"
            "sys.exit(1)\n",
            encoding="utf-8",
        )
        proc = subprocess.Popen(
            [sys.executable, str(probe)], stdout=subprocess.PIPE, text=True
        )
        try:
            assert proc.stdout.readline().strip() == "READY"
            proc.send_signal(signal.SIGTERM)
            out = proc.stdout.read()
            rc = proc.wait(timeout=20)
        finally:
            if proc.poll() is None:
                proc.kill()
        assert rc == 0, f"exited {rc} (143 means SIGTERM killed it outright)"
        assert "GRACEFUL" in out

    @pytest.mark.parametrize("module", ["adaptive_app.py", "fused_app.py"])
    def test_python_run_app_installs_a_sigterm_handler(self, module):
        source = (PYTHON_DIR / module).read_text(encoding="utf-8")
        assert "signal.signal(signal.SIGTERM" in source

    @pytest.mark.parametrize("header", ["adaptive_app.h", "fused_app.h"])
    def test_cpp_run_app_installs_a_sigterm_handler(self, header):
        source = (EXAMPLE_DIR / "src" / "cpp" / header).read_text(encoding="utf-8")
        assert "std::signal(SIGTERM, request_stop)" in source
        assert "std::signal(SIGTERM, previous_sigterm)" in source, "must be restored"


@requires_pipelines
class TestForeignDetectorStopWaits:
    """Killing a detector before its SIGTERM teardown finishes strands the pools
    it was releasing. Now that SIGTERM is actually handled, a fixed two-second
    sleep is no longer long enough - these paths must poll like stop_app() does.
    """

    @pytest.mark.parametrize("name", ["scale", "live", "group"])
    def test_pipelines_wait_rather_than_sleeping_two_seconds(self, name):
        source = (PIPELINES_DIR / f"pipeline-{name}" / "pipeline.py").read_text(encoding="utf-8")
        assert "true; sleep 2; " not in source, "unconditional SIGKILL after 2s"
        assert "_term_then_kill" in source

    def test_launcher_waits_rather_than_sleeping_two_seconds(self):
        source = (PIPELINES_DIR / "launcher.py").read_text(encoding="utf-8")
        assert "true; sleep 2; " not in source, "unconditional SIGKILL after 2s"
        assert "def stop_detector" in source

    def test_the_wait_polls_for_the_full_grace_period(self):
        """A helper that polls but only once would pass a substring check."""
        source = (PIPELINES_DIR / "pipeline-scale" / "pipeline.py").read_text(encoding="utf-8")
        helper = source[source.index("def _term_then_kill"):]
        helper = helper[: helper.index("\ndef ")]
        assert "for _ in range(grace_s)" in helper
        assert "time.sleep(1)" in helper
        assert "kill -9" in helper, "must still force-kill a genuinely hung process"


@requires_pipelines
class TestShellPathsAreQuoted:
    """Paths all derive from __file__, so a clone path containing whitespace or
    a shell metacharacter would split these commands into the wrong tokens."""

    @pytest.mark.parametrize("name", ["scale", "live", "group"])
    def test_start_commands_quote_interpolated_paths(self, name):
        source = (PIPELINES_DIR / f"pipeline-{name}" / "pipeline.py").read_text(encoding="utf-8")
        assert "import shlex" in source
        assert "shlex.quote(PYTHON)" in source
        assert "setsid nohup {app_command()}" in source
        # The bare, unquoted forms must not come back.
        assert "rm -f {LOG};" not in source
        assert "rm -f {log};" not in source
        assert "--config {CONFIG}" not in source

    def test_launcher_quotes_the_ui_script_path(self):
        source = (PIPELINES_DIR / "launcher.py").read_text(encoding="utf-8")
        assert "import shlex" in source
        assert 'bash "{ui}" start' not in source
        assert "shlex.quote(str(ui))" in source


@requires_pipelines
class TestStopRacesInFlightOperation:
    """Stop deliberately bypasses the job queue so it stays responsive during a
    long rebuild - but that let it race the very operation it interrupted.

    Sequence: the operator hits Add, its worker stages sources and is about to
    start the detector; the operator hits Stop, which stops everything and
    clears the saved list; the worker then resumes, starts the detector and
    re-saves its stale list. The panel shows "pipeline stopped" over a pipeline
    that is back up with the pre-Stop streams.
    """

    @pytest.mark.parametrize("name", ["scale", "live", "group"])
    def test_a_stop_mid_operation_wins(self, name, monkeypatch):
        ui = _load_ui_server(name)
        started = threading.Event()
        release = threading.Event()
        state = {"running": False, "streams": ["cam-1", "cam-2"]}

        def slow_add():
            started.set()
            release.wait(timeout=10)      # Stop happens during this window
            state["running"] = True       # the worker brings the pipeline back up
            state["streams"] = ["cam-1", "cam-2", "cam-3"]

        def fake_stop():
            state["running"] = False

        # Route both the worker's undo and Stop's own teardown at our fake state.
        if name == "group":
            monkeypatch.setattr(ui.pipeline, "stop_all_groups", fake_stop)
            monkeypatch.setattr(ui, "save_groups", lambda v: state.update(streams=list(v)))
        else:
            monkeypatch.setattr(ui.pipeline, "stop_app", fake_stop)
            monkeypatch.setattr(ui, "save_streams", lambda v: state.update(streams=list(v)))

        assert ui.submit(slow_add, "add cam-3") is True
        assert started.wait(timeout=10)

        ui.begin_stop()               # what POST /api/down does first
        fake_stop()
        state["streams"] = []

        release.set()
        for _ in range(200):          # let the worker finish and reconcile
            if not ui.STATUS["busy"]:
                break
            time.sleep(0.05)

        assert ui.STATUS["busy"] is False
        assert state["running"] is False, "the interrupted worker restarted the pipeline"
        assert state["streams"] == [], "the interrupted worker restored its stale stream list"
        assert ui.STATUS["message"] == "pipeline stopped"

    @pytest.mark.parametrize("name", ["scale", "live", "group"])
    def test_an_uninterrupted_operation_still_reports_done(self, name, monkeypatch):
        """The guard must not fire when no Stop happened."""
        ui = _load_ui_server(name)
        if name == "group":
            monkeypatch.setattr(ui.pipeline, "stop_all_groups", lambda: None)
            monkeypatch.setattr(ui, "save_groups", lambda v: None)
        else:
            monkeypatch.setattr(ui.pipeline, "stop_app", lambda: None)
            monkeypatch.setattr(ui, "save_streams", lambda v: None)

        done = threading.Event()
        assert ui.submit(done.set, "add cam-3") is True
        assert done.wait(timeout=10)
        for _ in range(200):
            if not ui.STATUS["busy"]:
                break
            time.sleep(0.05)
        assert ui.STATUS["message"] == "done: add cam-3"

    @pytest.mark.parametrize("name", ["scale", "live", "group"])
    def test_down_invalidates_before_doing_any_work(self, name):
        """begin_stop() must run before the teardown thread starts, or a worker
        finishing in between would still see a fresh token."""
        source = (PIPELINES_DIR / f"pipeline-{name}" / "ui_server.py").read_text(encoding="utf-8")
        down = source.index('if path == "/api/down":')
        assert source.index("begin_stop()", down) < source.index("def do_down():", down)


class TestFusedDecoderDefaults:
    """The declared default and the loader fallback must be the same profile.

    fused_app declared 18/auto with a comment defending it, while
    load_app_config() hardcoded 4/throughput-low-latency - and since the loader
    always passes explicit values, the declared default was dead code and
    everyone got the 4-buffer, memory_opt-ON profile the comment blames for
    stutter and freezes under jitter. Both now resolve to 8/auto, matching what
    pipelines/pipeline-{scale,group} generate and run at 16 streams.
    """

    def test_declared_default_is_what_an_omitted_key_actually_gets(self, tmp_path):
        import fused_app

        cfg_path = tmp_path / "fused.yaml"
        cfg_path.write_text(
            "model:\n  path: /tmp/m.tar.gz\n  labels: /tmp/l.txt\n"
            "streams:\n  - rtsp://127.0.0.1:8554/src1\n"
            "output:\n  insight:\n    host: 127.0.0.1\n",
            encoding="utf-8",
        )
        cfg = fused_app.load_app_config(cfg_path)
        assert cfg.decoder_buffers == fused_app.DEFAULT_DECODER_BUFFERS == 8
        assert cfg.decoder_tuning == fused_app.DEFAULT_DECODER_TUNING == "auto"

    def test_the_unstable_profile_is_gone_from_both_languages(self):
        py = (EXAMPLE_DIR / "src" / "python" / "fused_app.py").read_text(encoding="utf-8")
        cpp = (EXAMPLE_DIR / "src" / "cpp" / "fused_app.h").read_text(encoding="utf-8")
        assert '"decoder_tuning", "throughput-low-latency"' not in py
        assert '"input.decoder_tuning", "throughput-low-latency"' not in cpp
        assert "kDefaultDecoderBuffers = 8" in cpp
        assert 'kDefaultDecoderTuning = "auto"' in cpp

    def test_an_explicit_config_value_still_wins(self, tmp_path):
        """pipelines/ sets these keys explicitly; that must be unaffected."""
        import fused_app

        cfg_path = tmp_path / "fused.yaml"
        cfg_path.write_text(
            "model:\n  path: /tmp/m.tar.gz\n  labels: /tmp/l.txt\n"
            "streams:\n  - rtsp://127.0.0.1:8554/src1\n"
            "input:\n  decoder_buffers: 18\n  decoder_tuning: low-memory\n"
            "output:\n  insight:\n    host: 127.0.0.1\n",
            encoding="utf-8",
        )
        cfg = fused_app.load_app_config(cfg_path)
        assert cfg.decoder_buffers == 18
        assert cfg.decoder_tuning == "low-memory"


class TestReloadHonoursMaxStreams:
    """Startup rejects an over-limit file; a live reload used to accept it
    partially - starting what fitted and warning per stream about the rest."""

    @staticmethod
    def _config(tmp_path, count, max_streams):
        sources = "\n".join(
            f"    - id: cam-{i + 1}\n      rtsp_url: rtsp://127.0.0.1:8554/src{i + 1}"
            for i in range(count)
        )
        path = tmp_path / "live.yaml"
        path.write_text(
            "model:\n  path: /tmp/m.tar.gz\n"
            f"streams:\n  max_streams: {max_streams}\n  sources:\n{sources}\n"
            "output:\n  insight:\n    host: 127.0.0.1\n",
            encoding="utf-8",
        )
        return path

    def test_a_reload_within_the_limit_is_accepted(self, tmp_path):
        import adaptive_app

        path = self._config(tmp_path, count=3, max_streams=8)
        assert len(adaptive_app.reload_sources(path, 8)) == 3

    def test_a_reload_over_the_files_own_maximum_is_rejected(self, tmp_path):
        import adaptive_app

        path = self._config(tmp_path, count=9, max_streams=8)
        with pytest.raises(ValueError, match="max_streams"):
            adaptive_app.reload_sources(path, 8)

    def test_max_streams_cannot_be_raised_by_a_live_edit(self, tmp_path):
        """Channels are allocated once at startup, so a raised ceiling in the
        edited file cannot conjure more of them."""
        import adaptive_app

        path = self._config(tmp_path, count=12, max_streams=16)
        with pytest.raises(ValueError, match="channels this run started with"):
            adaptive_app.reload_sources(path, 8)

    def test_cpp_reload_applies_the_same_check(self):
        cpp = (EXAMPLE_DIR / "src" / "cpp" / "adaptive_app.h").read_text(encoding="utf-8")
        reload_site = cpp[cpp.index("[config] reload:") - 900:]
        assert "channels this run started with" in reload_site


class TestPassthroughDisablesExactTimestamps:
    """Encoded passthrough and an exact rtp_timestamp cannot both work.

    Passthrough forwards the source's own H.264 through a payloader whose RTP
    base is random (VideoSenderOptions.rtp exposes no timestamp-offset in
    pyneat - see send_metadata in fused_app.py, which declines the key for
    exactly this reason). A PTS-derived key then matches nothing, and Insight
    stops falling back to arrival order the moment the field is present, so it
    renders NO boxes. That is the shipped default's combination:
    encoded_passthrough true + metadata_rtp_timestamp auto.
    """

    def test_shipped_defaults_are_the_affected_combination(self):
        """If either default changes, this test should be revisited."""
        import adaptive_app

        cfg_path = EXAMPLE_DIR / "src" / "common" / "config.yaml"
        text = cfg_path.read_text(encoding="utf-8")
        assert "encoded_passthrough" not in text
        assert "metadata_rtp_timestamp" not in text
        fields = {f.name: f for f in dataclasses.fields(adaptive_app.AppConfig)}
        assert fields["encoded_passthrough"].default is True
        assert fields["metadata_rtp_timestamp"].default == "auto"

    def test_auto_declines_the_key_under_passthrough(self):
        source = (EXAMPLE_DIR / "src" / "python" / "adaptive_app.py").read_text(encoding="utf-8")
        assert "not restamped and not passthrough" in source, (
            "auto must decline the exact key when passthrough is on"
        )

    def test_explicit_on_still_wins_but_warns(self):
        source = (EXAMPLE_DIR / "src" / "python" / "adaptive_app.py").read_text(encoding="utf-8")
        block = source[source.index('if cfg.metadata_rtp_timestamp == "on":'):]
        block = block[: block.index("elif ")]
        assert "runtime.emit_rtp_timestamp = True" in block
        assert "random" in block, "an explicit 'on' under passthrough must warn"


class TestRemovedStreamHoldsItsChannel:
    """A worker mid-build does not see stop_event until the build returns, and a
    build takes 30-90s. Releasing its channel on the 15s join timeout let the
    next add bind the same Insight video/metadata ports as a worker that was
    still running."""

    @staticmethod
    def _manager(tmp_path, monkeypatch, init):
        import adaptive_app

        monkeypatch.setattr(adaptive_app, "init_stream_runtime", init)
        cfg = adaptive_app.load_app_config(write_config(tmp_path, RICH_TWO))
        return adaptive_app, adaptive_app.StreamManager(cfg, ["person"])

    def test_channel_is_not_reused_while_the_worker_is_still_building(self, tmp_path, monkeypatch):
        building = threading.Event()
        finish = threading.Event()

        def slow_init(cfg, channel, source, labels):
            building.set()
            finish.wait(timeout=30)       # simulates a 30-90s build
            raise RuntimeError("aborted after the build")

        adaptive_app, manager = self._manager(tmp_path, monkeypatch, slow_init)
        try:
            manager.add(adaptive_app.StreamSource("cam-1", "rtsp://127.0.0.1:8554/src1"))
            assert building.wait(timeout=10)
            taken = manager.streams["cam-1"].channel

            # Short timeout: the point is the TIMEOUT path, not waiting out 15s.
            manager.remove("cam-1", join_timeout=0.2)
            assert taken not in manager.free_channels, (
                "channel reused while its worker was still building"
            )

            finish.set()
            assert _wait_for(lambda: taken in manager.free_channels), (
                "channel never came back after the worker exited"
            )
        finally:
            finish.set()
            manager.shutdown()

    def test_a_worker_that_has_already_exited_releases_immediately(self, tmp_path, monkeypatch):
        adaptive_app, manager = self._manager(tmp_path, monkeypatch, _fake_init_stream_runtime)
        try:
            manager.add(adaptive_app.StreamSource("cam-1", "rtsp://127.0.0.1:8554/src1"))
            assert _wait_for(lambda: manager.active_count() == 1)
            taken = manager.streams["cam-1"].channel
            manager.remove("cam-1")
            assert taken in manager.free_channels
        finally:
            manager.shutdown()


class TestFusedRunMarker:
    """wait_for_streams() counted per-stream banners that the fused app prints
    BEFORE building the shared graph, so a build failure reported success."""

    @pytest.mark.parametrize("path", ["src/python/fused_app.py", "src/cpp/fused_app.h"])
    def test_marker_is_emitted_after_the_build(self, path):
        source = (EXAMPLE_DIR / path).read_text(encoding="utf-8")
        build = source.index("graph.build(build_run_options(")
        marker = source.index("[app] graph running", build)
        assert marker > build, "the marker must follow the build, not precede it"

    @requires_pipelines
    @pytest.mark.parametrize("name", ["scale", "live", "group"])
    def test_wait_requires_the_marker_for_fused(self, name):
        source = (PIPELINES_DIR / f"pipeline-{name}" / "pipeline.py").read_text(encoding="utf-8")
        assert '"[app] graph running" in text' in source
        # The bare banner-only check must be gone.
        assert 'if text.count("] rtsp=") >= n:\n            return True' not in source


class TestCppRejectsUnsupportedPassthrough:
    """C++ adaptive has no passthrough topology - it re-encodes every stream.
    Silently doing the opposite of what the config asks is the defect; it now
    refuses an explicit `true` instead."""

    def test_cpp_reads_and_rejects_the_key(self):
        cpp = (EXAMPLE_DIR / "src" / "cpp" / "adaptive_app.h").read_text(encoding="utf-8")
        assert 'raw.bool_or("output.encoded_passthrough", false)' in cpp
        assert "not supported by the C++ implementation" in cpp

    def test_cpp_defaults_to_false_so_existing_configs_are_unaffected(self):
        cpp = (EXAMPLE_DIR / "src" / "cpp" / "adaptive_app.h").read_text(encoding="utf-8")
        assert "bool encoded_passthrough = false;" in cpp

    @requires_pipelines
    @pytest.mark.parametrize("name", ["scale", "live"])
    def test_the_shipped_pipelines_set_it_false(self, name):
        source = (PIPELINES_DIR / f"pipeline-{name}" / "pipeline.py").read_text(encoding="utf-8")
        assert "encoded_passthrough: false" in source


class TestDeferredAddIsRetried:
    """Holding a removed stream's channel must not silently drop the add.

    At capacity, a reload that replaces one stream removes the old one - whose
    channel is now held until its worker exits - and immediately adds the new
    one. That add finds no free channel. Without a retry the requested source is
    dropped for good, leaving the run one stream short of the config on disk.
    """

    def test_a_blocked_add_starts_once_the_channel_comes_back(self, tmp_path, monkeypatch):
        import adaptive_app

        building = threading.Event()
        finish = threading.Event()
        started: list[str] = []

        def init(cfg, channel, source, labels):
            started.append(source.id)
            if source.id == "cam-old":
                building.set()
                finish.wait(timeout=30)
                raise RuntimeError("aborted after the build")
            return _fake_init_stream_runtime(cfg, channel, source, labels)

        monkeypatch.setattr(adaptive_app, "init_stream_runtime", init)
        cfg = adaptive_app.load_app_config(write_config(tmp_path, RICH_TWO))
        object.__setattr__(cfg, "max_streams", 1)          # capacity of exactly one
        manager = adaptive_app.StreamManager(cfg, ["person"])
        try:
            manager.add(adaptive_app.StreamSource("cam-old", "rtsp://127.0.0.1:8554/old"))
            assert building.wait(timeout=10)

            # The reload swaps cam-old for cam-new at full capacity.
            manager.remove("cam-old", join_timeout=0.2)
            manager.apply_sources([adaptive_app.StreamSource("cam-new", "rtsp://127.0.0.1:8554/new")])
            assert "cam-new" not in started, "no channel was free; it cannot have started yet"
            assert any(p.id == "cam-new" for p in manager.pending_adds)

            finish.set()                                    # old worker exits, channel frees
            assert _wait_for(lambda: "cam-new" in started), (
                "the deferred add was never retried"
            )
            assert _wait_for(lambda: manager.active_count() == 1)
        finally:
            finish.set()
            manager.shutdown()

    def test_a_source_dropped_from_the_config_is_not_retried(self, tmp_path, monkeypatch):
        """A queued add must not resurrect after the config stops asking for it."""
        import adaptive_app

        monkeypatch.setattr(adaptive_app, "init_stream_runtime", _fake_init_stream_runtime)
        cfg = adaptive_app.load_app_config(write_config(tmp_path, RICH_TWO))
        manager = adaptive_app.StreamManager(cfg, ["person"])
        try:
            queued = adaptive_app.StreamSource("cam-x", "rtsp://127.0.0.1:8554/x")
            manager.pending_adds.append(queued)
            manager.apply_sources([])                        # config no longer names cam-x
            assert manager.pending_adds == []
        finally:
            manager.shutdown()


@requires_pipelines
class TestPanelSurfacesReadinessFailure:
    """wait_for_streams()/wait_for_group() became a trustworthy signal only once
    the post-build marker was required - and every caller was discarding it, so
    the panel still reported "done" over a detector that never came up."""

    @pytest.mark.parametrize("name", ["scale", "live"])
    def test_rebuild_raises_when_the_detector_is_not_ready(self, name, monkeypatch):
        ui = _load_ui_server(name)
        monkeypatch.setattr(ui, "plan", lambda s: (["rtsp://x/1"], [], ui.pipeline.TIERS[0]))
        monkeypatch.setattr(ui, "stage_insight", lambda staging: None)
        monkeypatch.setattr(ui, "save_streams", lambda v: None)
        monkeypatch.setattr(ui.pipeline, "write_config_urls", lambda *a, **k: None)
        monkeypatch.setattr(ui.pipeline, "start_app", lambda: None)
        monkeypatch.setattr(ui.pipeline, "wait_for_streams", lambda *a, **k: False)
        with pytest.raises(RuntimeError, match="ready"):
            ui.rebuild([{"kind": "external", "url": "rtsp://x/1"}])

    @pytest.mark.parametrize("name", ["scale", "live"])
    def test_rebuild_succeeds_when_it_is_ready(self, name, monkeypatch):
        ui = _load_ui_server(name)
        monkeypatch.setattr(ui, "plan", lambda s: (["rtsp://x/1"], [], ui.pipeline.TIERS[0]))
        monkeypatch.setattr(ui, "stage_insight", lambda staging: None)
        monkeypatch.setattr(ui, "save_streams", lambda v: None)
        monkeypatch.setattr(ui.pipeline, "write_config_urls", lambda *a, **k: None)
        monkeypatch.setattr(ui.pipeline, "start_app", lambda: None)
        monkeypatch.setattr(ui.pipeline, "wait_for_streams", lambda *a, **k: True)
        ui.rebuild([{"kind": "external", "url": "rtsp://x/1"}])


@requires_pipelines
class TestForeignKillForcesRuntimeReclaim:
    """A foreign detector killed after its grace period strands pools exactly as
    ours does, and once it is gone nothing can discover that later - so its
    result has to reach the reset_runtime() decision, not just a warning."""

    @pytest.mark.parametrize("name", ["scale", "live"])
    def test_start_app_resets_when_a_foreign_detector_was_killed(self, name, monkeypatch):
        mod = _load_ui_server(name).pipeline
        reset = []
        monkeypatch.setattr(mod, "stop_app", lambda: True)          # OUR stop was clean
        monkeypatch.setattr(mod, "stop_any_detector", lambda: False)  # a foreign one was killed
        monkeypatch.setattr(mod, "reset_runtime", lambda: reset.append(True))
        monkeypatch.setattr(mod, "exec_devkit", lambda *a, **k: None)
        mod.start_app()
        assert reset == [True], "a killed foreign detector must force a reclaim"

    @pytest.mark.parametrize("name", ["scale", "live"])
    def test_no_reset_when_everything_stopped_cleanly(self, name, monkeypatch):
        mod = _load_ui_server(name).pipeline
        reset = []
        monkeypatch.setattr(mod, "stop_app", lambda: True)
        monkeypatch.setattr(mod, "stop_any_detector", lambda: True)
        monkeypatch.setattr(mod, "reset_runtime", lambda: reset.append(True))
        monkeypatch.setattr(mod, "exec_devkit", lambda *a, **k: None)
        mod.start_app()
        assert reset == [], "a clean stop must not pay the ~60s reclaim"


class TestUnexpectedFusedClosureFails:
    """A continuous fused run that loses its shared output produces no further
    metadata; exiting 0 would tell a supervisor the experiment succeeded."""

    @pytest.mark.parametrize("path", ["src/python/fused_app.py", "src/cpp/fused_app.h"])
    def test_closure_is_reported_as_a_failure(self, path):
        source = (EXAMPLE_DIR / path).read_text(encoding="utf-8")
        assert "closed unexpectedly" in source
        # ...but only for a continuous run that nobody asked to stop.
        assert "frames <= 0" in source

    def test_a_requested_stop_is_still_a_clean_exit(self):
        source = (EXAMPLE_DIR / "src" / "python" / "fused_app.py").read_text(encoding="utf-8")
        block = source[source.index("if _output_closed_unexpectedly"):]
        assert "not _stop_requested" in block[: block.index("raise")]
