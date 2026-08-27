"""Unit tests for the adaptive-resolution-object-detector example."""

from __future__ import annotations

import json
from pathlib import Path
import importlib.util
import io
import os
import subprocess
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
    """The panel writes metadata_rtp_timestamp into the config both languages read."""

    def test_python_and_cpp_both_honour_the_setting(self):
        cpp = (EXAMPLE_DIR / "src" / "cpp" / "adaptive_app.h").read_text(encoding="utf-8")
        assert "output.insight.metadata_rtp_timestamp" in cpp
        assert "send_raw_json" in cpp, "the convenience API cannot carry rtp_timestamp"
        assert "do-timestamp=true" in cpp, "auto must read what Core actually lowered"

    def test_live_config_requests_it(self):
        live = (PIPELINES_DIR / "pipeline-live" / "pipeline.py").read_text(encoding="utf-8")
        assert 'metadata_rtp_timestamp: "on"' in live


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
        """More streams than GROUP_SIZE * MAX_GROUPS cannot fit; cmd_up() warns
        about this separately rather than group_plan() raising."""
        mod = self._pipeline()
        ceiling = mod.GROUP_SIZE * mod.MAX_GROUPS
        assert sum(mod.group_plan(ceiling + 5)) == ceiling

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
