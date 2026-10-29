"""Unit tests for adaptive-resolution-object-detector (Python)."""

from __future__ import annotations

import json
from pathlib import Path
import struct
import subprocess
import sys
from types import SimpleNamespace

import pytest

from tests.utils.config_cases import config_writer, load_example_main
from tests.utils.fake_run import FakeRun


EXAMPLE_DIR = Path(__file__).resolve().parent.parent.parent
MAIN_PY = EXAMPLE_DIR / "src" / "python" / "main.py"

pytestmark = pytest.mark.unit


main_module = load_example_main(EXAMPLE_DIR, "adaptive_resolution_object_detector_main")


VALID_CONFIG = {
    "mode": "live",
    "model": {"path": "model.tar.gz", "labels": "coco_label.txt"},
    "source": {"type": "rtsp", "codec": "h264", "tcp": True, "latency_ms": 100, "fps": 0},
    "streams": {
        "max_streams": 16,
        "sources": [
            {"id": "cam-1", "url": "rtsp://127.0.0.1:8554/src1"},
            {"id": "cam-2", "url": "rtsp://127.0.0.1:8554/src2"},
        ],
    },
    "group": {"size": 4},
    "inference": {"frames": 0, "fps": 0, "min_score": 0.30, "nms_iou": 0.60, "max_detections": 50},
    "runtime": {"profile": False, "warmup_frames": 30, "config_watch_seconds": 1.0},
    "output": {
        "save_dir": "",
        "save_every": 0,
        "video_enabled": True,
        "insight": {"host": "127.0.0.1", "video_port": 9000, "metadata_port": 9100},
    },
}

write_full_config = config_writer(VALID_CONFIG)


def sources(count: int) -> list[dict]:
    return [{"id": f"cam-{i + 1}", "url": f"rtsp://127.0.0.1:8554/src{i + 1}"} for i in range(count)]


def run_main(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(MAIN_PY), *args],
        capture_output=True,
        text=True,
        cwd=str(EXAMPLE_DIR),
        timeout=20,
    )


class TestMainEntrypoint:
    def test_help_lists_config_options_and_no_mode_flag(self):
        result = run_main("--help")

        assert result.returncode == 0
        assert "--config" in result.stdout
        assert "--validate-config-only" in result.stdout
        assert "--mode" not in result.stdout

    def test_mode_flag_is_rejected(self, tmp_path):
        result = run_main("--mode", "scale", "--config", str(write_full_config(tmp_path)))

        assert result.returncode != 0
        assert "--mode" in result.stderr

    def test_missing_config_file_fails_cleanly(self, tmp_path):
        result = run_main("--config", str(tmp_path / "missing.yaml"))

        assert result.returncode == 2
        assert "config file not found" in result.stderr

    @pytest.mark.parametrize("mode", ["live", "scale", "group"])
    def test_validate_config_only_reports_the_selected_mode(self, tmp_path, mode):
        config_path = write_full_config(tmp_path, [(("mode",), mode)])

        result = run_main("--config", str(config_path), "--validate-config-only")

        assert result.returncode == 0, result.stderr
        assert f"mode={mode}" in result.stdout
        assert "streams=2" in result.stdout

    def test_invalid_config_exits_with_an_error_line(self, tmp_path):
        config_path = write_full_config(tmp_path, [(("mode",), "fused")])

        result = run_main("--config", str(config_path), "--validate-config-only")

        assert result.returncode == 1
        assert "[ERR] mode must be one of: live, scale, group" in result.stderr

    def test_packaged_config_validates(self):
        result = run_main("--config", str(EXAMPLE_DIR / "src" / "common" / "config.yaml"), "--validate-config-only")

        assert result.returncode == 0, result.stderr
        assert "mode=live" in result.stdout


class TestValidBaseline:
    def test_the_baseline_config_loads(self, tmp_path):
        cfg = main_module.load_app_config(write_full_config(tmp_path))

        assert cfg.mode == "live"
        assert [s.id for s in cfg.sources] == ["cam-1", "cam-2"]
        assert cfg.sources[0].url == "rtsp://127.0.0.1:8554/src1"
        assert cfg.channel_base == 0

    def test_omitted_optional_values_fall_back_to_documented_defaults(self, tmp_path):
        minimal = {
            "mode": "scale",
            "model": {"path": "model.tar.gz"},
            "streams": {"sources": sources(1)},
            "output": {"insight": {"host": "127.0.0.1"}},
        }
        cfg = main_module.load_app_config(write_full_config(tmp_path, root=minimal))

        assert cfg.max_streams == 16
        assert cfg.group_size == 4
        assert (cfg.source_type, cfg.source_codec) == ("rtsp", "h264")
        assert (cfg.tcp, cfg.latency_ms, cfg.source_fps) == (True, 100, 0)
        assert (cfg.frames, cfg.fps) == (0, 0)
        assert (cfg.min_score, cfg.nms_iou, cfg.max_detections) == (0.30, 0.60, 50)
        assert (cfg.profile, cfg.warmup_frames, cfg.config_watch_seconds) == (False, 30, 1.0)
        assert (cfg.video_port, cfg.metadata_port, cfg.video_enabled) == (9000, 9100, True)
        assert (cfg.save_dir, cfg.save_every) == ("", 0)
        assert cfg.labels_path.name == "coco_label.txt"

    @pytest.mark.parametrize("codec", ["h264", "H264", "avc", "h.264"])
    def test_h264_aliases_are_accepted(self, tmp_path, codec):
        cfg = main_module.load_app_config(write_full_config(tmp_path, [(("source", "codec"), codec)]))

        assert cfg.source_codec == "h264"

    @pytest.mark.parametrize("mode", ["LIVE", " scale ", "Group"])
    def test_mode_is_case_and_space_insensitive(self, tmp_path, mode):
        cfg = main_module.load_app_config(write_full_config(tmp_path, [(("mode",), mode)]))

        assert cfg.mode == mode.strip().lower()


class TestRejectedValues:
    @pytest.mark.parametrize(
        ("path", "value", "message"),
        [
            (("mode",), "", "mode must be set"),
            (("mode",), "adaptive", "mode must be one of"),
            (("mode",), 3, "mode must be a string"),
            (("model", "path"), "", "model.path must be set"),
            (("model", "labels"), "", "model.labels must be set"),
            (("source", "type"), "http", "source.type must be rtsp"),
            (("source", "codec"), "h265", "source.codec must be h264"),
            (("source", "codec"), "mjpeg", "source.codec must be h264"),
            (("source", "latency_ms"), -1, "source.latency_ms must be >= 0"),
            (("source", "fps"), -1, "source.fps must be >= 0"),
            (("source", "tcp"), "yes", "tcp must be true or false"),
            (("streams", "max_streams"), 0, "streams.max_streams must be >= 1"),
            (("streams", "max_streams"), 1, "more than streams.max_streams"),
            (("group", "size"), 0, "group.size must be >= 1"),
            (("inference", "frames"), -1, "inference.frames must be >= 0"),
            (("inference", "fps"), -1, "inference.fps must be >= 0"),
            (("inference", "min_score"), 1.5, "inference.min_score must be between 0 and 1"),
            (("inference", "nms_iou"), -0.1, "inference.nms_iou must be between 0 and 1"),
            (("inference", "max_detections"), 0, "inference.max_detections must be > 0"),
            (("runtime", "warmup_frames"), -1, "runtime.warmup_frames must be >= 0"),
            (("runtime", "config_watch_seconds"), 0, "runtime.config_watch_seconds must be > 0"),
            (("output", "save_every"), -1, "output.save_every must be >= 0"),
            (("output", "insight", "host"), "", "output.insight.host must be set"),
            (("output", "insight", "video_port"), 0, "output.insight.video_port must be > 0"),
            (("output", "insight", "metadata_port"), 0, "output.insight.metadata_port must be > 0"),
        ],
    )
    def test_invalid_value_is_rejected_with_an_actionable_message(self, tmp_path, path, value, message):
        with pytest.raises(ValueError, match=message):
            main_module.load_app_config(write_full_config(tmp_path, [(path, value)]))

    def test_missing_mode_is_rejected(self, tmp_path):
        raw = {key: value for key, value in VALID_CONFIG.items() if key != "mode"}

        with pytest.raises(ValueError, match="mode must be set to one of: live, scale, group"):
            main_module.load_app_config(write_full_config(tmp_path, root=raw))

    def test_a_non_mapping_root_is_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="config root must be a mapping"):
            main_module.load_app_config(write_full_config(tmp_path, root=["mode", "live"]))

    def test_an_empty_file_is_reported_as_missing_settings_not_a_crash(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text("", encoding="utf-8")

        with pytest.raises(ValueError, match="mode must be set"):
            main_module.load_app_config(config_path)


class TestStreamList:
    def test_bare_url_list_is_rejected(self, tmp_path):
        config_path = write_full_config(tmp_path, [(("streams",), ["rtsp://127.0.0.1:8554/src1"])])

        with pytest.raises(ValueError, match="streams must be a mapping"):
            main_module.load_app_config(config_path)

    def test_sources_must_be_a_list(self, tmp_path):
        config_path = write_full_config(tmp_path, [(("streams", "sources"), "rtsp://127.0.0.1/a")])

        with pytest.raises(ValueError, match="streams.sources must be a list"):
            main_module.load_app_config(config_path)

    @pytest.mark.parametrize(
        ("entry", "message"),
        [
            ("rtsp://127.0.0.1:8554/src1", r"streams.sources\[1\] must be a mapping"),
            ({"url": "rtsp://127.0.0.1:8554/src1"}, r"streams.sources\[1\].id must be set"),
            ({"id": "cam-9", "url": "  "}, r"streams.sources\[1\].url must be set"),
            ({"id": "cam-9", "rtsp_url": "rtsp://a/b"}, r"streams.sources\[1\].url must be set"),
        ],
    )
    def test_a_bad_entry_names_its_index(self, tmp_path, entry, message):
        config_path = write_full_config(tmp_path, [(("streams", "sources"), [sources(1)[0], entry])])

        with pytest.raises(ValueError, match=message):
            main_module.load_app_config(config_path)

    def test_duplicate_ids_are_rejected(self, tmp_path):
        duplicate = [sources(1)[0], {"id": "cam-1", "url": "rtsp://127.0.0.1:8554/other"}]
        config_path = write_full_config(tmp_path, [(("streams", "sources"), duplicate)])

        with pytest.raises(ValueError, match="ids must be unique"):
            main_module.load_app_config(config_path)

    def test_live_mode_may_start_with_no_streams(self, tmp_path):
        cfg = main_module.load_app_config(write_full_config(tmp_path, [(("streams", "sources"), [])]))

        assert cfg.sources == []

    @pytest.mark.parametrize("mode", ["scale", "group"])
    def test_scale_and_group_need_at_least_one_stream(self, tmp_path, mode):
        config_path = write_full_config(tmp_path, [(("mode",), mode), (("streams", "sources"), [])])

        with pytest.raises(ValueError, match=f"must not be empty in {mode} mode"):
            main_module.load_app_config(config_path)

    def test_stream_order_is_preserved(self, tmp_path):
        config_path = write_full_config(tmp_path, [(("streams", "sources"), list(reversed(sources(3))))])

        cfg = main_module.load_app_config(config_path)

        assert [s.id for s in cfg.sources] == ["cam-3", "cam-2", "cam-1"]


class TestLiveReload:
    def test_a_valid_edit_returns_the_new_sources(self, tmp_path):
        config_path = write_full_config(tmp_path, [(("streams", "sources"), sources(3))])

        reloaded = main_module.reload_sources(config_path, capacity=16)

        assert [s.id for s in reloaded] == ["cam-1", "cam-2", "cam-3"]

    def test_an_edit_past_the_startup_capacity_is_rejected_whole(self, tmp_path):
        config_path = write_full_config(
            tmp_path, [(("streams", "max_streams"), 32), (("streams", "sources"), sources(5))]
        )

        with pytest.raises(ValueError, match="more than the 4 channels this run started with"):
            main_module.reload_sources(config_path, capacity=4)

    def test_an_invalid_edit_raises_instead_of_applying_part_of_it(self, tmp_path):
        duplicate = [sources(1)[0], sources(1)[0]]
        config_path = write_full_config(tmp_path, [(("streams", "sources"), duplicate)])

        with pytest.raises(ValueError, match="ids must be unique"):
            main_module.reload_sources(config_path, capacity=16)


class RecordingManager(main_module.StreamManager):
    """A StreamManager whose add/remove record calls instead of starting threads."""

    def __init__(self, cfg, running: dict[str, str]):
        super().__init__(cfg, labels=["person"])
        self.added: list[str] = []
        self.removed: list[str] = []
        for channel, (stream_id, url) in enumerate(running.items()):
            self.streams[stream_id] = main_module.ManagedStream(stream_id, url, channel, None)
            self.free_channels.remove(channel)

    def add(self, source):
        self.added.append(source.id)
        self.streams[source.id] = main_module.ManagedStream(source.id, source.url, 0, None)
        return True

    def remove(self, stream_id, join_timeout=15.0):
        self.removed.append(stream_id)
        self.streams.pop(stream_id, None)


class TestLiveStreamDiff:
    @staticmethod
    def _cfg(tmp_path):
        return main_module.load_app_config(write_full_config(tmp_path))

    def test_only_new_and_removed_ids_change(self, tmp_path):
        manager = RecordingManager(
            self._cfg(tmp_path), {"cam-1": "rtsp://a/1", "cam-2": "rtsp://a/2"}
        )

        manager.apply_sources(
            [main_module.StreamSource("cam-2", "rtsp://a/2"), main_module.StreamSource("cam-3", "rtsp://a/3")]
        )

        assert manager.removed == ["cam-1"]
        assert manager.added == ["cam-3"]

    def test_a_changed_url_restarts_that_stream(self, tmp_path):
        manager = RecordingManager(self._cfg(tmp_path), {"cam-1": "rtsp://a/1"})

        manager.apply_sources([main_module.StreamSource("cam-1", "rtsp://a/new")])

        assert manager.removed == ["cam-1"]
        assert manager.added == ["cam-1"]

    def test_an_unchanged_config_touches_nothing(self, tmp_path):
        manager = RecordingManager(self._cfg(tmp_path), {"cam-1": "rtsp://a/1"})

        manager.apply_sources([main_module.StreamSource("cam-1", "rtsp://a/1")])

        assert manager.removed == []
        assert manager.added == []

    def test_a_dropped_source_stops_counting_as_failed(self, tmp_path):
        manager = RecordingManager(self._cfg(tmp_path), {})
        manager.failed["cam-9"] = "source closed"

        manager.apply_sources([])

        assert manager.failed == {}


class TestGroupSplit:
    def _cfg(self, tmp_path, count: int, size: int):
        return main_module.load_app_config(
            write_full_config(
                tmp_path,
                [(("mode",), "group"), (("group", "size"), size), (("streams", "sources"), sources(count))],
            )
        )

    def test_streams_are_chunked_in_order_with_their_channel_range(self, tmp_path):
        groups = main_module.split_groups(self._cfg(tmp_path, count=10, size=4))

        assert [[s.id for s in g.sources] for g in groups] == [
            ["cam-1", "cam-2", "cam-3", "cam-4"],
            ["cam-5", "cam-6", "cam-7", "cam-8"],
            ["cam-9", "cam-10"],
        ]
        assert [g.channel_base for g in groups] == [0, 4, 8]

    def test_fewer_streams_than_a_group_make_one_group(self, tmp_path):
        groups = main_module.split_groups(self._cfg(tmp_path, count=3, size=4))

        assert len(groups) == 1
        assert groups[0].channel_base == 0
        assert len(groups[0].sources) == 3

    def test_groups_keep_every_other_setting(self, tmp_path):
        cfg = self._cfg(tmp_path, count=5, size=2)

        for group in main_module.split_groups(cfg):
            assert group.insight_host == cfg.insight_host
            assert group.model_path == cfg.model_path
            assert group.mode == "group"


class FakeProcess:
    def __init__(self, target, args, name, outcome, log):
        self.target = target
        self.args = args
        self.name = name
        self.pid = 1000 + len(log)
        self.exitcode = None
        self.terminated = False
        self._outcome = outcome
        self._log = log

    @property
    def group_index(self):
        return self.args[1]

    @property
    def stream_ids(self):
        return [s.id for s in self.args[0].sources]

    def start(self):
        self._log.append(("start", self.group_index, self.stream_ids))
        _, _, ready = self.args
        if self._outcome == "ready":
            ready.set()
        else:
            self.exitcode = 1

    def is_alive(self):
        return self.exitcode is None

    def terminate(self):
        self._log.append(("stop", self.group_index, self.stream_ids))
        self.terminated = True
        self.exitcode = 0

    def kill(self):
        self.exitcode = -9

    def join(self, timeout=None):
        pass


class FakeEvent:
    def __init__(self):
        self._set = False

    def set(self):
        self._set = True

    def wait(self, timeout=None):
        return self._set


class FakeContext:
    """Starts fake group processes; each takes the next outcome, default "ready"."""

    def __init__(self, outcomes=()):
        self.outcomes = list(outcomes)
        self.processes: list[FakeProcess] = []
        self.log: list[tuple] = []

    def Event(self):
        return FakeEvent()

    def Process(self, target, args, name):
        outcome = self.outcomes.pop(0) if self.outcomes else "ready"
        process = FakeProcess(target, args, name, outcome, self.log)
        self.processes.append(process)
        return process


class TestChangedGroups:
    @staticmethod
    def _groups(tmp_path, count, size=4):
        cfg = main_module.load_app_config(
            write_full_config(
                tmp_path,
                [(("mode",), "group"), (("group", "size"), size), (("streams", "sources"), sources(count))],
            )
        )
        return main_module.split_groups(cfg)

    def test_adding_a_stream_to_the_last_group_changes_only_that_group(self, tmp_path):
        running = dict(enumerate(self._groups(tmp_path, 5)))

        assert main_module.changed_groups(running, self._groups(tmp_path, 6)) == [1]

    def test_a_stream_that_needs_a_new_group_changes_only_the_new_group(self, tmp_path):
        running = dict(enumerate(self._groups(tmp_path, 4)))

        assert main_module.changed_groups(running, self._groups(tmp_path, 5)) == [1]

    def test_emptying_the_last_group_removes_it(self, tmp_path):
        running = dict(enumerate(self._groups(tmp_path, 5)))

        assert main_module.changed_groups(running, self._groups(tmp_path, 4)) == [1]

    def test_an_unchanged_config_changes_nothing(self, tmp_path):
        running = dict(enumerate(self._groups(tmp_path, 6)))

        assert main_module.changed_groups(running, self._groups(tmp_path, 6)) == []

    def test_removing_an_early_stream_shifts_every_later_group(self, tmp_path):
        running = dict(enumerate(self._groups(tmp_path, 9, size=2)))
        wanted = self._groups(tmp_path, 9, size=2)
        # Drop cam-2: every stream after it moves up one slot.
        trimmed = [s for s in wanted[0].sources + wanted[1].sources + wanted[2].sources
                   + wanted[3].sources + wanted[4].sources if s.id != "cam-2"]
        cfg = main_module.replace(wanted[0], sources=trimmed, channel_base=0)

        assert main_module.changed_groups(running, main_module.split_groups(cfg)) == [0, 1, 2, 3, 4]


class TestGroupSupervisor:
    @staticmethod
    def _cfg(tmp_path, count: int, size: int = 4, frames: int = 0):
        return main_module.load_app_config(
            write_full_config(
                tmp_path,
                [
                    (("mode",), "group"),
                    (("group", "size"), size),
                    (("inference", "frames"), frames),
                    (("streams", "sources"), sources(count)),
                ],
            )
        )

    @staticmethod
    def _edit(cfg, count: int) -> None:
        """Rewrite the watched config with `count` streams and a newer mtime."""
        raw = main_module.load_raw_config(cfg.config_path)
        raw["streams"]["sources"] = sources(count)
        cfg.config_path.write_text(main_module.yaml.safe_dump(raw), encoding="utf-8")
        stamp = cfg.config_path.stat().st_mtime + 10
        main_module.os.utime(cfg.config_path, (stamp, stamp))

    @staticmethod
    def _script_sleeps(monkeypatch, *steps):
        """Run one step per supervisor poll, then stop it as Ctrl-C would."""
        pending = list(steps)

        def sleep(_seconds):
            if not pending:
                raise KeyboardInterrupt()
            pending.pop(0)()

        monkeypatch.setattr(main_module.time, "sleep", sleep)

    def _context(self, monkeypatch, outcomes=()):
        context = FakeContext(outcomes)
        monkeypatch.setattr(main_module.multiprocessing, "get_context", lambda method: context)
        return context

    def test_every_group_starts_in_its_own_process_with_its_channel_range(self, tmp_path, monkeypatch):
        context = self._context(monkeypatch)
        self._script_sleeps(monkeypatch)

        main_module.run_group(self._cfg(tmp_path, count=6))

        assert [(p.group_index, p.stream_ids) for p in context.processes] == [
            (0, ["cam-1", "cam-2", "cam-3", "cam-4"]),
            (1, ["cam-5", "cam-6"]),
        ]
        assert [p.args[0].channel_base for p in context.processes] == [0, 4]
        assert all(p.target is main_module._group_worker for p in context.processes)
        # Stopping the supervisor stops every group through its normal shutdown.
        assert all(p.terminated for p in context.processes)

    def test_adding_a_stream_restarts_only_its_group(self, tmp_path, monkeypatch):
        context = self._context(monkeypatch)
        cfg = self._cfg(tmp_path, count=5)
        self._script_sleeps(monkeypatch, lambda: self._edit(cfg, 6), lambda: None)

        main_module.run_group(cfg)

        assert context.log[:4] == [
            ("start", 0, ["cam-1", "cam-2", "cam-3", "cam-4"]),
            ("start", 1, ["cam-5"]),
            ("stop", 1, ["cam-5"]),
            ("start", 1, ["cam-5", "cam-6"]),
        ]
        # Group 0 kept running through the edit and stopped only at shutdown.
        assert context.log.index(("stop", 0, ["cam-1", "cam-2", "cam-3", "cam-4"])) > 3
        assert context.processes[-1].args[0].channel_base == 4

    def test_a_stream_beyond_the_last_group_starts_a_new_group_only(self, tmp_path, monkeypatch):
        context = self._context(monkeypatch)
        cfg = self._cfg(tmp_path, count=4)
        self._script_sleeps(monkeypatch, lambda: self._edit(cfg, 5), lambda: None)

        main_module.run_group(cfg)

        assert context.log[:2] == [
            ("start", 0, ["cam-1", "cam-2", "cam-3", "cam-4"]),
            ("start", 1, ["cam-5"]),
        ]
        assert context.processes[0].terminated  # only at shutdown
        assert context.log.index(("stop", 0, ["cam-1", "cam-2", "cam-3", "cam-4"])) > 1

    def test_removing_the_last_groups_streams_stops_that_group(self, tmp_path, monkeypatch):
        context = self._context(monkeypatch)
        cfg = self._cfg(tmp_path, count=5)
        self._script_sleeps(monkeypatch, lambda: self._edit(cfg, 4), lambda: None)

        main_module.run_group(cfg)

        assert context.log[:3] == [
            ("start", 0, ["cam-1", "cam-2", "cam-3", "cam-4"]),
            ("start", 1, ["cam-5"]),
            ("stop", 1, ["cam-5"]),
        ]
        assert len(context.processes) == 2

    def test_an_invalid_edit_is_ignored(self, tmp_path, monkeypatch, capsys):
        context = self._context(monkeypatch)
        cfg = self._cfg(tmp_path, count=5)

        def bad_edit():
            raw = main_module.load_raw_config(cfg.config_path)
            raw["streams"]["sources"] = [sources(1)[0], sources(1)[0]]
            cfg.config_path.write_text(main_module.yaml.safe_dump(raw), encoding="utf-8")
            stamp = cfg.config_path.stat().st_mtime + 10
            main_module.os.utime(cfg.config_path, (stamp, stamp))

        self._script_sleeps(monkeypatch, bad_edit, lambda: None)

        main_module.run_group(cfg)

        assert len(context.processes) == 2
        assert "ignoring invalid config reload" in capsys.readouterr().err

    def test_a_failed_group_does_not_block_the_next_and_fails_a_finite_run(self, tmp_path, monkeypatch):
        context = self._context(monkeypatch, ["fail", "ready"])

        def finish():
            context.processes[1].exitcode = 0

        self._script_sleeps(monkeypatch, finish)

        with pytest.raises(RuntimeError, match=r"group\(s\) 0 \(exit 1\) failed"):
            main_module.run_group(self._cfg(tmp_path, count=8, frames=10))

        assert len(context.processes) == 2

    def test_a_finite_run_ends_when_every_group_finishes(self, tmp_path, monkeypatch):
        context = self._context(monkeypatch)

        def finish():
            for process in context.processes:
                process.exitcode = 0

        self._script_sleeps(monkeypatch, finish)

        main_module.run_group(self._cfg(tmp_path, count=2, frames=10))

        assert context.processes[0].terminated is False

    def test_an_edit_restarts_a_group_that_had_failed(self, tmp_path, monkeypatch):
        context = self._context(monkeypatch, ["ready", "fail"])
        cfg = self._cfg(tmp_path, count=5)
        self._script_sleeps(monkeypatch, lambda: self._edit(cfg, 6), lambda: None)

        main_module.run_group(cfg)

        assert ("start", 1, ["cam-5", "cam-6"]) in context.log
        assert context.processes[-1].is_alive() is False  # stopped at shutdown


class TestDetectorGeometry:
    @pytest.mark.parametrize(
        ("probes", "expected"),
        [
            # Same size everywhere: nothing is scaled.
            ([(1920, 1080, 30), (1920, 1080, 30)], (1920, 1080)),
            # Mixed 720p and 4K: the largest size, clamped to 2x the smallest.
            ([(1280, 720, 30), (3840, 2160, 30)], (2560, 1440)),
            # A lone small source stays native and skips the scaler.
            ([(640, 480, 30)], (640, 480)),
            # Never below the model input unless every source is smaller.
            ([(320, 180, 30), (1920, 1080, 30)], (640, 640)),
            # NV12 needs even dimensions.
            ([(1281, 721, 30)], (1280, 720)),
        ],
    )
    def test_shared_detector_geometry(self, probes, expected):
        assert main_module.detector_geometry(probes) == expected


def bbox_payload(boxes: list[tuple[int, int, int, int, float, int]], declared: int | None = None) -> bytes:
    payload = struct.pack("<I", len(boxes) if declared is None else declared)
    for box in boxes:
        payload += struct.pack("<iiiifi", *box)
    return payload


class TestBoxesAndMetadata:
    def test_boxes_are_clamped_to_the_frame(self):
        boxes = main_module.parse_boxes_strict(bbox_payload([(-5, 10, 50, 2000, 0.9, 0)]), 100, 100, 50)

        assert boxes == [{"x1": 0.0, "y1": 10.0, "x2": 45.0, "y2": 100.0, "score": pytest.approx(0.9), "class_id": 0}]

    def test_a_header_beyond_the_payload_is_rejected(self):
        with pytest.raises(RuntimeError, match="bbox header exceeds payload count"):
            main_module.parse_boxes_strict(bbox_payload([(0, 0, 1, 1, 0.5, 0)], declared=3), 100, 100, 50)

    def test_a_header_beyond_top_k_is_rejected(self):
        payload = bbox_payload([(0, 0, 1, 1, 0.5, 0)] * 3)

        with pytest.raises(RuntimeError, match="bbox header exceeds expected topk"):
            main_module.parse_boxes_strict(payload, 100, 100, 2)

    def test_metadata_objects_are_compact(self):
        boxes = [
            {"x1": 10.6, "y1": 20.2, "x2": 40.9, "y2": 60.4, "score": 0.756, "class_id": 0},
            {"x1": 90.0, "y1": 90.0, "x2": 120.0, "y2": 120.0, "score": 0.5, "class_id": 99},
        ]

        objects = main_module.metadata_objects(boxes, ["person"], 100, 100)

        assert objects == [
            {"label": "person", "confidence": 0.76, "bbox": [10, 20, 30, 40]},
            {"label": "unknown", "confidence": 0.5, "bbox": [90, 90, 10, 10]},
        ]
        assert "id" not in objects[0]

    @pytest.mark.parametrize(
        ("pts_ns", "expected"),
        [(-1, 0), (0, 0), (1_000_000_000, 90_000), (2**63 - 1, ((2**63 - 1) * 90_000 // 10**9) & 0xFFFFFFFF)],
    )
    def test_rtp_timestamp_is_the_32_bit_90khz_clock(self, pts_ns, expected):
        assert main_module.rtp_timestamp_from_pts_ns(pts_ns) == expected


class FakeMetadataSender:
    def __init__(self):
        self.metadata_calls = []
        self.raw_calls = []

    def send_metadata(self, metadata_type, data_json, timestamp_ms, frame_id):
        self.metadata_calls.append((metadata_type, data_json, timestamp_ms, frame_id))
        return True

    def send_raw_json(self, payload):
        self.raw_calls.append(payload)
        return True


class FakeSample:
    frame_id = 42
    pts_ns = 1_234_000_000
    dts_ns = -1
    duration_ns = 40_000_000
    input_seq = 7
    stream_id = "stream1"


BOX = {"x1": 10.0, "y1": 20.0, "x2": 40.0, "y2": 60.0, "score": 0.75, "class_id": 0}


class TestLiveMetadata:
    def test_live_metadata_carries_the_rtp_timestamp_insight_matches_on(self):
        sender = FakeMetadataSender()
        stream = main_module.LiveStream(
            channel=3, id="cam-4", url="rtsp://a/4", labels=["person"], width=100, height=100,
            metadata_sender=sender,
        )

        main_module.send_live_metadata(stream, FakeSample(), [dict(BOX)], stream_count=5)

        assert len(sender.raw_calls) == 1
        payload = json.loads(sender.raw_calls[0])
        assert payload["type"] == "object-detection"
        assert payload["rtp_timestamp"] == main_module.rtp_timestamp_from_pts_ns(FakeSample.pts_ns)
        assert payload["timestamp"] == 1234
        assert payload["frame_id"] == "42"
        assert (payload["stream_id"], payload["stream_index"]) == ("cam-4", 3)
        assert payload["data"] == {
            "objects": [{"label": "person", "confidence": 0.75, "bbox": [10, 20, 30, 40]}],
            "stream_count": 5,
            "stream_id": "cam-4",
        }

    def test_a_failed_send_does_not_stop_detection(self):
        class FailingSender:
            def send_raw_json(self, payload):
                raise OSError("network unreachable")

        stream = main_module.LiveStream(
            channel=0, id="cam-1", url="rtsp://a/1", labels=["person"], width=100, height=100,
            metadata_sender=FailingSender(),
        )

        main_module.send_live_metadata(stream, FakeSample(), [dict(BOX)], stream_count=1)


def fused_stream(sender, index=0, frame=(100, 100), det=(100, 100)) -> object:
    return main_module.FusedStream(
        index=index,
        channel=4 + index,
        id=f"cam-{index + 1}",
        url=f"rtsp://a/{index + 1}",
        metadata_sender=sender,
        labels=["person"],
        profile=main_module.ProfileWindow(False, f"cam-{index + 1}"),
        frame_w=frame[0],
        frame_h=frame[1],
        det_w=det[0],
        det_h=det[1],
        source_fps=30,
        video_port=9004 + index,
    )


def fused_cfg(**overrides):
    values = {"frames": 0, "fps": 0, "warmup_frames": 0, "max_detections": 50, "save_dir": "", "save_every": 0}
    values.update(overrides)
    return SimpleNamespace(**values)


class TestFusedMetadata:
    def test_scale_metadata_is_arrival_order_without_rtp_timestamp(self):
        sender = FakeMetadataSender()

        main_module.send_fused_metadata(fused_stream(sender), FakeSample(), [dict(BOX)])

        assert sender.raw_calls == []
        metadata_type, data_json, timestamp_ms, frame_id = sender.metadata_calls[0]
        assert metadata_type == "object-detection"
        assert (timestamp_ms, frame_id) == (1234, "42")
        assert json.loads(data_json) == {
            "objects": [{"label": "person", "confidence": 0.75, "bbox": [10, 20, 30, 40]}]
        }

    def test_boxes_are_scaled_from_the_detector_geometry_to_the_native_frame(self, monkeypatch):
        sender = FakeMetadataSender()
        stream = fused_stream(sender, frame=(1280, 720), det=(2560, 1440))
        payload = bbox_payload([(200, 100, 400, 200, 0.8, 0)])
        monkeypatch.setattr(main_module, "extract_bbox_payload", lambda sample: payload)

        main_module.process_fused_sample(stream, fused_cfg(), FakeSample(), 1.0)

        objects = json.loads(sender.metadata_calls[0][1])["objects"]
        assert objects[0]["bbox"] == [100, 50, 200, 100]

    def test_warmup_detections_are_not_published(self, monkeypatch):
        sender = FakeMetadataSender()
        stream = fused_stream(sender)
        monkeypatch.setattr(main_module, "extract_bbox_payload", lambda sample: bbox_payload([]))

        for _ in range(3):
            main_module.process_fused_sample(stream, fused_cfg(warmup_frames=2), FakeSample(), 1.0)

        assert stream.processed == 3
        assert len(sender.metadata_calls) == 1

    def test_inference_fps_caps_how_often_a_stream_publishes(self, monkeypatch):
        sender = FakeMetadataSender()
        stream = fused_stream(sender)
        monkeypatch.setattr(main_module, "extract_bbox_payload", lambda sample: bbox_payload([]))
        clock = iter([1000.0, 1050.0, 1100.0])
        monkeypatch.setattr(main_module, "time_ms", lambda: next(clock))

        # Warmup covers every frame, so the clock is read only by the fps check.
        for _ in range(3):
            main_module.process_fused_sample(stream, fused_cfg(fps=10, warmup_frames=10), FakeSample(), 1.0)

        # 10 fps allows one publish per 100 ms: t=1000 and t=1100, not t=1050.
        assert stream.processed == 2

    def test_a_stream_stops_at_its_frame_limit(self, monkeypatch):
        sender = FakeMetadataSender()
        stream = fused_stream(sender)
        monkeypatch.setattr(main_module, "extract_bbox_payload", lambda sample: bbox_payload([]))

        for _ in range(4):
            main_module.process_fused_sample(stream, fused_cfg(frames=2), FakeSample(), 1.0)

        assert stream.processed == 2
        assert main_module.all_streams_done([stream], 2)


class TestFusedPullOutcomes:
    """The scale pull loop, driven by a scripted run."""

    def test_a_sample_is_routed_to_its_stream_by_stream_id(self, monkeypatch):
        senders = [FakeMetadataSender(), FakeMetadataSender()]
        streams = [fused_stream(senders[0], 0), fused_stream(senders[1], 1)]
        monkeypatch.setattr(main_module, "extract_bbox_payload", lambda sample: bbox_payload([]))
        run = FakeRun(("sample", FakeSample()))

        assert main_module.process_fused_run_once(run, fused_cfg(), streams) is True

        assert (streams[0].processed, streams[1].processed) == (0, 1)
        assert run.pulls == [("detections", 50)]

    def test_timeout_keeps_the_run_going(self):
        streams = [fused_stream(FakeMetadataSender())]

        assert main_module.process_fused_run_once(FakeRun("timeout"), fused_cfg(), streams) is True
        assert streams[0].closed is False

    def test_runtime_error_ends_the_run(self):
        streams = [fused_stream(FakeMetadataSender())]

        with pytest.raises(RuntimeError, match="queue torn down"):
            main_module.process_fused_run_once(FakeRun(("error", "queue torn down")), fused_cfg(), streams)

    def test_a_stopped_run_closes_every_stream(self):
        class StoppedRun(FakeRun):
            def running(self):
                return False

        streams = [fused_stream(FakeMetadataSender(), 0), fused_stream(FakeMetadataSender(), 1)]

        assert main_module.process_fused_run_once(StoppedRun("closed"), fused_cfg(), streams) is False
        assert all(s.closed for s in streams)
        assert main_module.all_streams_done(streams, 0)

    @pytest.mark.parametrize("stream_id", ["camera0", "stream9", "streamx"])
    def test_an_unknown_stream_id_is_an_error(self, stream_id):
        sample = SimpleNamespace(stream_id=stream_id)

        with pytest.raises(RuntimeError, match="stream id"):
            main_module.stream_index_from_sample(sample, 2)


class NeatError(Exception):
    """Stands in for pyneat.NeatError, which is not a RuntimeError."""


class ScriptedRun:
    """A run whose pulls raise or return per a script; `on_pull` runs before each."""

    def __init__(self, *outcomes, on_pull=None):
        self._outcomes = list(outcomes)
        self._on_pull = on_pull
        self.closed = False

    def pull(self, output_name, timeout_ms=0):
        if self._on_pull is not None:
            self._on_pull()
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def running(self):
        return True

    def last_error(self):
        return ""

    def close(self):
        self.closed = True


ACCELERATOR_ERROR = NeatError(
    "[infra.accelerator_execution_failed] The accelerator could not execute this model stage."
)


class TestLivePullRecovery:
    """A live stream rebuilt by its own worker, driven by scripted runs."""

    @staticmethod
    def _manager(tmp_path, monkeypatch, runs: list, builds: list):
        cfg = main_module.load_app_config(
            write_full_config(
                tmp_path, [(("inference", "frames"), 1), (("runtime", "warmup_frames"), 0)]
            )
        )
        manager = main_module.StreamManager(cfg, ["person"])

        def init_live_stream(cfg, channel, source, labels):
            return main_module.LiveStream(
                channel=channel, id=source.id, url=source.url, labels=labels, width=100,
                height=100, run=runs.pop(0), profile=main_module.ProfileWindow(False, source.id),
            )

        def build_live_pipeline(cfg, stream):
            builds.append(stream.id)
            stream.run = runs.pop(0)

        monkeypatch.setattr(main_module, "init_live_stream", init_live_stream)
        monkeypatch.setattr(main_module, "build_live_pipeline", build_live_pipeline)
        monkeypatch.setattr(main_module, "extract_bbox_payload", lambda sample: bbox_payload([]))
        monkeypatch.setattr(main_module, "send_live_metadata", lambda *args: None)
        monkeypatch.setattr(main_module.time, "sleep", lambda _s: None)
        return manager

    def test_an_accelerator_failure_rebuilds_the_stream_instead_of_failing_it(self, tmp_path, monkeypatch):
        builds: list[str] = []
        runs = [ScriptedRun(*[ACCELERATOR_ERROR] * 8), ScriptedRun(FakeSample())]
        manager = self._manager(tmp_path, monkeypatch, runs, builds)
        managed = main_module.ManagedStream("cam-1", "rtsp://a/1", 0, main_module.threading.Event())

        manager._consume(managed)

        assert builds == ["cam-1"]
        assert manager.failed == {}
        assert manager.errors == []

    def test_a_pull_error_while_stopping_is_not_a_stream_failure(self, tmp_path, monkeypatch):
        manager_ref = {}
        runs = [ScriptedRun(ACCELERATOR_ERROR, on_pull=lambda: manager_ref["m"].stop_event.set())]
        manager = self._manager(tmp_path, monkeypatch, runs, [])
        manager_ref["m"] = manager
        managed = main_module.ManagedStream("cam-1", "rtsp://a/1", 0, main_module.threading.Event())

        manager._consume(managed)

        assert manager.failed == {}

    def test_a_persistent_non_accelerator_error_fails_only_that_stream(self, tmp_path, monkeypatch):
        runs = [ScriptedRun(NeatError("queue torn down"))]
        manager = self._manager(tmp_path, monkeypatch, runs, [])
        managed = main_module.ManagedStream("cam-1", "rtsp://a/1", 0, main_module.threading.Event())

        manager._consume(managed)

        assert "queue torn down" in manager.failed["cam-1"]
        assert manager.stop_event.is_set() is False
