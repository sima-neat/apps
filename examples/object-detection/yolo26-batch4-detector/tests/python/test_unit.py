"""Unit tests for the batch-4 YOLO26 detector example.

These cover the parts that need no hardware: config loading and validation, and
the shape-based head mapping, which is the one piece of plumbing this example adds
on top of the standard YOLO26 decode.
"""

from __future__ import annotations

import json
import socket
import subprocess
import sys
import textwrap
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from tests.utils.metadata_json_listener import MetadataJsonListener

EXAMPLE_DIR = Path(__file__).resolve().parent.parent.parent
PYTHON_DIR = EXAMPLE_DIR / "src" / "python"
MAIN_PY = PYTHON_DIR / "main.py"

if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

pytestmark = pytest.mark.unit


def write_config(
    tmp_path: Path,
    streams: list[str],
    model_path: str = "assets/models/yolo26m-det-int8-b4.tar.gz",
    **overrides,
) -> Path:
    stream_lines = "\n".join(f"  - {url}" for url in streams)
    inference = ""
    if overrides:
        lines = "\n".join(f"  {key}: {value}" for key, value in overrides.items())
        inference = f"inference:\n{lines}\n"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "model:\n"
        f"  path: {model_path}\n"
        "streams:\n"
        f"{stream_lines}\n"
        f"{inference}"
        "output:\n"
        "  insight:\n"
        "    host: 127.0.0.1\n",
        encoding="utf-8",
    )
    return config_path


class TestMainEntrypoint:
    def test_help_runs(self):
        result = subprocess.run(
            [sys.executable, str(MAIN_PY), "--help"],
            check=False,
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
            check=False,
            capture_output=True,
            text=True,
            cwd=str(EXAMPLE_DIR),
            timeout=20,
        )
        assert result.returncode == 2
        assert "config file not found" in result.stderr

    def test_validate_config_only_reports_stream_count(self, tmp_path: Path):
        config_path = write_config(
            tmp_path, ["rtsp://127.0.0.1:8554/src1", "rtsp://127.0.0.1:8554/src2"]
        )
        result = subprocess.run(
            [
                sys.executable,
                str(MAIN_PY),
                "--config",
                str(config_path),
                "--validate-config-only",
            ],
            check=False,
            capture_output=True,
            text=True,
            cwd=str(EXAMPLE_DIR),
            timeout=20,
        )
        assert result.returncode == 0
        assert "streams=2" in result.stdout


class TestConfigLoading:
    def test_accepts_four_streams(self, tmp_path: Path):
        from main import load_app_config

        cfg = load_app_config(
            write_config(tmp_path, [f"rtsp://127.0.0.1:8554/src{i}" for i in range(4)])
        )
        assert len(cfg.rtsp_urls) == 4
        assert cfg.insight_host == "127.0.0.1"
        assert cfg.score_threshold == 0.35
        assert cfg.max_detections == 100

    def test_rejects_more_than_four_streams(self, tmp_path: Path):
        from main import load_app_config

        config_path = write_config(
            tmp_path, [f"rtsp://127.0.0.1:8554/src{i}" for i in range(5)]
        )
        with pytest.raises(ValueError, match="up to 4 streams"):
            load_app_config(config_path)

    def test_rejects_placeholder_stream(self, tmp_path: Path):
        from main import load_app_config

        with pytest.raises(ValueError, match="placeholder"):
            load_app_config(write_config(tmp_path, ["<rtsp-url-1>"]))

    def test_rejects_placeholder_model_path(self, tmp_path: Path):
        from main import load_app_config

        with pytest.raises(ValueError, match="placeholder"):
            load_app_config(
                write_config(
                    tmp_path,
                    ["rtsp://127.0.0.1:8554/src1"],
                    model_path="<model-path>",
                )
            )

    def test_rejects_out_of_range_score_threshold(self, tmp_path: Path):
        from main import load_app_config

        config_path = write_config(
            tmp_path, ["rtsp://127.0.0.1:8554/src1"], score_threshold=1.5
        )
        with pytest.raises(ValueError, match="score_threshold"):
            load_app_config(config_path)

    def test_rejects_empty_streams(self, tmp_path: Path):
        from main import load_app_config

        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            textwrap.dedent(
                """
                model:
                  path: assets/models/yolo26m-det-int8-b4.tar.gz
                streams: []
                output:
                  insight:
                    host: 127.0.0.1
                """
            ).strip(),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="streams"):
            load_app_config(config_path)


class TestHeadMapping:
    """The six model outputs must be sorted into levels by shape alone."""

    GRIDS = (80, 40, 20)

    @classmethod
    def outputs(cls, batch: int = 4, shuffle: bool = False) -> list[np.ndarray]:
        """Six [N,H,W,C] outputs, each filled with a value identifying it."""
        arrays = []
        for kind, channels in (("bbox", 4), ("cls", 80)):
            for grid in cls.GRIDS:
                marker = float(cls.GRIDS.index(grid) + (0 if kind == "bbox" else 10))
                arrays.append(
                    np.full((batch, grid, grid, channels), marker, dtype=np.float32)
                )
        if shuffle:
            arrays = [arrays[i] for i in (4, 0, 5, 2, 3, 1)]
        return arrays

    def test_maps_heads_to_levels(self):
        import main

        heads = main.heads_from_outputs(self.outputs(), lane=0, class_count=80)
        assert set(heads) == {f"bbox_{lv}" for lv in range(3)} | {
            f"class_logit_{lv}" for lv in range(3)
        }
        for level, grid in enumerate(self.GRIDS):
            assert heads[f"bbox_{level}"].shape == (grid, grid, 4)
            assert heads[f"class_logit_{level}"].shape == (grid, grid, 80)

    def test_output_order_does_not_matter(self):
        import main

        ordered = main.heads_from_outputs(
            self.outputs(shuffle=False), lane=0, class_count=80
        )
        shuffled = main.heads_from_outputs(
            self.outputs(shuffle=True), lane=0, class_count=80
        )
        for name, plane in ordered.items():
            assert np.array_equal(shuffled[name], plane)

    def test_lanes_are_independent(self):
        import main

        arrays = self.outputs()
        arrays[0][2] = 7.0  # bbox level 0, lane 2 only
        assert (
            main.heads_from_outputs(arrays, lane=2, class_count=80)["bbox_0"][0, 0, 0]
            == 7.0
        )
        assert (
            main.heads_from_outputs(arrays, lane=1, class_count=80)["bbox_0"][0, 0, 0]
            == 0.0
        )

    def test_rejects_unbatched_output(self):
        import main

        with pytest.raises(RuntimeError, match=r"\[N,H,W,C\]"):
            main.heads_from_outputs(
                [np.zeros((80, 80, 4), dtype=np.float32)], lane=0, class_count=80
            )

    def test_rejects_lane_out_of_range(self):
        import main

        with pytest.raises(RuntimeError, match="out of range"):
            main.heads_from_outputs(self.outputs(batch=4), lane=4, class_count=80)

    def test_rejects_unknown_channel_count(self):
        import main

        arrays = self.outputs()
        arrays[0] = np.zeros((4, 80, 80, 7), dtype=np.float32)
        with pytest.raises(RuntimeError, match="neither a bbox head"):
            main.heads_from_outputs(arrays, lane=0, class_count=80)

    def test_rejects_wrong_head_count(self):
        import main

        with pytest.raises(RuntimeError, match="expected 3 bbox and 3 class heads"):
            main.heads_from_outputs(self.outputs()[:4], lane=0, class_count=80)


class TestModelContract:
    @staticmethod
    def valid_outputs(classes: int = 6):
        outputs = []
        for channels in (4, classes):
            outputs.extend([[4, grid, grid, channels] for grid in (80, 40, 20)])
        return outputs

    def test_accepts_reordered_non80_heads(self):
        import main

        outputs = self.valid_outputs()
        outputs = [outputs[index] for index in (4, 0, 5, 2, 3, 1)]
        contract = main.validate_model_contract([[4, 640, 640, 3]], outputs, 6)
        assert contract.net == 640
        assert contract.class_count == 6
        assert contract.grids == ((80, 80), (40, 40), (20, 20))

    @pytest.mark.parametrize(
        ("inputs", "outputs", "labels", "message"),
        [
            ([[2, 640, 640, 3]], valid_outputs.__func__(), 6, "shape \\[4,N,N,3\\]"),
            ([[4, 640, 320, 3]], valid_outputs.__func__(), 6, "shape \\[4,N,N,3\\]"),
            ([[4, 640, 640, 3]], [[2, 80, 80, 4]] * 6, 6, "every model output"),
            (
                [[4, 640, 640, 3]],
                [[4, 80, 40, 4]] + valid_outputs.__func__()[1:],
                6,
                "every model output",
            ),
            (
                [[4, 640, 640, 3]],
                valid_outputs.__func__()[:-1] + [[4, 20, 20, 7]],
                6,
                "common class count",
            ),
            ([[4, 640, 640, 3]], valid_outputs.__func__(), 5, "label count"),
            ([[4, 640, 640, 3]], valid_outputs.__func__(4), 4, "ambiguous"),
        ],
    )
    def test_rejects_invalid_contracts(self, inputs, outputs, labels, message):
        import main

        with pytest.raises(RuntimeError, match=message):
            main.validate_model_contract(inputs, outputs, labels)

    def test_rejects_mismatched_grid_pairs(self):
        import main

        outputs = self.valid_outputs()
        outputs[-1] = [4, 10, 10, 6]
        with pytest.raises(RuntimeError, match="matching grids"):
            main.validate_model_contract([[4, 640, 640, 3]], outputs, 6)

    def test_rejects_duplicate_levels(self):
        import main

        outputs = [[4, 80, 80, 4], [4, 80, 80, 4], [4, 20, 20, 4]] + [
            [4, 80, 80, 6],
            [4, 80, 80, 6],
            [4, 20, 20, 6],
        ]
        with pytest.raises(RuntimeError, match="distinct matching grids"):
            main.validate_model_contract([[4, 640, 640, 3]], outputs, 6)


class TestDecode:
    def test_decode_finds_a_planted_box(self):
        import main

        main.np = np  # decode_heads uses the module-level numpy handle
        heads = {
            f"bbox_{lv}": np.zeros((grid, grid, 4), dtype=np.float32)
            for lv, grid in enumerate((80, 40, 20))
        }
        heads.update(
            {
                f"class_logit_{lv}": np.full((grid, grid, 80), -20.0, dtype=np.float32)
                for lv, grid in enumerate((80, 40, 20))
            }
        )
        # one confident cell at grid 20 (stride 32), class 2, half-cell box
        heads["class_logit_2"][5, 7, 2] = 10.0
        heads["bbox_2"][5, 7] = [0.5, 0.5, 0.5, 0.5]

        dets = main.decode_heads(
            heads, net=640, score_threshold=0.35, max_detections=10
        )

        assert len(dets) == 1
        det = dets[0]
        assert det["class_id"] == 2
        assert det["score"] > 0.99
        # anchor centre (7.5, 5.5) grid units, +/- 0.5 -> 7.0..8.0, 5.0..6.0, x32
        assert det["x1"] == pytest.approx(224.0)
        assert det["x2"] == pytest.approx(256.0)
        assert det["y1"] == pytest.approx(160.0)
        assert det["y2"] == pytest.approx(192.0)

    def test_to_original_undoes_letterbox(self):
        import main

        dets = [
            {
                "score": 0.9,
                "class_id": 0,
                "x1": 100.0,
                "y1": 200.0,
                "x2": 200.0,
                "y2": 300.0,
            }
        ]
        # 1280x720 letterboxed into 640: scale 0.5, dy 140, dx 0
        mapped = main.to_original(dets, scale=0.5, dx=0, dy=140, width=1280, height=720)
        assert len(mapped) == 1
        assert mapped[0]["x1"] == pytest.approx(200.0)
        assert mapped[0]["y1"] == pytest.approx(120.0)
        assert mapped[0]["x2"] == pytest.approx(400.0)
        assert mapped[0]["y2"] == pytest.approx(320.0)


class TestPullLatest:
    class FakeRun:
        def __init__(self, pulls, error=""):
            self.pulls = iter(pulls)
            self.error = error

        def pull(self, _name, _timeout):
            return next(self.pulls)

        def last_error(self):
            return self.error

    def test_transient_timeout_returns_none(self):
        import main

        assert main.pull_latest(self.FakeRun([None]), "frame_0", 20) is None

    def test_run_error_is_fatal(self):
        import main

        with pytest.raises(RuntimeError, match="decoder failed"):
            main.pull_latest(
                self.FakeRun([None], error="decoder failed"), "frame_0", 20
            )

    def test_drain_timeout_returns_latest_sample(self):
        import main

        sample = object()
        assert main.pull_latest(self.FakeRun([sample, None]), "frame_0", 20) is sample


class TestMetadata:
    def test_uses_the_analysed_frames_rtp_timestamp(self):
        import main

        class Sender:
            payload = ""

            def send_raw_json(self, payload):
                self.payload = payload
                return True

        class Stream:
            metadata_sender = Sender()

        sample = main.FrameRef(pts_ns=12_402_999_999, frame_id=327)
        main.send_metadata(Stream(), sample, [], ["person"])
        payload = json.loads(Stream.metadata_sender.payload)

        assert payload["timestamp"] == 12_402
        assert payload["frame_id"] == "327"
        assert (
            payload["_insight"]["rtp_timestamp"]
            == (sample.pts_ns * 90_000) // 1_000_000_000
        )

    def test_wraps_rtp_timestamp_to_32_bits(self):
        import main

        class Sender:
            payload = ""

            def send_raw_json(self, payload):
                self.payload = payload
                return True

        class Stream:
            metadata_sender = Sender()

        sample = main.FrameRef(pts_ns=50_000_000_000_000, frame_id=1)
        main.send_metadata(Stream(), sample, [], ["person"])
        payload = json.loads(Stream.metadata_sender.payload)

        expected = ((sample.pts_ns * 90_000) // 1_000_000_000) & 0xFFFFFFFF
        assert payload["_insight"]["rtp_timestamp"] == expected
        assert 0 <= payload["_insight"]["rtp_timestamp"] <= 0xFFFFFFFF


class TestLetterbox:
    """The batch lane is filled in place, so geometry and padding must be exact."""

    @staticmethod
    def prepared():
        cv2 = pytest.importorskip("cv2")
        import main

        main.np = np
        main.cv2 = cv2
        return main

    def test_writes_lane_in_place_with_expected_geometry(self):
        main = self.prepared()

        rgb = np.full((720, 1280, 3), 255, dtype=np.uint8)
        lane = np.empty((640, 640, 3), dtype=np.float32)
        scale, dx, dy = main.letterbox_into(rgb, lane, 640)

        # 1280x720 into 640 keeps aspect: 640x360, centered vertically.
        assert scale == pytest.approx(0.5)
        assert (dx, dy) == (0, 140)
        # Image area is the scaled frame, the rest is pad 114.
        assert lane[dy + 10, 10, 0] == pytest.approx(1.0)
        assert lane[0, 0, 0] == pytest.approx(114.0 / 255.0)
        assert lane[639, 639, 0] == pytest.approx(114.0 / 255.0)

    def test_reuses_the_same_buffer(self):
        main = self.prepared()

        lane = np.zeros((640, 640, 3), dtype=np.float32)
        before = lane.__array_interface__["data"][0]
        main.letterbox_into(np.full((720, 1280, 3), 255, dtype=np.uint8), lane, 640)
        assert lane.__array_interface__["data"][0] == before

    def test_square_input_fills_the_lane(self):
        main = self.prepared()

        lane = np.empty((640, 640, 3), dtype=np.float32)
        scale, dx, dy = main.letterbox_into(
            np.zeros((480, 480, 3), dtype=np.uint8), lane, 640
        )
        assert (dx, dy) == (0, 0)
        assert scale == pytest.approx(640.0 / 480.0)
        assert float(lane.max()) == pytest.approx(0.0)

    def test_positive_half_tie_matches_cpp_lround(self):
        main = self.prepared()

        lane = np.empty((640, 640, 3), dtype=np.float32)
        scale, dx, dy = main.letterbox_into(
            np.zeros((721, 1280, 3), dtype=np.uint8), lane, 640
        )
        assert scale == pytest.approx(0.5)
        assert (dx, dy) == (0, 139)


class TestMetadataJsonListener:
    @staticmethod
    def payload(frame_id: str, padding: int = 0) -> bytes:
        return json.dumps(
            {
                "type": "object-detection",
                "timestamp": 42,
                "frame_id": frame_id,
                "data": {"objects": [{"label": "person", "padding": "x" * padding}]},
            }
        ).encode()

    @classmethod
    def chunks(cls, payload: bytes, message_id: int) -> list[bytes]:
        parts = [
            payload[offset : offset + 1188] for offset in range(0, len(payload), 1188)
        ]
        assert len(parts) >= 2
        return [
            bytes((0x4E, 0x01))
            + message_id.to_bytes(8, "big")
            + bytes((index, len(parts)))
            + part
            for index, part in enumerate(parts)
        ]

    @staticmethod
    def bound_listener(**kwargs):
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
        probe.close()
        return port, MetadataJsonListener("127.0.0.1", port, 1, **kwargs)

    def test_accepts_raw_json_datagram(self):
        port, listener = self.bound_listener()
        with listener, socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
            sender.sendto(self.payload("raw"), ("127.0.0.1", port))
            result = listener.wait_for_messages(0.5)
        assert result.success
        assert result.messages[0].frame_id == "raw"

    def test_reassembles_out_of_order_chunks_and_exact_duplicate(self):
        port, listener = self.bound_listener()
        chunks = self.chunks(self.payload("chunked", padding=3000), 7)
        with listener, socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
            sender.sendto(chunks[1], ("127.0.0.1", port))
            sender.sendto(chunks[1], ("127.0.0.1", port))
            sender.sendto(chunks[0], ("127.0.0.1", port))
            for chunk in chunks[2:]:
                sender.sendto(chunk, ("127.0.0.1", port))
            result = listener.wait_for_messages(0.5)
        assert result.success
        assert result.messages[0].frame_id == "chunked"
        assert result.messages[0].object_count == 1

    def test_rejects_conflicting_duplicate(self):
        port, listener = self.bound_listener()
        chunks = self.chunks(self.payload("conflict", padding=1500), 8)
        conflicting = chunks[0][:-1] + bytes((chunks[0][-1] ^ 1,))
        with listener, socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
            sender.sendto(chunks[0], ("127.0.0.1", port))
            sender.sendto(conflicting, ("127.0.0.1", port))
            sender.sendto(chunks[1], ("127.0.0.1", port))
            result = listener.wait_for_messages(0.1)
        assert not result.success
        assert "conflicting duplicate" in result.error

    def test_expires_incomplete_sequence(self):
        port, listener = self.bound_listener(chunk_expiry_s=0.05)
        chunks = self.chunks(self.payload("expired", padding=1500), 9)
        with (
            listener,
            socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender,
            ThreadPoolExecutor(max_workers=1) as pool,
        ):
            future = pool.submit(listener.wait_for_messages, 0.2)
            sender.sendto(chunks[0], ("127.0.0.1", port))
            time.sleep(0.07)
            sender.sendto(chunks[1], ("127.0.0.1", port))
            result = future.result()
        assert not result.success

    def test_rejects_malformed_sequence(self):
        port, listener = self.bound_listener()
        malformed = bytes((0x4E, 0x02)) + bytes(10)
        with listener, socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
            sender.sendto(malformed, ("127.0.0.1", port))
            result = listener.wait_for_messages(0.1)
        assert not result.success
        assert "unsupported version" in result.error
