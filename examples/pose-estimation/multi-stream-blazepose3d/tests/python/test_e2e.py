"""Hardware E2E tests for the multi-stream BlazePose application."""

from __future__ import annotations

import importlib.util
import json
import math
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.utils.metadata_json_listener import MetadataJsonListener

EXAMPLE_DIR = Path(__file__).resolve().parents[2]
MAIN_PY = EXAMPLE_DIR / "src" / "python" / "main.py"
DETECTOR_MODEL = "yolo26m-det-int8-b1.tar.gz"
POSE_MODEL = "blazepose_ghum_heavy_modalix_bf16_mpk.tar.gz"
INSIGHT_HOST = "127.0.0.1"
MAX_STREAMS = 4
LANDMARK_NAMES = (
    "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner",
    "right_eye", "right_eye_outer", "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist",
    "right_wrist", "left_pinky", "right_pinky", "left_index", "right_index", "left_thumb",
    "right_thumb", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle",
    "right_ankle", "left_heel", "right_heel", "left_foot_index", "right_foot_index",
)


class RtpVideoListener:
    """Requires codec configuration and a complete VCL access unit on each video port."""

    def __init__(self, host: str, base_port: int, num_ports: int, codec: str):
        self._stop = threading.Event()
        self._codec = codec
        self._config_counts = [0] * num_ports
        self._vcl_counts = [0] * num_ports
        self._vcl_timestamps = [set() for _ in range(num_ports)]
        self._fu_progress = [None] * num_ports
        self._lock = threading.Lock()
        self._sockets = []
        self._threads = []
        for offset in range(num_ports):
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.settimeout(0.1)
            sock.bind((host, base_port + offset))
            self._sockets.append(sock)

    def __enter__(self):
        for index, sock in enumerate(self._sockets):
            thread = threading.Thread(target=self._receive, args=(index, sock))
            thread.start()
            self._threads.append(thread)
        return self

    def __exit__(self, *_args):
        self._stop.set()
        for thread in self._threads:
            thread.join()
        for sock in self._sockets:
            sock.close()

    def _receive(self, index: int, sock: socket.socket) -> None:
        while not self._stop.is_set():
            try:
                packet = sock.recv(65536)
            except TimeoutError:
                continue
            config = self._is_codec_config_rtp(packet)
            evidence = self._vcl_evidence_rtp(packet)
            with self._lock:
                if config:
                    self._config_counts[index] += 1
                if evidence is not None and self._completes_vcl_access_unit(index, evidence):
                    self._vcl_counts[index] += 1
                    self._vcl_timestamps[index].add(evidence[6])

    @staticmethod
    def _rtp_payload(packet: bytes):
        if len(packet) < 13 or packet[0] >> 6 != 2 or packet[1] & 0x7F != 96:
            return None
        header_size = 12 + 4 * (packet[0] & 0x0F)
        if header_size >= len(packet):
            return None
        if packet[0] & 0x10:
            if header_size + 4 > len(packet):
                return None
            extension_words = int.from_bytes(packet[header_size + 2 : header_size + 4], "big")
            header_size += 4 + 4 * extension_words
            if header_size >= len(packet):
                return None
        payload_end = len(packet)
        if packet[0] & 0x20:
            padding = packet[-1]
            if padding == 0 or padding > payload_end - header_size:
                return None
            payload_end -= padding
        payload = packet[header_size:payload_end]
        if not payload:
            return None
        return (
            payload,
            int.from_bytes(packet[2:4], "big"),
            int.from_bytes(packet[4:8], "big"),
            bool(packet[1] & 0x80),
        )

    def _is_codec_config_rtp(self, packet: bytes) -> bool:
        parsed = self._rtp_payload(packet)
        if parsed is None:
            return False
        payload, _sequence, _timestamp, _marker = parsed
        if self._codec == "h264":
            nal_type = payload[0] & 0x1F
            if payload[0] & 0x80:
                return False
            if nal_type == 7:
                return len(payload) >= 2
            if nal_type == 28:
                return (
                    len(payload) >= 3
                    and payload[1] & 0xC0 == 0x80
                    and payload[1] & 0x20 == 0
                    and payload[1] & 0x1F == 7
                )
            if nal_type != 24:
                return False
            position = 1
            found_sps = False
            max_nri = 0
            while position < len(payload):
                if position + 2 > len(payload):
                    return False
                nal_size = int.from_bytes(payload[position : position + 2], "big")
                position += 2
                if nal_size == 0 or position + nal_size > len(payload):
                    return False
                if payload[position] & 0x80:
                    return False
                max_nri = max(max_nri, payload[position] & 0x60)
                found_sps |= nal_size >= 2 and payload[position] & 0x9F == 7
                position += nal_size
            return found_sps and payload[0] & 0x60 == max_nri

        if (
            len(payload) < 2
            or payload[0] & 0x81
            or payload[1] & 0xF8
            or payload[1] & 0x07 == 0
        ):
            return False
        nal_type = payload[0] >> 1 & 0x3F
        if nal_type == 32:
            return len(payload) >= 3
        if nal_type == 49:
            return len(payload) >= 4 and payload[2] & 0xC0 == 0x80 and payload[2] & 0x3F == 32
        if nal_type != 48:
            return False
        position = 2
        found_vps = False
        while position < len(payload):
            if position + 2 > len(payload):
                return False
            nal_size = int.from_bytes(payload[position : position + 2], "big")
            position += 2
            if nal_size < 2 or position + nal_size > len(payload):
                return False
            nal = payload[position : position + nal_size]
            found_vps |= (
                nal_size >= 3
                and not nal[0] & 0x81
                and not nal[1] & 0xF8
                and nal[0] >> 1 & 0x3F == 32
                and nal[1] & 0x07 != 0
            )
            position += nal_size
        return found_vps

    def _vcl_evidence_rtp(self, packet: bytes):
        parsed = self._rtp_payload(packet)
        if parsed is None:
            return None
        payload, sequence, timestamp, marker = parsed

        def evidence(*, complete=False, fragment=False, start=False, end=False, signature=0):
            return complete, fragment, start, end, marker, sequence, timestamp, signature

        if self._codec == "h264":
            if payload[0] & 0x80:
                return None
            nal_type = payload[0] & 0x1F
            if 1 <= nal_type <= 5:
                return evidence(complete=marker) if len(payload) >= 2 else None
            if nal_type == 28:
                if len(payload) < 3 or payload[1] & 0x20:
                    return None
                start, end = bool(payload[1] & 0x80), bool(payload[1] & 0x40)
                fragmented_type = payload[1] & 0x1F
                if start and end or not 1 <= fragmented_type <= 5:
                    return None
                return evidence(
                    fragment=True,
                    start=start,
                    end=end,
                    signature=(payload[0] & 0x60) << 8 | fragmented_type,
                )
            if nal_type != 24:
                return None
            position = 1
            found_vcl = False
            max_nri = 0
            while position < len(payload):
                if position + 2 > len(payload):
                    return None
                nal_size = int.from_bytes(payload[position : position + 2], "big")
                position += 2
                if nal_size == 0 or position + nal_size > len(payload) or payload[position] & 0x80:
                    return None
                max_nri = max(max_nri, payload[position] & 0x60)
                member_type = payload[position] & 0x1F
                found_vcl |= nal_size >= 2 and 1 <= member_type <= 5
                position += nal_size
            return evidence(complete=marker) if found_vcl and payload[0] & 0x60 == max_nri else None

        if (
            len(payload) < 2
            or payload[0] & 0x81
            or payload[1] & 0xF8
            or payload[1] & 0x07 == 0
        ):
            return None
        nal_type = payload[0] >> 1 & 0x3F
        if nal_type <= 31:
            return evidence(complete=marker) if len(payload) >= 3 else None
        if nal_type == 49:
            if len(payload) < 4 or payload[2] & 0x20:
                return None
            start, end = bool(payload[2] & 0x80), bool(payload[2] & 0x40)
            if start and end or payload[2] & 0x3F > 31:
                return None
            return evidence(
                fragment=True,
                start=start,
                end=end,
                signature=(payload[1] & 0x07) << 8 | (payload[2] & 0x3F),
            )
        if nal_type != 48:
            return None
        position = 2
        found_vcl = False
        while position < len(payload):
            if position + 2 > len(payload):
                return None
            nal_size = int.from_bytes(payload[position : position + 2], "big")
            position += 2
            if nal_size < 2 or position + nal_size > len(payload):
                return None
            nal = payload[position : position + nal_size]
            if nal[0] & 0x81 or nal[1] & 0xF8 or nal[1] & 0x07 == 0:
                return None
            found_vcl |= nal_size >= 3 and nal[0] >> 1 & 0x3F <= 31
            position += nal_size
        return evidence(complete=marker) if found_vcl else None

    def _completes_vcl_access_unit(self, index: int, evidence) -> bool:
        complete, fragment, start, end, marker, sequence, timestamp, signature = evidence
        if complete:
            self._fu_progress[index] = None
            return True
        if not fragment:
            return False
        if start:
            if marker:
                self._fu_progress[index] = None
                return False
            self._fu_progress[index] = (timestamp, (sequence + 1) & 0xFFFF, signature)
            return False
        if self._fu_progress[index] != (timestamp, sequence, signature):
            self._fu_progress[index] = None
            return False
        self._fu_progress[index] = (timestamp, (sequence + 1) & 0xFFFF, signature)
        if end:
            self._fu_progress[index] = None
            return marker
        if marker:
            self._fu_progress[index] = None
            return False
        return False

    @property
    def received_all_ports(self) -> bool:
        with self._lock:
            return all(
                config > 0 and vcl > 0
                for config, vcl in zip(self._config_counts, self._vcl_counts, strict=True)
            )

    def has_correlated_timestamp(self, port_offset: int, timestamp_ms: int) -> bool:
        expected = timestamp_ms * 90 & 0xFFFFFFFF
        with self._lock:
            return any(
                min((actual - expected) & 0xFFFFFFFF, (expected - actual) & 0xFFFFFFFF) <= 89
                for actual in self._vcl_timestamps[port_offset]
            )


def test_rtp_video_packet_validation():
    h264 = RtpVideoListener(INSIGHT_HOST, 0, 0, "h264")
    h265 = RtpVideoListener(INSIGHT_HOST, 0, 0, "h265")
    header = bytes([0x80, 96, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1])
    h264_sps = header + bytes([0x67, 0x01])
    h265_vps = header + bytes([0x40, 0x01, 0x01])
    assert h264._is_codec_config_rtp(h264_sps)
    assert not h265._is_codec_config_rtp(h264_sps)
    assert h265._is_codec_config_rtp(h265_vps)
    assert not h264._is_codec_config_rtp(h265_vps)
    # Ordinary slice headers overlap and therefore cannot prove the codec.
    assert not h264._is_codec_config_rtp(header + bytes([0x61, 0x01]))
    assert not h265._is_codec_config_rtp(header + bytes([0x61, 0x01]))
    assert not h264._is_codec_config_rtp(header + bytes([0x02, 0x01]))
    assert not h265._is_codec_config_rtp(header + bytes([0x02, 0x01]))
    assert not h265._is_codec_config_rtp(header + bytes([0x41, 0x9A, 0x01]))
    assert h264._is_codec_config_rtp(header + bytes([0x78, 0, 2, 0x67, 1]))
    assert h264._is_codec_config_rtp(header + bytes([28, 0x87, 1]))
    assert h265._is_codec_config_rtp(header + bytes([0x60, 1, 0, 3, 0x40, 1, 1]))
    assert h265._is_codec_config_rtp(header + bytes([0x62, 1, 0xA0, 1]))
    assert not h264._is_codec_config_rtp(bytes([0x80, 97]) + h264_sps[2:])
    assert not h264._is_codec_config_rtp(header + bytes([24]))
    assert not h264._is_codec_config_rtp(header + bytes([28]))
    assert not h265._is_codec_config_rtp(header + bytes([0x60, 1]))
    assert not h265._is_codec_config_rtp(header + bytes([0x62, 1]))
    assert not h264._is_codec_config_rtp(header + bytes([0x67]))
    assert not h265._is_codec_config_rtp(header + bytes([0x40, 1]))
    assert not h264._is_codec_config_rtp(header + bytes([24, 0, 1, 0x67]))
    assert not h264._is_codec_config_rtp(header + bytes([24, 0, 2, 0x67, 1]))
    assert not h265._is_codec_config_rtp(header + bytes([0x60, 1, 0, 2, 0x40, 1]))
    assert not h265._is_codec_config_rtp(header + bytes([0x60, 1, 0, 3, 0x41, 1, 1]))
    assert not h265._is_codec_config_rtp(header + bytes([0x61, 1, 0, 3, 0x40, 1, 1]))
    assert not h265._is_codec_config_rtp(header + bytes([0x63, 1, 0xA0, 1]))
    assert not h264._is_codec_config_rtp(header + bytes([28, 0xC7, 1]))
    assert not h265._is_codec_config_rtp(header + bytes([0x62, 1, 0xE0, 1]))
    assert not h264._is_codec_config_rtp(b"not-rtp")

    def packet(payload, *, marker=False, sequence=1, timestamp=1):
        return bytes(
            [
                0x80,
                (0x80 if marker else 0) | 96,
                sequence >> 8,
                sequence & 0xFF,
                timestamp >> 24,
                timestamp >> 16 & 0xFF,
                timestamp >> 8 & 0xFF,
                timestamp & 0xFF,
                0,
                0,
                0,
                1,
            ]
        ) + bytes(payload)

    assert h264._vcl_evidence_rtp(h264_sps) is None
    assert h264._vcl_evidence_rtp(packet([0x65, 1], marker=True))[0]
    assert h265._vcl_evidence_rtp(packet([0x26, 1, 1], marker=True))[0]
    h264._fu_progress = [None]
    h264_fragments = [
        packet([28, 0x85, 1], sequence=10, timestamp=99),
        packet([28, 0x05, 1], sequence=11, timestamp=99),
        packet([28, 0x45, 1], marker=True, sequence=12, timestamp=99),
    ]
    assert not h264._completes_vcl_access_unit(0, h264._vcl_evidence_rtp(h264_fragments[0]))
    assert not h264._completes_vcl_access_unit(0, h264._vcl_evidence_rtp(h264_fragments[1]))
    assert h264._completes_vcl_access_unit(0, h264._vcl_evidence_rtp(h264_fragments[2]))
    h265._fu_progress = [None]
    h265_start = packet([0x62, 1, 0x93, 1], sequence=20, timestamp=100)
    h265_end = packet([0x62, 1, 0x53, 1], marker=True, sequence=21, timestamp=100)
    assert not h265._completes_vcl_access_unit(0, h265._vcl_evidence_rtp(h265_start))
    assert h265._completes_vcl_access_unit(0, h265._vcl_evidence_rtp(h265_end))
    h264._config_counts = [1]
    h264._vcl_counts = [0]
    h264._vcl_timestamps = [{99}]
    assert not h264.received_all_ports
    h264._vcl_counts[0] = 1
    assert h264.received_all_ports
    assert h264.has_correlated_timestamp(0, 1)
    assert not h264.has_correlated_timestamp(0, 3)


def runtime_dependencies_ready() -> bool:
    return all(
        importlib.util.find_spec(name) is not None
        for name in ("numpy", "pyneat")
    )


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name, "").strip()
    return int(value) if value else default


def source_caps(url: str) -> tuple[int, int, int]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-rtsp_transport",
            "tcp",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate",
            "-of",
            "json",
            url,
        ],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(f"failed to probe E2E source {url}: {result.stderr}")
    streams = json.loads(result.stdout).get("streams", [])
    if len(streams) != 1:
        raise RuntimeError(f"E2E source must expose one video stream: {url}")
    stream = streams[0]
    numerator, denominator = map(int, stream.get("avg_frame_rate", "0/1").split("/"))
    fps = round(numerator / denominator) if denominator else 0
    caps = (stream.get("width", 0), stream.get("height", 0), fps)
    if any(value <= 0 for value in caps):
        raise RuntimeError(f"failed to resolve E2E source caps: {url}")
    return caps


def is_probability(value) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1


def has_body_points(pose, name: str, axes: str) -> bool:
    """pose[name] holds all named body points with finite coordinates and confidence."""
    points = pose.get(name, [])
    return len(points) == len(LANDMARK_NAMES) and all(
        point.get("name") == LANDMARK_NAMES[index]
        and is_probability(point.get("confidence"))
        and all(
            type(point.get(axis)) in (int, float) and math.isfinite(point[axis])
            for axis in axes
        )
        for index, point in enumerate(points)
    )


def valid_overlay_pose(pose) -> bool:
    bbox = pose.get("bbox")
    return (
        isinstance(pose.get("id"), str)
        and bool(pose["id"])
        and pose.get("label") == "person"
        and is_probability(pose.get("presence"))
        and is_probability(pose.get("confidence"))
        and isinstance(bbox, list)
        and len(bbox) == 4
        and all(type(value) in (int, float) and math.isfinite(value) for value in bbox)
        and bbox[2] >= 0
        and bbox[3] >= 0
        and has_body_points(pose, "keypoints", "xy")
        and has_body_points(pose, "world_keypoints", "xyz")
    )


def valid_auxiliary_pose(pose) -> bool:
    return (
        isinstance(pose.get("id"), str)
        and bool(pose["id"])
        and is_probability(pose.get("presence"))
        and has_body_points(pose, "keypoints", "xyz")
    )


def paired_pose_content_matches(overlay, auxiliary) -> bool:
    return len(overlay) == len(auxiliary) and all(
        overlay_pose["id"] == auxiliary_pose["id"]
        and overlay_pose["presence"] == auxiliary_pose["presence"]
        and overlay_pose["world_keypoints"] == auxiliary_pose["keypoints"]
        for overlay_pose, auxiliary_pose in zip(overlay, auxiliary, strict=True)
    )


def metadata_problem(
    messages,
    metadata_port_base: int,
    num_ports: int,
    video: RtpVideoListener | None = None,
) -> str | None:
    """Every message carries its port's stream id and 33 valid keypoints per pose
    (image and world for 2D poses), every port publishes a 2D/3D pair for one
    frame, and one such pair is non-empty."""
    pose_frames: dict[str, dict[tuple, list]] = {
        "pose-estimation": {},
        "auxiliary-visualization": {},
    }
    for message in messages:
        data = json.loads(message.payload)["data"]
        if data.get("stream_id") != f"camera{message.port - metadata_port_base}":
            return f"metadata on port {message.port} did not carry its stream id"
        if message.metadata_type == "pose-estimation":
            poses = data["poses"]
            valid_pose = valid_overlay_pose
        elif (data.get("id"), data.get("renderer")) == ("world-pose", "blazepose-3d"):
            poses = data["payload"]["poses"]
            valid_pose = valid_auxiliary_pose
        else:
            return "auxiliary metadata did not use the world-pose BlazePose 3D envelope"
        if not all(valid_pose(pose) for pose in poses):
            return f"a {message.metadata_type} pose did not satisfy the advertised schema"
        frame = (message.port, message.timestamp_ms, message.frame_id)
        pose_frames[message.metadata_type][frame] = poses
    world = pose_frames["auxiliary-visualization"]
    paired = {
        frame: poses
        for frame, poses in pose_frames["pose-estimation"].items()
        if frame in world and paired_pose_content_matches(poses, world[frame])
    }
    if not paired:
        return "no stream published a non-empty content-matched 2D/3D pair for one frame"
    if len({port for port, _, _ in paired}) < num_ports:
        return "not every metadata port published a content-matched 2D/3D pair for one frame"
    if video is not None:
        paired = {
            frame: poses
            for frame, poses in paired.items()
            if video.has_correlated_timestamp(
                frame[0] - metadata_port_base,
                frame[1],
            )
        }
        if len({port for port, _, _ in paired}) < num_ports:
            return "not every metadata port published a pair correlated to video PTS"
    if not any(paired.values()):
        return "no stream published a non-empty content-matched 2D/3D pair for one frame"
    return None


def test_metadata_problem_requires_pairs_on_every_port_and_stream_ids():
    def message(
        port,
        metadata_type,
        frame_id,
        poses,
        stream_id=None,
        image_points=33,
        world_z=3.0,
    ):
        data = {"stream_id": stream_id or f"camera{port - 9100}"}
        image_points_data = [
            {"name": name, "x": 1.0, "y": 2, "confidence": 0.9}
            for name in LANDMARK_NAMES[:image_points]
        ]
        world_points_data = []
        for name in LANDMARK_NAMES:
            point = {"name": name, "x": 1.0, "y": 2, "confidence": 0.9}
            if world_z is not None:
                point["z"] = world_z
            world_points_data.append(point)
        if metadata_type == "pose-estimation":
            pose = {
                "id": "pose_1",
                "label": "person",
                "presence": 0.9,
                "confidence": 0.8,
                "bbox": [1, 2, 3, 4],
                "keypoints": image_points_data,
                "world_keypoints": world_points_data,
            }
            data["poses"] = [pose] * poses
        else:
            data.update(id="world-pose", renderer="blazepose-3d")
            data["payload"] = {
                "poses": [
                    {"id": "pose_1", "presence": 0.9, "keypoints": world_points_data}
                ]
                * poses
            }
        return SimpleNamespace(
            port=port,
            metadata_type=metadata_type,
            timestamp_ms=1,
            frame_id=frame_id,
            payload=json.dumps({"data": data}),
        )

    pair = [
        message(9101, "pose-estimation", "1", 2),
        message(9101, "auxiliary-visualization", "1", 2),
    ]
    empty = [
        message(9100, "pose-estimation", "1", 0),
        message(9100, "auxiliary-visualization", "1", 0),
    ]
    assert metadata_problem(empty + pair, 9100, 2) is None
    correlated_video = SimpleNamespace(has_correlated_timestamp=lambda _port, timestamp: timestamp == 1)
    assert metadata_problem(empty + pair, 9100, 2, correlated_video) is None
    uncorrelated_video = SimpleNamespace(has_correlated_timestamp=lambda _port, _timestamp: False)
    assert "correlated to video PTS" in metadata_problem(
        empty + pair, 9100, 2, uncorrelated_video
    )
    assert "every metadata port" in metadata_problem(pair, 9100, 2)
    assert "every metadata port" in metadata_problem(empty + pair[:1], 9100, 2)
    split = [pair[0], message(9101, "auxiliary-visualization", "2", 2)]
    assert "non-empty" in metadata_problem(split, 9100, 2)
    assert "stream id" in metadata_problem([message(9100, "pose-estimation", "1", 1, "x")], 9100, 1)
    short_2d = message(9100, "pose-estimation", "1", 1, image_points=32)
    assert "advertised schema" in metadata_problem([short_2d], 9100, 1)
    missing_overlay_z = message(9100, "pose-estimation", "1", 1, world_z=None)
    assert "advertised schema" in metadata_problem([missing_overlay_z], 9100, 1)
    missing_auxiliary_z = message(9100, "auxiliary-visualization", "1", 1, world_z=None)
    assert "advertised schema" in metadata_problem([missing_auxiliary_z], 9100, 1)
    non_finite_z = message(9100, "auxiliary-visualization", "1", 1, world_z=float("nan"))
    assert "advertised schema" in metadata_problem([non_finite_z], 9100, 1)
    boolean_z = message(9100, "auxiliary-visualization", "1", 1, world_z=True)
    assert "advertised schema" in metadata_problem([boolean_z], 9100, 1)

    def malformed(metadata_type, edit):
        item = message(9100, metadata_type, "1", 1)
        payload = json.loads(item.payload)
        poses = (
            payload["data"]["poses"]
            if metadata_type == "pose-estimation"
            else payload["data"]["payload"]["poses"]
        )
        edit(poses[0])
        item.payload = json.dumps(payload)
        return item

    assert "advertised schema" in metadata_problem(
        [malformed("pose-estimation", lambda pose: pose.pop("bbox"))], 9100, 1
    )
    assert "advertised schema" in metadata_problem(
        [malformed("pose-estimation", lambda pose: pose.update(confidence=True))], 9100, 1
    )
    assert "advertised schema" in metadata_problem(
        [malformed("auxiliary-visualization", lambda pose: pose.update(presence=2.0))], 9100, 1
    )
    assert "advertised schema" in metadata_problem(
        [malformed("pose-estimation", lambda pose: pose["keypoints"][0].pop("name"))], 9100, 1
    )
    assert "advertised schema" in metadata_problem(
        [
            malformed(
                "auxiliary-visualization",
                lambda pose: pose["keypoints"][0].update(confidence=float("nan")),
            )
        ],
        9100,
        1,
    )
    overlay = message(9100, "pose-estimation", "matched", 1)
    for field, value in (
        ("id", "pose_2"),
        ("presence", 0.8),
        ("keypoints", None),
    ):
        auxiliary = message(9100, "auxiliary-visualization", "matched", 1)
        payload = json.loads(auxiliary.payload)
        pose = payload["data"]["payload"]["poses"][0]
        if field == "keypoints":
            pose["keypoints"][0]["x"] += 1
        else:
            pose[field] = value
        auxiliary.payload = json.dumps(payload)
        assert "content-matched" in metadata_problem([overlay, auxiliary], 9100, 1)


@pytest.mark.e2e
class TestE2E:
    @pytest.mark.parametrize(
        ("codec", "urls_fixture"),
        [("h264", "rtsp_h264_urls"), ("h265", "rtsp_h265_urls")],
    )
    def test_multistream_pose_metadata(
        self,
        request,
        codec,
        urls_fixture,
        models_dir,
        test_timeout_ms,
        skip_unless_e2e_ready,
        e2e_config_writer,
    ):
        # The application accepts at most four streams; CI provides five per codec.
        urls = request.getfixturevalue(urls_fixture)[:MAX_STREAMS]
        detector_model = Path(
            os.environ.get(
                "SIMANEAT_APPS_TEST_DETECTOR_MODEL", models_dir / DETECTOR_MODEL
            )
        )
        pose_model = Path(
            os.environ.get(
                "SIMANEAT_APPS_TEST_BLAZEPOSE_MODEL", models_dir / POSE_MODEL
            )
        )
        skip_unless_e2e_ready(
            runtime_dependencies_ready(), "pyneat runtime dependencies unavailable"
        )
        skip_unless_e2e_ready(
            detector_model.is_file(), f"missing YOLO26 package: {detector_model}"
        )
        skip_unless_e2e_ready(
            pose_model.is_file(), f"missing BlazePose package: {pose_model}"
        )
        metadata_port_base = env_int("SIMANEAT_APPS_TEST_INSIGHT_METADATA_PORT", 9100)
        video_port_base = env_int("SIMANEAT_APPS_TEST_INSIGHT_VIDEO_PORT", 9000)
        streams = []
        for index, url in enumerate(urls):
            width, height, fps = source_caps(url)
            streams.append({
                "id": f"camera{index}",
                "url": url,
                "codec": codec,
                "insight_channel": index,
                "width": width,
                "height": height,
                "fps": fps,
            })
        config = e2e_config_writer(
            {
                "models": {
                    "detector_path": str(detector_model),
                    "pose_path": str(pose_model),
                },
                "streams": streams,
                "pose": {"max_people_per_frame": 2, "presence_threshold": 0.0},
                "runtime": {"frames": 30},
                "output": {
                    "insight": {
                        "host": INSIGHT_HOST,
                        "video_port_base": video_port_base,
                        "metadata_port_base": metadata_port_base,
                    },
                },
            }
        )

        with (
            MetadataJsonListener(
                INSIGHT_HOST,
                metadata_port_base,
                num_ports=len(urls),
                require_all_ports=True,
                metadata_contracts={
                    "pose-estimation": "poses",
                    "auxiliary-visualization": "payload.poses",
                },
            ) as listener,
            RtpVideoListener(
                INSIGHT_HOST, video_port_base, len(urls), codec
            ) as video,
        ):
            process = subprocess.run(
                [sys.executable, str(MAIN_PY), "--config", str(config)],
                cwd=str(EXAMPLE_DIR),
                capture_output=True,
                text=True,
                check=False,
                timeout=test_timeout_ms / 1000,
            )
            # The application has exited; drain the buffered metadata until every
            # port holds a 2D/3D pair, one of them non-empty, or a message is invalid.
            messages = []
            problem = "no metadata was received"
            deadline = time.monotonic() + 10.0
            while problem is not None and time.monotonic() < deadline:
                remaining = max(0.0, deadline - time.monotonic())
                messages.extend(listener.wait_for_messages(min(1.0, remaining)).messages)
                problem = metadata_problem(messages, metadata_port_base, len(urls), video)

        assert process.returncode == 0, (
            f"main.py exited with {process.returncode}\n"
            f"stdout:\n{process.stdout}\nstderr:\n{process.stderr}"
        )
        assert video.received_all_ports, (
            f"{codec} video was not emitted on every configured video port"
        )
        assert problem is None, problem
