"""End-to-end test: an RTSP H.264, H.265 or MJPEG stream in, H.264 video and the top prompts out to Insight."""

from __future__ import annotations

import importlib.util
import json
import os
import socket
import subprocess
import sys
import threading
from pathlib import Path

import cv2
import numpy as np
import pytest

from tests.utils.metadata_json_listener import MetadataJsonListener

EXAMPLE_DIR = Path(__file__).resolve().parent.parent.parent
PROMPTS_TXT = EXAMPLE_DIR / "src" / "common" / "prompts.txt"
MAIN_PY = EXAMPLE_DIR / "src" / "python" / "main.py"
sys.path.insert(0, str(MAIN_PY.parent))

from clip_tokenizer import ClipTokenizer  # noqa: E402

_SPEC = importlib.util.spec_from_file_location("clip_zero_shot_main", MAIN_PY)
assert _SPEC is not None and _SPEC.loader is not None
main = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = main
_SPEC.loader.exec_module(main)

IMAGE_ENCODER = "clip_vit_b_16_w8a16_image_encoder_mpk.tar.gz"
TEXT_ENCODER = "clip_vit_b_16_w8a16_text_encoder_mpk.tar.gz"
TOKENIZER = "tokenizer.json"
TOKEN_EMBEDDINGS = "token_embeddings.npy"
FRAMES = 60
# The same prompt twice, spelled with different case and spacing, must score the same. The stream content is not
# known, so the test does not assert which prompt wins.
PROMPTS = [
    "a photo of people walking",
    "A Photo of  people   WALKING",
    "a photo of cars on a street",
    "a photo of a forest",
    "a photo of a kitchen",
    "a photo of a beach",
]
SOURCE_CASES = [
    pytest.param({"codec": "h264", "url_fixture": "rtsp_h264_url"}, id="rtsp-h264"),
    pytest.param({"codec": "h265", "url_fixture": "rtsp_h265_url"}, id="rtsp-h265"),
    pytest.param({"codec": "mjpeg", "url_fixture": "rtsp_mjpeg_url"}, id="rtsp-mjpeg"),
]
# Token ids of CLIP ViT-B/16's tokenizer.json, checked against Hugging Face tokenizers.
REAL_TOKENS = {
    "a photo of a dog.": [320, 1125, 539, 320, 1929, 269],
    "A Photo of  TWO   cats!": [320, 1125, 539, 1237, 3989, 256],
    "it's route 66, isn't it?": [585, 568, 5261, 277, 277, 267, 2923, 713, 585, 286],
    "black-and-white colobus": [1449, 268, 537, 268, 1579, 7300, 2840],
}


def read_video_frames(video: socket.socket, stop: threading.Event, frame_timestamps: list[int],
                      payloads: list[bytes]) -> None:
    """Appends the RTP timestamp of each H.264 video frame's last packet, and the payload of every H.264 packet,
    until stop is set."""
    while not stop.is_set():
        try:
            packet = video.recv(65536)
        except TimeoutError:
            continue
        if len(packet) > 12 and packet[0] >> 6 == 2 and packet[1] & 0x7F == 96:
            payloads.append(packet[12 + 4 * (packet[0] & 0x0F):])
            if packet[1] & 0x80:
                frame_timestamps.append(int.from_bytes(packet[4:8], "big"))


def last_video_frame(payloads: list[bytes], path: Path) -> np.ndarray:
    """The last frame of the RTP H.264 payloads, decoded, as I420 rows: Y, then U, then V."""
    stream = bytearray()
    for payload in payloads:
        kind = payload[0] & 0x1F
        if kind == 24:  # STAP-A: NAL units, each after its 2-byte size.
            at = 1
            while at + 2 < len(payload):
                size = int.from_bytes(payload[at:at + 2], "big")
                stream += b"\0\0\0\1" + payload[at + 2:at + 2 + size]
                at += 2 + size
        elif kind == 28:  # FU-A: one NAL unit split over packets, its header rebuilt from the first.
            if payload[1] & 0x80:
                stream += b"\0\0\0\1" + bytes([payload[0] & 0xE0 | payload[1] & 0x1F])
            stream += payload[2:]
        else:
            stream += b"\0\0\0\1" + payload
    path.write_bytes(bytes(stream))
    capture, frame, decoded = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG), None, True
    while decoded:
        decoded, image = capture.read()
        frame = image if decoded else frame
    capture.release()
    assert frame is not None, f"no frame of {path} decodes"
    return cv2.cvtColor(frame, cv2.COLOR_BGR2YUV_I420)


def panel_problem(i420: np.ndarray) -> str:
    """Empty when the frame shows the result panel: its box dimmed, and the best row's bar amber where the same area
    above the box is not."""
    height, width = i420.shape[0] * 2 // 3, i420.shape[1]
    layout = main.panel_layout(width, height, 3)
    x, y, w, h = layout.box
    # Dimming leaves luma at most 0.3 x 255 = 77 everywhere but the text and the filled bars, about a sixth of the
    # box. The area above it is no reference: it was as dark in the MJPEG stream's street scene.
    dimmed = float(np.mean(i420[y:y + h, x:x + w] <= 90))
    if dimmed < 0.75:
        return f"only {dimmed:.2f} of the panel box has luma at most 90"
    u = i420[height:height + height // 4].reshape(height // 2, width // 2)
    v = i420[height + height // 4:].reshape(height // 2, width // 2)
    # The best prompt has more than a sixth of the probability, so its bar's first fifth is filled.
    bar_x, bar_y, bar_width, bar_height = layout.rows[0].bar
    for top, where in ((bar_y, "bar"), (bar_y - h, "area above the box")):
        rows, columns = slice(top // 2, (top + bar_height) // 2), slice(bar_x // 2, (bar_x + bar_width // 5) // 2)
        chroma = (float(np.median(u[rows, columns])), float(np.median(v[rows, columns])))
        amber = abs(chroma[0] - main.ACCENT_YUV[1]) <= 20 and abs(chroma[1] - main.ACCENT_YUV[2]) <= 20
        if amber != (where == "bar"):
            return f"the best row's {where} has chroma U {chroma[0]} V {chroma[1]}"
    return ""


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name, "").strip()
    return int(value) if value else default


def model_files(models_dir: Path, skip_unless_e2e_ready) -> dict[str, str]:
    files = {"image_encoder": IMAGE_ENCODER, "text_encoder": TEXT_ENCODER, "tokenizer": TOKENIZER,
             "token_embeddings": TOKEN_EMBEDDINGS}
    paths = {key: models_dir / name for key, name in files.items()}
    for path in paths.values():
        skip_unless_e2e_ready(path.exists(), f"missing CLIP ViT-B/16 artifact: {path}")
    return {key: str(path) for key, path in paths.items()}


@pytest.mark.e2e
def test_tokenizer_matches_clip(models_dir, skip_unless_e2e_ready):
    tokenizer_json = model_files(models_dir, skip_unless_e2e_ready)["tokenizer"]
    tokenizer = ClipTokenizer(tokenizer_json)
    for prompt, ids in REAL_TOKENS.items():
        assert tokenizer.encode(prompt, 77).tolist() == [49406, *ids] + [49407] * (76 - len(ids)), prompt


@pytest.mark.e2e
@pytest.mark.parametrize("source", SOURCE_CASES)
def test_publishes_insight_video_and_top_prompts(
    request, source, tmp_path, models_dir, test_timeout_ms, skip_unless_e2e_ready, e2e_config_writer
):
    source_url = request.getfixturevalue(source["url_fixture"])
    prompts_path = tmp_path / "prompts.txt"
    prompts_path.write_text("\n".join(PROMPTS) + "\n", encoding="utf-8")
    metadata_port = _env_int("SIMANEAT_APPS_TEST_INSIGHT_METADATA_PORT", 9100)
    video_port = _env_int("SIMANEAT_APPS_TEST_INSIGHT_VIDEO_PORT", 9000)
    config_path = e2e_config_writer(
        {
            "model": model_files(models_dir, skip_unless_e2e_ready),
            "prompts": {"path": str(prompts_path)},
            "source": {"rtsp_url": source_url, "codec": source["codec"]},
            "inference": {"frames": FRAMES},
            "output": {
                "insight": {"host": "127.0.0.1", "video_port": video_port, "metadata_port": metadata_port}
            },
        }
    )

    with MetadataJsonListener(
        "127.0.0.1", metadata_port, num_ports=1, metadata_type="classification", data_array_key="top_classes"
    ) as listener, socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as video:
        video.bind(("127.0.0.1", video_port))
        video.settimeout(0.1)
        stop, frame_timestamps, payloads = threading.Event(), [], []
        reader = threading.Thread(target=read_video_frames, args=(video, stop, frame_timestamps, payloads))
        reader.start()
        try:
            result = subprocess.run(
                [sys.executable, str(MAIN_PY), "--config", str(config_path)],
                capture_output=True,
                text=True,
                timeout=test_timeout_ms / 1000,
                cwd=str(EXAMPLE_DIR),
                check=False,
            )
        finally:
            stop.set()
            reader.join()
        assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        metadata = listener.drain_pending()

    assert f"processed={FRAMES} " in result.stdout
    assert f" codec={source['codec']} prompts={len(PROMPTS)} " in result.stdout
    assert metadata.success, metadata.error
    messages = metadata.messages
    # UDP on loopback can drop a datagram under load, so most of the frames must arrive, not all.
    assert len(frame_timestamps) >= FRAMES * 0.9, f"{len(frame_timestamps)} of {FRAMES} H.264 video frames arrived"
    assert len(messages) >= FRAMES * 0.9, f"{len(messages)} of {FRAMES} metadata messages arrived"
    # Insight draws a message on the video frame whose 90 kHz RTP timestamp is within 1 ms of the message's.
    matched = [m for m in messages if any(abs(m.timestamp_ms * 90 % 2**32 - t) <= 90 for t in frame_timestamps)]
    assert len(matched) >= len(messages) * 0.9, f"{len(matched)} of {len(messages)} messages match a video frame"
    problem = panel_problem(last_video_frame(payloads, tmp_path / "video.h264"))
    assert not problem, problem
    frame_ids = [int(message.frame_id) for message in messages]
    timestamps = [message.timestamp_ms for message in messages]
    assert frame_ids == sorted(frame_ids) and len(set(frame_ids)) == len(frame_ids)
    assert timestamps == sorted(timestamps) and timestamps[-1] > timestamps[0]

    duplicates = {PROMPTS[0], PROMPTS[1]}
    for message in messages:
        top = json.loads(message.payload)["data"]["top_classes"]
        assert len(top) == min(3, len(PROMPTS))
        confidences = [entry["confidence"] for entry in top]
        assert confidences == sorted(confidences, reverse=True)
        for entry in top:
            assert entry["label"] in PROMPTS
            assert 0.0 <= entry["confidence"] <= 1.0
            # CLIP ViT-B/16 places real images at a cosine of about 0.1 to 0.35 from short captions.
            assert 0.05 < entry["similarity"] < 0.5, entry
        both = [entry for entry in top if entry["label"] in duplicates]
        if len(both) == 2:
            assert (both[0]["confidence"], both[0]["similarity"]) == (both[1]["confidence"], both[1]["similarity"])
        # A uniform result would give every prompt 1/6.
        assert confidences[0] > 1.5 / len(PROMPTS), top


@pytest.mark.e2e
def test_wrong_codec_fails_before_the_models_load(
    rtsp_h264_url, tmp_path, models_dir, skip_unless_e2e_ready, e2e_config_writer
):
    # The encoders point at missing files, so the codec error can only come first if the stream is checked before
    # either model loads.
    models = model_files(models_dir, skip_unless_e2e_ready)
    models |= {"image_encoder": str(tmp_path / "missing_image_encoder.tar.gz"),
               "text_encoder": str(tmp_path / "missing_text_encoder.tar.gz")}
    config_path = e2e_config_writer(
        {
            "model": models,
            "prompts": {"path": str(PROMPTS_TXT)},
            "source": {"rtsp_url": rtsp_h264_url, "codec": "h265"},
        }
    )
    result = subprocess.run(
        [sys.executable, str(MAIN_PY), "--config", str(config_path)],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=str(EXAMPLE_DIR),
        check=False,
    )
    assert result.returncode != 0
    assert f"{rtsp_h264_url} is h264, but source.codec is h265; set source.codec to h264" in result.stderr, (
        result.stderr
    )
