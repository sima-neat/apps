"""Contract tests for support/python/sima_examples/rtsp_probe.py: configured values and the probe."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

APPS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(APPS_ROOT / "support" / "python"))
# The probe itself needs OpenCV; these tests replace it, so a host without OpenCV runs them.
sys.modules.setdefault("cv2", types.ModuleType("cv2"))

from sima_examples import rtsp_probe
from sima_examples.rtsp_probe import RtspStreamInfo, probe_rtsp_stream_info, probe_rtsp_streams

URL = "rtsp://source/stream"


@pytest.fixture
def probed(monkeypatch):
    """Make the probe report a 640x512 stream at 30 FPS and record each probe."""
    calls = []

    def read_stream_info(url, tcp):
        calls.append((url, tcp))
        return RtspStreamInfo(640, 512, 30)

    monkeypatch.setattr(rtsp_probe, "_read_stream_info", read_stream_info)
    return calls


def test_zero_values_come_from_the_stream(probed):
    assert probe_rtsp_stream_info(URL, True) == RtspStreamInfo(640, 512, 30)
    assert probed == [(URL, True)]


def test_configured_values_override_the_probe(probed):
    info = probe_rtsp_stream_info(URL, False, RtspStreamInfo(1280, 720, 0))
    assert info == RtspStreamInfo(1280, 720, 30)


def test_fully_configured_stream_is_not_probed(probed):
    configured = RtspStreamInfo(1280, 720, 25)
    assert probe_rtsp_stream_info(URL, True, configured) == configured
    assert probed == []


@pytest.mark.parametrize(
    ("reported", "message"),
    [
        (RtspStreamInfo(0, 0, 30), "set input.width and input.height"),
        (RtspStreamInfo(640, 512, 0), "set input.fps"),
    ],
)
def test_unknown_value_names_the_key_to_set(monkeypatch, reported, message):
    monkeypatch.setattr(rtsp_probe, "_read_stream_info", lambda url, tcp: reported)
    with pytest.raises(RuntimeError, match=message):
        probe_rtsp_stream_info(URL, True)


def test_streams_keep_their_order_and_each_url_is_probed_once(monkeypatch):
    other = "rtsp://source/other"
    rates = {URL: 30, other: 25}
    calls = []

    def read_stream_info(url, tcp):
        calls.append(url)
        return RtspStreamInfo(640, 512, rates[url])

    monkeypatch.setattr(rtsp_probe, "_read_stream_info", read_stream_info)
    streams = probe_rtsp_streams([other, URL, other], True)
    assert [stream.fps for stream in streams] == [25, 30, 25]
    assert calls == [other, URL]


def test_streams_of_another_frame_size_name_both_urls(monkeypatch):
    small = "rtsp://source/small"
    sizes = {URL: RtspStreamInfo(1280, 720, 30), small: RtspStreamInfo(640, 512, 30)}
    calls = []

    def read_stream_info(url, tcp):
        calls.append(url)
        return sizes[url]

    monkeypatch.setattr(rtsp_probe, "_read_stream_info", read_stream_info)
    with pytest.raises(
        ValueError,
        match=f"same frame size.*{URL} is 1280x720, {small} is 640x512",
    ):
        probe_rtsp_streams([URL, small, "rtsp://source/unprobed"], True)
    # The C++ probe stops at the first stream of another size too.
    assert calls == [URL, small]
