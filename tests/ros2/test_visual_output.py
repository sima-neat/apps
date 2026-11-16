"""Real image rendering and filesystem checks; fake video only tests dispatch policy."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'support/ros2'))
from neat_apps.visual_output import VisualOutput, image_bgr, render_result


def message(camera='camera', sec=7):
    # Include row padding to exercise the source layout, not just contiguous input.
    pixels = np.zeros((64, 100, 3), np.uint8)
    pixels[:, :96] = (10, 20, 30)
    return SimpleNamespace(width=96, height=64, step=300, encoding='bgr8', data=pixels.tobytes(),
                           header=SimpleNamespace(frame_id=camera, stamp=SimpleNamespace(sec=sec, nanosec=123)))


def test_save_exact_frame_and_header_without_mutating_input_or_result(tmp_path):
    msg = message()
    original = msg.data
    data = {'objects': [{'label': 'person', 'bbox': [20, 25, 30, 20], 'confidence': .8}]}
    before = deepcopy(data)
    sink = VisualOutput({'save': {'enabled': True, 'directory': str(tmp_path)}})
    sink.write(msg, data, 12)
    records = [json.loads(line) for line in (sink.directory / 'frames.jsonl').read_text().splitlines()]
    assert records[0]['header'] == {'frame_id': 'camera', 'stamp': {'sec': 7, 'nanosec': 123}}
    assert records[0]['sequence'] == 12
    saved = cv2.imread(str(sink.directory / records[0]['image']))
    assert saved.shape == (64, 96, 3)
    assert tuple(saved[25, 20]) == (0, 220, 0)
    assert tuple(saved[60, 90]) == (10, 20, 30)
    assert msg.data == original and data == before
    other = VisualOutput({'save': {'enabled': True, 'directory': str(tmp_path)}})
    other.write(msg, {}, 12)
    assert other.directory != sink.directory
    sink.close()
    other.close()


def test_disabled_output_does_not_read_image_or_create_files(tmp_path):
    sink = VisualOutput({'save': {'enabled': False, 'directory': str(tmp_path/'unused')}})
    sink.write(None, None, 0)
    sink.close()
    assert not (tmp_path/'unused').exists()


def test_sampling_is_per_stream_and_memory_is_bounded(tmp_path):
    sink = VisualOutput({'max_streams': 2, 'save': {'enabled': True, 'directory': str(tmp_path), 'every_n_frames': 2}})
    for i in range(6):
        sink.write(message(str(i % 2)), {}, i)
    records = [json.loads(line) for line in (sink.directory/'frames.jsonl').read_text().splitlines()]
    assert [r['sequence'] for r in records] == [0, 1, 4, 5]
    with pytest.raises(ValueError, match='max_streams'):
        sink.write(message('third'), {}, 6)
    sink.close()


def test_failed_image_write_is_not_reported_as_saved(tmp_path, monkeypatch):
    monkeypatch.setattr(cv2, 'imwrite', lambda *args: False)
    sink = VisualOutput({'save': {'enabled': True, 'directory': str(tmp_path)}})
    with pytest.raises(RuntimeError, match='Could not save'):
        sink.write(message(), {}, 0)
    assert sink.saved == 0
    assert not (sink.directory/'frames.jsonl').exists()


class Video:
    def __init__(self, settings, channel, shape):
        self.channel, self.frames, self.closed = channel, [], False
    def send(self, frame):
        self.frames.append(frame.copy())
    def close(self):
        self.closed = True


def test_insight_selects_channels_and_throttles_without_dropping_saved_frames(tmp_path, monkeypatch):
    clock = [10.0]
    monkeypatch.setattr('neat_apps.visual_output.time.monotonic', lambda: clock[0])
    sink = VisualOutput({'save': {'enabled': True, 'directory': str(tmp_path)},
                         'insight': {'enabled': True, 'host': 'example.test', 'fps': 10,
                                     'channels': {'front': 2}}}, video_factory=Video)
    sink.write(message('rear'), {}, 0)
    sink.write(message('front'), {}, 1)
    sink.write(message('front'), {}, 2)
    assert sink.saved == 3 and sink.sent == 1
    video = sink.videos['front']
    assert video.channel == 2 and 'rear' not in sink.videos
    clock[0] += .11
    sink.write(message('front'), {}, 3)
    assert sink.sent == 2
    sink.close()
    assert video.closed


def test_insight_visible_stream_limit_does_not_create_extra_encoders():
    sink = VisualOutput({'insight': {'enabled': True, 'host': 'example.test',
                                    'max_visible_streams': 1}}, video_factory=Video)
    sink.write(message('one'), {}, 0)
    sink.write(message('two'), {}, 1)
    assert list(sink.videos) == ['one']
    sink.close()


@pytest.mark.parametrize('settings', [
    {'save': {'enabled': 'false'}}, {'save': {'enabled': True}},
    {'save': {'every_n_frames': 0}}, {'insight': {'enabled': True}},
    {'insight': {'enabled': True, 'host': '<host>'}}, {'insight': {'fps': 0}},
    {'insight': {'channels': {'a': 0, 'b': 0}}}, {'insight': {'channels': {'a': -1}}},
    {'insight': {'enabled': True, 'host': 'localhost', 'video_port_base': 65535}},
])
def test_invalid_output_settings_fail_before_any_frames(settings):
    with pytest.raises(ValueError):
        VisualOutput(settings)


def test_planar_nv12_and_nv16_convert_to_correct_bgr():
    msg = message()
    # Neutral chroma and white luma produce white in either packed YUV layout.
    for encoding, rows in [('nv12', 96), ('nv16', 128)]:
        msg.encoding, msg.step = encoding, msg.width
        pixels = np.full((rows, msg.width), 128, np.uint8)
        pixels[:64] = 235
        msg.data = pixels.tobytes()
        bgr = image_bgr(msg)
        assert bgr.shape == (64, 96, 3)
        assert np.all(bgr == 255)


def test_masks_obb_tracks_and_pose_render_source_coordinates():
    data = {'segments': [{'bbox': [30,30,20,20], 'mask': [[30,30],[50,30],[50,50],[30,50]]}],
            'objects': [{'oriented_bbox': [70,40,20,10,.5]}],
            'tracks': [{'bbox': [5,5,10,10], 'id': '17', 'label': 'person', 'confidence': .9}],
            'poses': [{'keypoints': [{'name':'left_shoulder','x':20,'y':40,'confidence':1},
                                     {'name':'right_shoulder','x':40,'y':40,'confidence':1}]}]}
    before = deepcopy(data)
    frame = render_result(np.zeros((64,96,3), np.uint8), data)
    assert frame[5,5].any() and frame[40,20].any() and frame[35,35].any()
    assert data == before


def test_depth_visualization_preserves_numerical_output():
    values = np.arange(24, dtype=np.float32).reshape(4,6)
    before = values.copy()
    frame = render_result(np.zeros((64,96,3), np.uint8), {'images': [('depth', values, '32FC1')]})
    assert frame.shape == (64,192,3)
    np.testing.assert_array_equal(values, before)
    assert frame[:,96:].any()


def test_auxiliary_output_uses_same_result_as_video():
    received=[]
    class AuxiliaryVideo(Video):
        def send(self, frame, data):
            received.append((frame.copy(),data))
    sink=VisualOutput({'insight':{'enabled':True,'host':'example.test'}},
                      video_factory=AuxiliaryVideo, auxiliary=lambda data:{'world':data['world']})
    data={'world':[[1.,2.,3.]]}
    sink.write(message(),data,0)
    assert received[0][1]=={'world':[[1.,2.,3.]]}
    assert received[0][0].shape==(64,96,3)
    assert data=={'world':[[1.,2.,3.]]}
    sink.close()
