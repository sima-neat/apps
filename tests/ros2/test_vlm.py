"""Standalone VLM app: full-image questions, crop-only periodic mode and bounded work."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from builtin_interfaces.msg import Time
import yaml

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / 'examples/ros2/genai/detection-to-vlm-assistant'
from tests.ros2.local_modules import load_node
VLMNode = load_node(APP / 'src/python/vlm_node.py', 'tested_vlm').VLMNode
detection_node = load_node(APP / 'src/python/detection_node.py', 'tested_vlm_detection')
DetectionNode = detection_node.DetectionNode
transport = load_node(APP / 'src/python/image_transport.py', 'tested_vlm_images')
to_message, image_array = transport.to_message, transport.image_array

pytestmark = pytest.mark.ros2


def settings(mode='on_demand'):
    config = yaml.safe_load((APP / 'src/common/config.yaml').read_text())
    config['model']['labels'] = str(APP / 'src/common/coco_label.txt')
    config['genai'].update(mode=mode, interval_seconds=0.05, image_max_age_seconds=0.5)
    return config


def image(pixel=7, stamp=123, shape=(12, 16, 3)):
    return to_message(np.full(shape, pixel, np.uint8), Time(sec=stamp), 'camera', 'rgb8')


@pytest.fixture
def setup():
    rclpy.init()
    nodes = []
    executor = SingleThreadedExecutor()
    observer = Node('vlm_test_client')
    executor.add_node(observer)
    answers, previews = [], []
    observer.create_subscription(String, 'captions', lambda m: answers.append(json.loads(m.data)), 8)
    observer.create_subscription(Image, 'vlm_input', previews.append, 1)
    images = observer.create_publisher(Image, 'image', 1)
    crops = observer.create_publisher(Image, 'person_crop', 1)
    questions = observer.create_publisher(String, 'question', 8)

    def until(predicate, timeout=4):
        deadline = time.monotonic() + timeout
        while not predicate():
            assert time.monotonic() < deadline, 'Timed out waiting for ROS response'
            executor.spin_once(timeout_sec=0.01)

    def make(infer=lambda pixels, prompt: prompt, mode='on_demand'):
        node = VLMNode(settings(mode))
        node.infer = infer
        nodes.append(node)
        executor.add_node(node)
        source = images if mode == 'on_demand' else crops
        until(lambda: source.get_subscription_count() == 1 and questions.get_subscription_count() == 1
              and node.captions.get_subscription_count() == 1 and node.inputs.get_subscription_count() == 1)
        return node

    yield SimpleNamespace(make=make, until=until, answers=answers, previews=previews,
                          images=images, crops=crops, questions=questions, executor=executor)
    for node in nodes:
        node.close()
        executor.remove_node(node)
        node.destroy_node()
    executor.shutdown()
    observer.destroy_node()
    rclpy.try_shutdown()


def test_on_demand_uses_full_image_ignores_crops_and_preserves_snapshot(setup):
    started, release = threading.Event(), threading.Event()
    calls = []

    def infer(pixels, prompt):
        calls.append((pixels.copy(), prompt))
        started.set()
        assert release.wait(3)
        return 'Full scene answer'

    node = setup.make(infer)
    try:
        setup.images.publish(image())
        setup.crops.publish(image(99, 456, shape=(2, 3, 3)))
        setup.until(lambda: node.latest_image is not None)
        assert setup.crops.get_subscription_count() == 0
        end = time.monotonic() + 0.15
        while time.monotonic() < end:
            setup.executor.spin_once(timeout_sec=0.01)
        assert not calls
        setup.questions.publish(String(data='Describe the scene'))
        setup.until(started.is_set)
        setup.images.publish(image(88, 789))
        setup.until(lambda: node.latest_image[0].header.stamp.sec == 789)
        setup.questions.publish(String(data='Another question'))
        setup.until(lambda: len(setup.answers) == 1)
        assert 'busy' in setup.answers[0]['error']
        release.set()
        setup.until(lambda: len(setup.answers) == 2 and setup.previews)
        answer = setup.answers[1]
        assert answer['header']['stamp']['sec'] == 123
        assert answer['input_kind'] == 'full_image'
        assert (answer['width'], answer['height']) == (16, 12)
        assert answer['prompt'] == 'Describe the scene'
        assert answer['text'] == 'Full scene answer'
        assert calls[0][0].shape == (12, 16, 3) and np.all(calls[0][0] == 7)
        assert np.array_equal(image_array(setup.previews[0]), calls[0][0])
    finally:
        release.set()


def test_periodic_uses_crop_and_predefined_prompt(setup):
    calls = []
    node = setup.make(lambda pixels, prompt: calls.append((pixels.copy(), prompt)) or 'Person answer', mode='periodic')
    assert setup.images.get_subscription_count() == 0
    setup.images.publish(image(99))
    setup.crops.publish(image(11, 456, shape=(2, 3, 3)))
    setup.until(lambda: any(answer['text'] for answer in setup.answers))
    answer = setup.answers[0]
    assert answer['input_kind'] == 'person_crop'
    assert (answer['width'], answer['height']) == (3, 2)
    assert calls[0][1] == node.genai['user_prompt']
    assert np.all(calls[0][0] == 11)
    setup.questions.publish(String(data='Manual question'))
    setup.until(lambda: any(answer['error'] for answer in setup.answers))
    assert any('on_demand' in (answer['error'] or '') for answer in setup.answers)


def test_missing_stale_and_empty_questions_do_not_invoke_vlm(setup):
    calls = []
    node = setup.make(lambda *args: calls.append(args))
    node.on_question(String(data='Where?'))
    node.latest_image = (image(), time.monotonic() - 2)
    node.on_question(String(data='Where now?'))
    node.on_question(String(data='  '))
    setup.until(lambda: len(setup.answers) == 3)
    assert all(answer['error'] for answer in setup.answers)
    assert not calls


def test_backend_failure_releases_request_slot(setup):
    calls = []

    def infer(pixels, prompt):
        calls.append(prompt)
        if len(calls) == 1:
            raise TimeoutError('backend timeout')
        return 'Recovered'

    node = setup.make(infer)
    node.on_image(image())
    node.on_question(String(data='First'))
    setup.until(lambda: len(setup.answers) == 1)
    assert setup.answers[0]['error'] == 'backend timeout'
    node.on_image(image())
    node.on_question(String(data='Second'))
    setup.until(lambda: len(setup.answers) == 2)
    assert setup.answers[1]['text'] == 'Recovered'


def test_shutdown_during_request_suppresses_late_answer(setup):
    started, release = threading.Event(), threading.Event()

    def infer(pixels, prompt):
        started.set()
        assert release.wait(3)
        return 'Late answer'

    node = setup.make(infer)
    node.on_image(image())
    node.on_question(String(data='Describe'))
    setup.until(started.is_set)
    closer = threading.Thread(target=node.close)
    closer.start()
    try:
        setup.until(node.stopping.is_set)
    finally:
        release.set()
        closer.join(timeout=3)
    assert not closer.is_alive() and not node.worker.is_alive()
    assert not setup.answers


def test_preview_fits_changing_crop_sizes_without_changing_input(setup):
    node = setup.make()
    previews = []
    node.preview.write = lambda message, data, sequence: previews.append(message)
    for shape in ((12, 16, 3), (30, 7, 3)):
        original = image(shape=shape)
        node.preview_image = original
        node.show_input()
        assert (previews[-1].width, previews[-1].height) == (640, 480)
        assert (original.height, original.width) == shape[:2]
        assert previews[-1].header == original.header


def test_detector_selects_highest_score_person_and_exact_crop(monkeypatch):
    boxes = [dict(bbox=[1, 1, 4, 3], score=0.7, class_id=0),
             dict(bbox=[2, 2, 5, 4], score=0.9, class_id=0),
             dict(bbox=[0, 0, 8, 6], score=0.99, class_id=1)]
    monkeypatch.setattr(detection_node, 'parse_boxes', lambda sample: boxes)
    sent = []
    frame = image()
    node = SimpleNamespace(config={'inference': {}}, lock=threading.Lock(), latest_person=None,
                           label=lambda value: ['person', 'car'][value], completed=0, first=None,
                           visuals=SimpleNamespace(write=lambda *args: None),
                           results=SimpleNamespace(publish=lambda result: None),
                           crops=SimpleNamespace(get_subscription_count=lambda: 1, publish=sent.append),
                           crop_interval=1, last_crop=None)
    DetectionNode.publish_result(node, None, frame)
    DetectionNode.publish_crop(node)
    assert sent[0].header == frame.header
    assert np.array_equal(image_array(sent[0]), image_array(frame)[2:6, 2:7])
    DetectionNode.publish_crop(node)
    assert len(sent) == 1
    boxes.clear()
    DetectionNode.publish_result(node, None, frame)
    assert node.latest_person is None


@pytest.mark.parametrize('source,enabled,server,count', [('live',True,True,4), ('none',True,False,2), ('none',False,True,1)])
def test_local_launch_components(tmp_path, monkeypatch, source, enabled, server, count):
    from launch import LaunchContext
    from launch.actions import ExecuteProcess
    path = tmp_path / 'config.yaml'
    config = settings()
    config['genai']['enabled'] = enabled
    path.write_text(yaml.safe_dump(config))
    spec = importlib.util.spec_from_file_location('vlm_local_launch', APP / 'launch/app.launch.py')
    launch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launch)
    monkeypatch.setattr(launch, 'get_package_prefix', lambda package: '/test-install')
    context = LaunchContext()
    context.launch_configurations.update(config=str(path), source=source, namespace='neat', server=str(server).lower())
    assert sum(isinstance(action, ExecuteProcess) for action in launch.nodes(context)) == count


@pytest.mark.parametrize('fault', [None, 'timestamp', 'unknown_id', 'blocked_push'])
def test_detector_workers_match_headers_and_stop_on_fault(monkeypatch, fault):
    from array import array
    from queue import Queue, Empty

    class Runner:
        def __init__(self):
            self.outputs = Queue()
            self.closed = threading.Event()
            self.entered = threading.Event()

        def push(self, samples):
            self.entered.set()
            if fault == 'blocked_push':
                self.closed.wait(3)
                return False
            sample = samples[0]
            if fault == 'timestamp':
                sample.pts_ns += 1
            if fault == 'unknown_id':
                sample.frame_id += 999
            self.outputs.put(sample)
            return True

        def pull(self, timeout_ms):
            try:
                return self.outputs.get(timeout=timeout_ms / 1000)
            except Empty:
                return None

        def close(self):
            self.closed.set()

    runner = Runner()
    monkeypatch.setattr(DetectionNode, 'initialize_model', lambda self, image: setattr(self, 'runner', runner))
    monkeypatch.setattr(detection_node, 'parse_boxes', lambda sample: [])
    rclpy.init()
    node = DetectionNode(settings())
    observer = Node('detector_test_observer')
    results = []
    observer.create_subscription(String, 'results', lambda m: results.append(json.loads(m.data)), 16)
    try:
        deadline = time.monotonic() + 4
        while node.results.get_subscription_count() == 0:
            assert time.monotonic() < deadline
            rclpy.spin_once(observer, timeout_sec=0.01)
        frame = Image()
        frame.width = frame.height = frame.step = 2
        frame.encoding = 'nv12'
        frame.data = array('B', [100, 100, 100, 100, 128, 128])
        frame.header.stamp.sec = 123
        frame.header.frame_id = 'camera'
        node.on_image(frame)
        deadline = time.monotonic() + 4
        while not (results or node.error or (fault == 'blocked_push' and runner.entered.is_set())):
            assert time.monotonic() < deadline
            rclpy.spin_once(observer, timeout_sec=0.01)
        if fault is None:
            assert results[0]['header']['stamp']['sec'] == 123
            assert results[0]['data'] == {'objects': []}
        elif fault != 'blocked_push':
            assert node.error is not None and not results
    finally:
        node.close()
        assert not node.submitter.is_alive() and not node.receiver.is_alive()
        node.destroy_node()
        observer.destroy_node()
        rclpy.try_shutdown()
