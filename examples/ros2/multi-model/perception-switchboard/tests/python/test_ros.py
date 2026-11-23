"""Real ROS callbacks with controlled inference: switching, ownership and failures.

Run on Modalix after colcon build and source install/setup.bash. These checks do
not claim model accuracy; on-board model comparisons are documented separately.
"""
import json
from pathlib import Path
from queue import Queue, Empty
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
import pyneat
import pytest
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.executors import SingleThreadedExecutor
from builtin_interfaces.msg import Time
from std_msgs.msg import String

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'src/python'))
from yolo_node import YoloNode
from depth_node import DepthNode
from vlm_node import VLMNode
from output_node import OutputNode
from image_transport import to_message


def frame(number=1):
    return to_message(np.full((8, 8, 3), number, np.uint8), Time(sec=number), 'test_camera')


class Runner:
    def __init__(self):
        self.inputs = Queue()
        self.pushed = threading.Event()
        self.release = threading.Event()
        self.closed = False
    def push(self, samples):
        self.inputs.put(samples[0])
        self.pushed.set()
        return True
    def pull(self, timeout_ms):
        if not self.release.wait(timeout_ms / 1000):
            return None
        try:
            return self.inputs.get(timeout=timeout_ms / 1000)
        except Empty:
            return None
    def close(self):
        self.closed = True
        self.release.set()


@pytest.fixture
def ros():
    rclpy.init()
    executor = SingleThreadedExecutor()
    nodes = []
    def until(predicate, timeout=5):
        deadline = time.monotonic()+timeout
        while not predicate():
            assert time.monotonic() < deadline, 'ROS condition timed out'
            executor.spin_once(timeout_sec=.01)
    def add(node):
        nodes.append(node)
        executor.add_node(node)
        return node
    yield SimpleNamespace(add=add, until=until, executor=executor)
    for node in nodes:
        if hasattr(node, 'close'): node.close()
        executor.remove_node(node)
        node.destroy_node()
    executor.shutdown()
    rclpy.try_shutdown()


def make_controlled(base, ros, runner, **settings):
    class Controlled(base):
        def prepare_input(self, image):
            return [pyneat.Tensor.from_numpy(np.zeros((8,8,3),np.uint8), copy=True)]
        def initialize_model(self, tensors, image):
            return runner
        def decode_result(self, sample, context):
            if base is YoloNode:
                result = String(data=str(context['image'].header.stamp.sec))
            else:
                result = to_message(np.ones((8,8),np.float32), context['image'].header.stamp,
                                    context['image'].header.frame_id, '32FC1')
            return result, None
    node = ros.add(Controlled(parameter_overrides=[Parameter(k, value=v) for k,v in settings.items()]))
    return node


@pytest.mark.parametrize('base', [YoloNode, DepthNode])
def test_standby_drain_and_reenable_never_publish_old_generation(ros, base):
    runner = Runner()
    node = make_controlled(base, ros, runner, enabled=False)
    node.on_image(frame(1))
    assert node.pending is None and node.runner is None and node.received == 0
    assert node.set_parameters_atomically([Parameter('enabled', value=True)]).successful
    node.on_image(frame(2))
    ros.until(runner.pushed.is_set)
    assert node.set_parameters_atomically([Parameter('enabled', value=False)]).successful
    assert node.set_parameters_atomically([Parameter('enabled', value=True)]).successful
    runner.release.set()
    ros.until(lambda: node.completed == 1)
    assert node.published == 0 and node.discarded == 1
    node.on_image(frame(3))
    ros.until(lambda: node.published == 1)
    assert node.error is None and node.runner is runner
    node.set_parameters_atomically([Parameter('enabled', value=False)])
    ros.until(lambda: not node.inflight)
    completed = node.completed
    node.on_image(frame(4))
    assert node.completed == completed and node.pending is None


def test_invalid_atomic_update_does_not_disable_yolo(ros):
    node = make_controlled(YoloNode, ros, Runner(), enabled=True)
    result = node.set_parameters_atomically([Parameter('enabled', value=False),
                                             Parameter('overlay_mode', value='invalid')])
    assert not result.successful
    assert node.enabled and node.generation == 0
    assert node.get_parameter('enabled').value is True
    assert node.set_parameters_atomically([Parameter('overlay_mode', value='segmentation')]).successful
    assert node.overlay_mode == 'segmentation' and node.generation == 0
    assert not node.set_parameters_atomically([Parameter('model_path', value='new-model')]).successful


def test_pending_images_are_bounded_and_replaced(ros):
    runner = Runner()
    node = make_controlled(YoloNode, ros, runner, inflight_frames=1)
    node.on_image(frame(1))
    ros.until(runner.pushed.is_set)
    for i in range(2, 20): node.on_image(frame(i))
    assert len(node.inflight) == 1 and node.pending.header.stamp.sec == 19
    assert node.replaced == 17
    runner.release.set()
    ros.until(lambda: node.completed == 2)
    assert node.published == 2


def test_inference_timeout_fails_visibly(ros):
    node = make_controlled(YoloNode, ros, Runner(), timeout_seconds=.1)
    node.on_image(frame())
    ros.until(node.stopping.is_set)
    assert 'timed out' in node.error


def test_shutdown_releases_pending_native_work(ros):
    runner = Runner()
    node = make_controlled(DepthNode, ros, runner, enabled=True)
    node.on_image(frame())
    ros.until(runner.pushed.is_set)
    node.close()
    assert runner.closed and not node.feeder.is_alive() and not node.consumer.is_alive()


def test_vlm_snapshots_full_frame_rejects_busy_and_recovers(ros):
    node = ros.add(VLMNode())
    observer = ros.add(Node('answer_observer'))
    answers = []
    observer.create_subscription(String, 'vlm/answer', lambda m: answers.append(json.loads(m.data)), 8)
    ros.until(lambda: node.answers.get_subscription_count() == 1)
    entered, release = threading.Event(), threading.Event()
    seen = []
    def infer(image, prompt, on_chunk):
        seen.append((image.header.stamp.sec, image.width, prompt))
        entered.set()
        assert release.wait(3)
        return 'test answer'
    node.infer = infer
    node.on_question(String(data='first'))
    ros.until(lambda: len(answers) == 1)
    assert 'recent image' in answers[0]['error']
    node.on_image(frame(2))
    node.on_question(String(data='describe'))
    ros.until(entered.is_set)
    node.on_image(frame(3))
    node.on_question(String(data='second'))
    ros.until(lambda: len(answers) == 2)
    assert 'busy' in answers[-1]['error']
    release.set()
    ros.until(lambda: len(answers) == 3)
    assert seen == [(2, 8, 'describe')]
    assert answers[-1]['header']['stamp']['sec'] == 2
    assert answers[-1]['text'] == 'test answer'
    node.infer = lambda *_: (_ for _ in ()).throw(ConnectionError('backend unavailable'))
    node.on_question(String(data='try again'))
    ros.until(lambda: len(answers) == 4)
    assert answers[-1]['error'] == 'backend unavailable' and not node.busy.is_set()
    node.latest = (frame(4), time.monotonic()-5)
    node.on_question(String(data='stale'))
    ros.until(lambda: len(answers) == 5)
    assert 'recent image' in answers[-1]['error']


def test_output_defaults_do_not_require_insight_or_create_encoders(ros):
    node = ros.add(OutputNode())
    assert all(not output.enabled for output in node.outputs.values())
    assert not node.pending and node.error is None


def test_main_closes_workers_before_ros_context_shutdown(monkeypatch):
    import node_tools
    closed_with_live_context = []
    class MinimalNode:
        stopping = threading.Event()
        error = None
        def close(self):
            closed_with_live_context.append(rclpy.ok())
        def destroy_node(self):
            pass
    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt()
    monkeypatch.setattr(rclpy, 'spin_once', interrupt)
    node_tools.run(MinimalNode)
    assert closed_with_live_context == [True]
    assert not rclpy.ok()


def test_signal_requests_stop_without_interrupting_callback(monkeypatch):
    import os
    import signal
    import node_tools
    observed = []
    class MinimalNode:
        stopping = threading.Event()
        error = None
        def close(self):
            observed.append(('closed', rclpy.ok()))
        def destroy_node(self):
            pass
    def callback(*args, **kwargs):
        os.kill(os.getpid(), signal.SIGINT)
        observed.append(('callback_finished', True))
    previous = signal.getsignal(signal.SIGINT)
    monkeypatch.setattr(rclpy, 'spin_once', callback)
    node_tools.run(MinimalNode)
    assert observed == [('callback_finished', True), ('closed', True)]
    assert signal.getsignal(signal.SIGINT) is previous


def test_vlm_preloads_without_question_and_retains_answer(ros):
    from rclpy.qos import QoSProfile, DurabilityPolicy
    loaded = threading.Event()
    class PreloadedVLM(VLMNode):
        def initialize_server(self):
            loaded.set()
            self.publish_status('ready')
        def infer(self, image, prompt, on_chunk):
            return 'retained answer'
    node = ros.add(PreloadedVLM(parameter_overrides=[Parameter('start_server', value=True)]))
    ros.until(loaded.is_set)
    assert node.request_id == 0 and node.jobs.empty()
    node.on_image(frame(7))
    node.on_question(String(data='describe'))
    ros.until(lambda: not node.busy.is_set())
    observer = ros.add(Node('late_answer_observer'))
    answers = []
    observer.create_subscription(String, 'vlm/answer', lambda m: answers.append(json.loads(m.data)),
        QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
    ros.until(lambda: bool(answers))
    assert answers[-1]['text'] == 'retained answer'
    assert answers[-1]['header']['stamp']['sec'] == 7


def test_vlm_failed_preload_keeps_node_alive_and_retries_on_question(ros):
    attempts = []
    class RetryVLM(VLMNode):
        def initialize_server(self):
            attempts.append(1)
            if len(attempts) == 1:
                raise RuntimeError('model unavailable')
        def infer(self, image, prompt, on_chunk):
            self.initialize_server()
            return 'recovered'
    node = ros.add(RetryVLM(parameter_overrides=[Parameter('start_server', value=True)]))
    ros.until(lambda: len(attempts) == 1)
    observer = ros.add(Node('retry_observer'))
    answers = []
    observer.create_subscription(String, 'vlm/answer', lambda m: answers.append(json.loads(m.data)), 8)
    ros.until(lambda: node.answers.get_subscription_count() == 1)
    node.on_image(frame())
    node.on_question(String(data='retry'))
    ros.until(lambda: bool(answers))
    assert answers[-1]['text'] == 'recovered' and len(attempts) == 2
    assert node.error is None and not node.stopping.is_set()


@pytest.mark.parametrize('backend_error', [False, True])
def test_vlm_streams_before_completion_with_stable_request_id(ros, monkeypatch, backend_error):
    import vlm_node
    release = threading.Event()
    class Response:
        def __enter__(self): return self
        def __exit__(self, *_): pass
        def __iter__(self):
            yield b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n'
            yield b'data: {"choices":[{"delta":{"content":"A car"}}]}\n'
            assert release.wait(4), 'test did not release the response'
            if backend_error:
                yield b'data: {"error":"backend failed"}\n'
            else:
                yield b'data: {"choices":[{"delta":{"content":" is moving."}}]}\n'
                yield b'data: [DONE]\n'
    monkeypatch.setattr(vlm_node, 'urlopen', lambda *_args, **_kwargs: Response())
    node = ros.add(VLMNode())
    observer = ros.add(Node('chunk_observer'))
    chunks, answers = [], []
    observer.create_subscription(String, 'vlm/chunks', lambda m: chunks.append(json.loads(m.data)), 128)
    observer.create_subscription(String, 'vlm/answer', lambda m: answers.append(json.loads(m.data)), 8)
    ros.until(lambda: node.chunks.get_subscription_count() == 1 and node.answers.get_subscription_count() == 1)
    try:
        node.on_image(frame(9))
        node.on_question(String(data='describe'))
        ros.until(lambda: bool(chunks))
        assert chunks[0]['text'] == 'A car' and not answers and node.busy.is_set()
        # A rejected concurrent question must not change the active chunk ID.
        node.on_question(String(data='another'))
        ros.until(lambda: bool(answers))
        assert answers[0]['request_id'] == 2 and 'busy' in answers[0]['error']
        release.set()
        ros.until(lambda: len(answers) == 2)
        if backend_error:
            assert len(chunks) == 1 and answers[-1]['error'] == 'backend failed'
        else:
            ros.until(lambda: len(chunks) == 2)
            assert ''.join(c['text'] for c in chunks) == answers[-1]['text'] == 'A car is moving.'
        assert all(c['request_id'] == 1 for c in chunks)
        assert [c['elapsed_seconds'] for c in chunks] == sorted(c['elapsed_seconds'] for c in chunks)
        assert chunks[0]['elapsed_seconds'] >= 0
        assert answers[-1]['request_id'] == 1
    finally:
        release.set()
