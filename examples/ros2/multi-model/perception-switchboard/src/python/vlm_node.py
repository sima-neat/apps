#!/usr/bin/env python3
"""Ask about one full frame: one accepted question, one immutable snapshot, one answer."""
import base64
import json
from pathlib import Path
from queue import Empty, Queue
import threading
import time
from urllib.request import Request, urlopen

import cv2
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile, DurabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String

from image_transport import to_message, header_dict
from visual_output import image_bgr
from node_tools import read_only, run


class VLMNode(Node):
    def __init__(self, **kwargs):
        super().__init__('vlm', **kwargs)
        self.declare_parameter('image_topic', 'image', read_only())
        self.declare_parameter('endpoint', 'http://127.0.0.1:9998/v1/chat/completions', read_only())
        self.declare_parameter('model', 'Qwen3-VL-2B-Instruct-GPTQ-a16w4', read_only())
        self.declare_parameter('max_tokens', 128, read_only())
        self.declare_parameter('image_max_age_seconds', 3.0, read_only())
        self.declare_parameter('timeout_seconds', 180.0, read_only())
        self.declare_parameter('system_prompt', 'Describe the supplied image accurately and concisely.', read_only())
        self.declare_parameter('start_server', False, read_only())
        self.declare_parameter('model_path', '', read_only())
        self.declare_parameter('server_port', 9998, read_only())
        if (self.get_parameter('image_max_age_seconds').value <= 0 or
                self.get_parameter('timeout_seconds').value <= 0 or self.get_parameter('max_tokens').value < 1):
            raise ValueError('Age limit, timeout and token limit must be positive')
        self.create_subscription(Image, self.get_parameter('image_topic').value,
                                 self.on_image, qos_profile_sensor_data)
        self.create_subscription(String, 'vlm/question', self.on_question, 8)
        retained = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.answers = self.create_publisher(String, 'vlm/answer', retained)
        self.chunks = self.create_publisher(String, 'vlm/chunks', 128)
        self.status = self.create_publisher(String, 'vlm/status', retained)
        self.inputs = self.create_publisher(Image, 'vlm/input', retained)
        self.previews = self.create_publisher(Image, 'vlm/preview', 2)
        self.latest = None
        self.request_id = 0
        self.busy = threading.Event()
        self.stopping = threading.Event()
        self.error = None
        self.jobs = Queue(maxsize=1)
        self.server = None
        self.worker = threading.Thread(target=self.answer_questions, name='vlm-worker')
        self.worker.start()

    def on_image(self, image):
        self.latest = (image, time.monotonic())  # No VLM work on incoming images.

    def on_question(self, question):
        self.request_id += 1
        selected = self.latest
        prompt = question.data.strip()
        reason = None
        if not prompt:
            reason = 'Question must not be empty'
        elif self.busy.is_set():
            reason = 'VLM busy; wait for its answer'
        elif selected is None or time.monotonic() - selected[1] > self.get_parameter('image_max_age_seconds').value:
            reason = 'No recent image; wait for the input topic'
        if reason:
            self.publish_answer(self.request_id, prompt, None, '', reason)
            return
        # Retain this exact ROS message. Later image callbacks cannot replace it.
        self.busy.set()
        self.get_logger().info(f'Accepted VLM question {self.request_id}; answer: vlm/answer')
        self.jobs.put_nowait((self.request_id, prompt, selected[0], time.monotonic()))

    def initialize_server(self):
        # Load the optional local service in the worker; image callbacks stay responsive.
        if self.get_parameter('start_server').value and self.server is None:
            self.publish_status('loading')
            self.get_logger().info('Loading local VLM; questions wait until loading finishes')
            import pyneat
            model_path = Path(self.get_parameter('model_path').value)
            if not model_path.is_dir():
                raise ValueError('Set model_path to the prepared VLM directory')
            options = pyneat.GenAIServerOptions()
            options.host, options.port = '127.0.0.1', self.get_parameter('server_port').value
            server = pyneat.GenAIServer(options)
            try:
                server.add_model(model_path, self.get_parameter('model').value)
                server.start()
            except Exception:
                server.stop()
                raise
            self.server = server
            self.publish_status('ready')
            self.get_logger().info('Local VLM ready; waiting for vlm/question')

    def infer(self, image, prompt, on_chunk):
        # Neat's service owns the VLM. This is the actual inference request.
        self.initialize_server()
        ok, jpeg = cv2.imencode('.jpg', image_bgr(image))
        if not ok:
            raise RuntimeError('Cannot encode VLM input')
        url = 'data:image/jpeg;base64,' + base64.b64encode(jpeg.tobytes()).decode('ascii')
        payload = dict(model=self.get_parameter('model').value, stream=True,
            max_tokens=self.get_parameter('max_tokens').value, messages=[
                {'role': 'system', 'content': self.get_parameter('system_prompt').value},
                {'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': url}},
                                            {'type': 'text', 'text': prompt}]}])
        endpoint = self.get_parameter('endpoint').value
        if self.get_parameter('start_server').value:
            endpoint = f"http://127.0.0.1:{self.get_parameter('server_port').value}/v1/chat/completions"
        request = Request(endpoint, data=json.dumps(payload).encode(),
                          headers={'Content-Type': 'application/json'}, method='POST')
        timeout = self.get_parameter('timeout_seconds').value
        deadline = time.monotonic() + timeout
        text = ''
        with urlopen(request, timeout=timeout) as response:
            for line in response:
                if self.stopping.is_set() or time.monotonic() > deadline:
                    raise TimeoutError('VLM response cancelled or timed out')
                line = line.decode().strip()
                if line == 'data: [DONE]':
                    break
                if line.startswith('data: '):
                    event = json.loads(line[6:])
                    if 'error' in event:
                        raise RuntimeError(str(event['error']))
                    for choice in event.get('choices', []):
                        chunk = choice.get('delta', {}).get('content') or ''
                        if chunk:
                            text += chunk
                            on_chunk(chunk)  # Publish immediately, without waiting for completion.
        if not text.strip():
            raise RuntimeError('VLM returned no answer text')
        return text.strip()

    def publish_status(self, state, error=None):
        self.status.publish(String(data=json.dumps(dict(state=state, error=error))))

    def answer_questions(self):
        try:
            if self.get_parameter('start_server').value:
                self.initialize_server()  # Preload only; inference still requires a question.
            else:
                self.publish_status('external')  # External endpoint is checked on a request.
        except Exception as exc:
            self.publish_status('error', str(exc))
            self.get_logger().error(f'VLM startup failed: {exc}; next question retries loading')
        while not self.stopping.is_set():
            try:
                request_id, prompt, image, accepted_at = self.jobs.get(timeout=0.1)
            except Empty:
                continue
            try:
                snapshot = to_message(image_bgr(image), image.header.stamp, image.header.frame_id)
                self.inputs.publish(snapshot)
                self.previews.publish(snapshot)
                self.publish_status('answering')
                answer = self.infer(snapshot, prompt,
                    lambda chunk: self.publish_chunk(request_id, accepted_at, chunk))
                error = None
            except Exception as exc:
                answer, error = '', str(exc)
            self.busy.clear()
            if not self.stopping.is_set():
                self.publish_answer(request_id, prompt, image, answer, error)
                state = 'ready' if self.server is not None else 'external'
                self.publish_status('error' if error else state, error)

    def publish_chunk(self, request_id, accepted_at, text):
        self.chunks.publish(String(data=json.dumps(dict(request_id=request_id, text=text,
            elapsed_seconds=time.monotonic() - accepted_at), allow_nan=False)))

    def publish_answer(self, request_id, prompt, image, answer, error):
        if error:
            self.get_logger().error(f'VLM question {request_id}: {error}')
        else:
            self.get_logger().info(f'VLM question {request_id} answered on vlm/answer')
        self.answers.publish(String(data=json.dumps(dict(request_id=request_id, prompt=prompt,
            header=header_dict(image.header) if image is not None else None,
            text=answer, error=error), allow_nan=False)))

    def close(self):
        self.stopping.set()
        self.worker.join()  # Active HTTP work is bounded by timeout_seconds.
        if self.server is not None:
            self.server.stop()


if __name__ == '__main__':
    run(VLMNode)
