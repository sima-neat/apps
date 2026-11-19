#!/usr/bin/env python3
"""On demand: ask about the full image. Periodic: describe a selected person crop."""
import base64
from copy import deepcopy
import json
from queue import Empty, Queue
import threading
import time
from urllib.request import Request, urlopen

import cv2
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String

from image_transport import header_dict, to_message
from visual_output import VisualOutput
from app_config import read_arguments, image_rgb, positive


class VLMNode(Node):
    def __init__(self, config):
        super().__init__('vlm')
        self.config = config
        self.genai = config['genai']
        ros = config['ros']
        self.mode = self.genai['mode']
        if self.mode not in ('on_demand', 'periodic'):
            raise ValueError('genai.mode must be on_demand or periodic')
        self.max_age = positive(self.genai['image_max_age_seconds'], 'genai.image_max_age_seconds')
        positive(self.genai['timeout_seconds'], 'genai.timeout_seconds')
        interval = positive(self.genai['interval_seconds'], 'genai.interval_seconds')
        self.input_kind = 'full_image' if self.mode == 'on_demand' else 'person_crop'
        input_topic = ros['image_topic'] if self.mode == 'on_demand' else ros['crop_topic']

        # ROS interfaces: the mode selects exactly one image subscription.
        self.create_subscription(Image, input_topic, self.on_image, 1)
        self.create_subscription(String, ros['question_topic'], self.on_question, 8)
        self.captions = self.create_publisher(String, ros['captions_topic'], 8)
        # Retain the exact accepted VLM input for late-joining inspection tools.
        retained = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.inputs = self.create_publisher(Image, ros['vlm_input_topic'], retained)
        if self.mode == 'periodic':
            self.create_timer(interval, self.on_periodic)

        # One active request, no waiting backlog. ROS callbacks remain responsive.
        self.latest_image = None
        self.preview_image = None
        self.request_id = 0
        self.busy = threading.Event()
        self.stopping = threading.Event()
        self.jobs = Queue(maxsize=1)
        self.preview_sequence = 0
        visual = deepcopy(ros.get('visualization', {}))
        visual['save'] = {'enabled': False}
        insight = visual.setdefault('insight', {})
        insight['channel_base'] = int(insight.get('channel_base', 0)) + 1
        insight['channels'] = {}
        self.preview = VisualOutput(visual)
        if self.preview.enabled:
            self.create_timer(0.5, self.show_input)
        self.worker = threading.Thread(target=self.infer_requests, name='vlm-inference')
        self.worker.start()
        self.get_logger().info(f'VLM mode={self.mode}; input={input_topic}; one request at a time')

    def on_image(self, image):
        self.latest_image = (image, time.monotonic())

    def on_question(self, message):
        if self.mode != 'on_demand':
            self.publish_answer(None, message.data, None, '', 'Set genai.mode: on_demand to ask questions')
            return
        self.submit(message.data.strip())

    def on_periodic(self):
        self.submit(self.genai['user_prompt'], periodic=True)

    def submit(self, prompt, periodic=False):
        self.request_id += 1
        request_id = self.request_id
        selected = self.latest_image
        error = None
        if not prompt:
            error = 'Question must not be empty'
        elif selected is None or time.monotonic() - selected[1] > self.max_age:
            error = f'No recent {self.input_kind}; wait for a new image'
        elif self.busy.is_set():
            error = 'VLM busy; wait for its answer'
        if error:
            if not periodic:
                self.publish_answer(request_id, prompt, None, '', error)
            return
        try:
            image = selected[0]
            rgb = image_rgb(image)  # Full image in on-demand mode; no detector crop.
            snapshot = to_message(rgb, image.header.stamp, image.header.frame_id, 'rgb8')
            self.busy.set()
            self.inputs.publish(snapshot)
            self.preview_image = snapshot
            self.jobs.put_nowait((request_id, prompt, snapshot, rgb))
        except Exception as error:
            self.busy.clear()
            self.publish_answer(request_id, prompt, None, '', str(error))

    def infer_requests(self):
        while not self.stopping.is_set():
            try:
                request_id, prompt, image, rgb = self.jobs.get(timeout=0.1)
            except Empty:
                continue
            try:
                text = self.infer(rgb, prompt)  # Actual VLM inference request.
                error_message = None
            except Exception as error:
                text, error_message = '', str(error)
            self.busy.clear()  # Free the slot before a client receives the answer.
            if not self.stopping.is_set():
                self.publish_answer(request_id, prompt, image, text, error_message)

    def infer(self, rgb, prompt):
        # The GenAI service initializes/owns the VLM. Send the same JPEG and
        # OpenAI-compatible streaming request used by the native application.
        ok, encoded = cv2.imencode('.jpg', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        if not ok:
            raise RuntimeError('Cannot encode VLM input')
        image_url = 'data:image/jpeg;base64,' + base64.b64encode(encoded.tobytes()).decode('ascii')
        model = self.genai.get('model') or self.config['genai_server']['model']['name']
        payload = dict(model=model, stream=True, max_tokens=self.genai['max_tokens'], messages=[
            {'role': 'system', 'content': self.genai['system_prompt']},
            {'role': 'user', 'content': [
                {'type': 'image_url', 'image_url': {'url': image_url}},
                {'type': 'text', 'text': prompt}]}])
        endpoint = f"http://{self.genai['host']}:{self.genai['port']}/v1/chat/completions"
        request = Request(endpoint, data=json.dumps(payload).encode(),
                          headers={'Content-Type': 'application/json'}, method='POST')
        text = ''
        with urlopen(request, timeout=self.genai['timeout_seconds']) as response:
            for line in response:
                line = line.decode().strip()
                if line == 'data: [DONE]':
                    break
                if line.startswith('data: '):
                    text += json.loads(line[6:]).get('choices', [{}])[0].get('delta', {}).get('content') or ''
        return text.strip()

    def show_input(self):
        if self.preview_image is not None:
            # Repeat the last accepted input so Insight shows it while the VLM
            # answers. These repeats never cause another inference request.
            # Crops change size. Keep the encoder canvas fixed and preserve the
            # aspect ratio; vlm_input still contains the unresized request pixels.
            image = self.preview_image
            scale = min(640 / image.width, 480 / image.height)
            width, height = max(1, round(image.width * scale)), max(1, round(image.height * scale))
            pixels = cv2.resize(image_rgb(image), (width, height))
            canvas = np.zeros((480, 640, 3), np.uint8)
            x, y = (640 - width) // 2, (480 - height) // 2
            canvas[y:y+height, x:x+width] = pixels
            preview = to_message(canvas, image.header.stamp, image.header.frame_id, 'rgb8')
            self.preview.write(preview, {}, self.preview_sequence)
            self.preview_sequence += 1

    def publish_answer(self, request_id, prompt, image, text, error):
        answer = dict(request_id=request_id, input_kind=self.input_kind, prompt=prompt,
                      header=header_dict(image.header) if image is not None else None,
                      width=image.width if image is not None else None,
                      height=image.height if image is not None else None, text=text, error=error)
        self.captions.publish(String(data=json.dumps(answer)))
        if error:
            self.get_logger().error(f'Request {request_id}: {error}')
        else:
            self.get_logger().info(f'Caption {request_id} ({self.input_kind}): {text}')

    def close(self):
        self.stopping.set()
        self.worker.join()
        self.preview.close()


def main(argv=None):
    config, ros_args = read_arguments(argv)
    rclpy.init(args=ros_args)
    node = None
    try:
        node = VLMNode(config)
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.close()
            node.destroy_node()
        rclpy.try_shutdown()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
