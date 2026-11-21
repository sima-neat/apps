#!/usr/bin/env python3
"""Bounded ROS requests to Studio's existing chat, vision and speech backend.

The native server owns model loading and inference. This node replaces its
client transport, so it uses the same models and generation parameters.
"""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from pathlib import Path
import threading
from urllib import request
import uuid
import cv2
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
import yaml
from image_transport import image_array, header_dict
from studio_tls import backend_ssl_context


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args, ros_args = parser.parse_known_args(argv)
    config_path = args.config.expanduser().resolve(strict=True)
    config = yaml.safe_load(config_path.read_text())
    settings = config.get("ros", {})
    app = config["app"]
    web = app.get("web", {})
    scheme = "https" if web.get("https", True) else "http"
    endpoint = settings.get("studio_url", f"{scheme}://127.0.0.1:{web.get('port', 5000)}").rstrip("/")
    timeout = float(settings.get("request_timeout_seconds", 120))
    capacity = int(settings.get("pending_requests", 1))
    if capacity < 1 or timeout <= 0:
        raise ValueError("ROS request capacity and timeout must be positive")
    rclpy.init(args=ros_args)
    node = Node("neat_genai_studio")
    publisher = node.create_publisher(String, "responses", 16)
    workers = ThreadPoolExecutor(max_workers=capacity, thread_name_prefix="studio-request")
    slots = threading.BoundedSemaphore(capacity)
    stop = threading.Event()
    latest_image = None

    def publish(identity, **data):
        if not stop.is_set():
            publisher.publish(String(data=json.dumps({"id": identity, **data}, allow_nan=False)))

    def receive_image(image):
        nonlocal latest_image
        if image.encoding not in ("bgr8", "rgb8"):
            node.get_logger().error("Studio vision input requires bgr8 or rgb8")
            return
        latest_image = image  # One bounded, immutable ROS message reference.

    def receive(message):
        identity = None
        acquired = False
        try:
            job = json.loads(message.data)
            identity = job["id"]
            if not isinstance(identity, str) or not identity:
                raise ValueError("Request id must be a nonempty string")
            if job.get("operation", "chat") not in ("chat", "speech", "transcribe", "translate"):
                raise ValueError("operation must be chat, speech, transcribe or translate")
            if not isinstance(job.get("body"), dict):
                raise ValueError("Request body must be an object")
            image = latest_image if job.get("use_image", False) else None
            if job.get("use_image", False) and image is None:
                raise ValueError("No image has arrived on the image topic")
            acquired = slots.acquire(blocking=False)
            if not acquired:
                raise RuntimeError("Request capacity reached")
            workers.submit(work, job, image)
        except Exception as error:
            if acquired:
                slots.release()
            publish(identity, done=True, error=str(error))

    def perform(job, image):
        operation = job.get("operation", "chat")
        body = deepcopy(job["body"])
        headers = {"Content-Type": "application/json"}
        suffix = {"chat": "/v1/chat/completions", "speech": "/v1/audio/speech",
                  "transcribe": "/v1/audio/transcriptions",
                  "translate": "/v1/audio/translations"}[operation]
        image_header = None
        if image is not None:
            if operation != "chat":
                raise ValueError("use_image is supported for chat requests only")
            pixels = image_array(image)
            if image.encoding == "rgb8":
                pixels = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
            ok, encoded = cv2.imencode(".jpg", pixels)
            if not ok:
                raise ValueError("Could not encode the request image")
            content = body["messages"][-1]["content"]
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]
            body["messages"][-1]["content"] = [*content, {"type": "image_url", "image_url": {
                "url": "data:image/jpeg;base64," + base64.b64encode(encoded).decode()}}]
            image_header = header_dict(image.header)
        if operation == "chat":
            body.setdefault("max_tokens", app.get("request", {}).get("max_tokens", 512))
        if operation in ("transcribe", "translate"):
            audio = base64.b64decode(job["audio_base64"], validate=True)
            boundary = "neat" + uuid.uuid4().hex
            parts = []
            for key, value in body.items():
                if not key.replace("_", "").isalnum():
                    raise ValueError("Invalid audio request field name")
                parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'.encode())
            parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="audio.wav"\r\nContent-Type: audio/wav\r\n\r\n'.encode())
            payload = b"".join(parts) + audio + f"\r\n--{boundary}--\r\n".encode()
            headers["Content-Type"] = f"multipart/form-data; boundary={boundary}"
        else:
            payload = json.dumps(body, allow_nan=False).encode()
        req = request.Request(endpoint + suffix, data=payload, headers=headers, method="POST")
        # Resolve trust per request so the backend may create its certificate
        # after the ROS node starts. All four operations use this verified context.
        context = backend_ssl_context(config_path, settings, endpoint)
        with request.urlopen(req, timeout=timeout, context=context) as response:
            if operation == "chat" and body.get("stream", False):
                for raw in response:
                    if stop.is_set():
                        return
                    line = raw.decode().strip()
                    if not line.startswith("data:"):
                        continue
                    chunk = line[5:].strip()
                    if chunk == "[DONE]":
                        break
                    publish(job["id"], done=False, header=image_header, data=json.loads(chunk))
                publish(job["id"], done=True, header=image_header)
            elif operation == "speech":
                publish(job["id"], done=True, content_type=response.headers.get_content_type(),
                        audio_base64=base64.b64encode(response.read()).decode(),
                        metadata={name: response.headers[name] for name in (
                            "X-RTF", "X-Audio-Duration", "X-Elapsed-Time", "X-Engine",
                            "X-Voice", "X-Language", "X-Speed") if name in response.headers})
            elif operation in ("transcribe", "translate") and body.get("response_format") == "text":
                publish(job["id"], done=True, data=response.read().decode("utf-8"))
            else:
                publish(job["id"], done=True, header=image_header, data=json.load(response))

    def work(job, image):
        try:
            if not stop.is_set():
                perform(job, image)
        except Exception as error:
            publish(job["id"], done=True, error=str(error))
        finally:
            slots.release()

    node.create_subscription(Image, settings.get("image_topic", "image"), receive_image, 1)
    node.create_subscription(String, "requests", receive, 8)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        stop.set()
        workers.shutdown(wait=True, cancel_futures=True)
        node.destroy_node()
        rclpy.try_shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
