"""Real ROS audio request transport with a controlled HTTP backend.

These checks cover routing, multipart bytes and response formats, not speech
model accuracy or performance. No model downloads or live service are needed.
"""
import base64
from email.parser import BytesParser
from email.policy import default
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

import pytest
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import yaml

pytestmark = pytest.mark.ros2
ROOT = Path(__file__).resolve().parents[2]


def test_audio_routes_payloads_and_response_formats(tmp_path):
    audio = b"RIFF\x00\xff\x80\nWAVE transport fixture"
    seen = []
    held_requests = []
    held_lock = threading.Lock()
    both_started = threading.Event()
    release = threading.Event()

    class Backend(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            payload = self.rfile.read(int(self.headers["Content-Length"]))
            if self.path == "/v1/audio/speech":
                body = json.loads(payload)
                seen.append((self.path, body, None))
                if body.get("input") == "hold":
                    with held_lock:
                        held_requests.append(body)
                        if len(held_requests) == 2:
                            both_started.set()
                    release.wait(timeout=10)
                content_type, result = "audio/wav", audio
            else:
                message = BytesParser(policy=default).parsebytes(
                    f'Content-Type: {self.headers["Content-Type"]}\r\n\r\n'.encode() + payload)
                fields = {}
                file_bytes = None
                for part in message.iter_parts():
                    name = part.get_param("name", header="content-disposition")
                    if name == "file":
                        file_bytes = part.get_payload(decode=True)
                    else:
                        fields[name] = part.get_payload(decode=True).decode()
                seen.append((self.path, fields, file_bytes))
                if fields["response_format"] == "text":
                    content_type, result = "text/plain", "Hello, café.\n".encode()
                else:
                    content_type, result = "application/json", json.dumps({"text": "Hello, café."}).encode()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(result)))
            if self.path == "/v1/audio/speech":
                self.send_header("X-RTF", "0.25")
                self.send_header("X-Engine", "controlled-test")
            self.end_headers()
            self.wfile.write(result)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Backend)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"app": {}, "ros": {
        "studio_url": f"http://127.0.0.1:{server.server_port}",
        "request_timeout_seconds": 5, "pending_requests": 2}}))
    rclpy.init()
    observer = Node("studio_transport_test")
    responses = []
    observer.create_subscription(String, "studio_test_responses",
                                 lambda msg: responses.append(json.loads(msg.data)), 16)
    publisher = observer.create_publisher(String, "studio_test_requests", 8)
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "support/ros2") + os.pathsep + environment.get("PYTHONPATH", "")
    entry = ROOT / "examples/ros2/genai/neat-genai-studio/src/python/ros_node.py"
    process = None

    def wait_until(condition, timeout=15):
        deadline = time.monotonic() + timeout
        while not condition():
            assert process.poll() is None, "Studio node exited"
            assert time.monotonic() < deadline, "Studio ROS response timed out"
            rclpy.spin_once(observer, timeout_sec=0.01)

    def send(job):
        publisher.publish(String(data=json.dumps(job)))
        wait_until(lambda: any(x["id"] == job["id"] for x in responses))
        return next(x for x in responses if x["id"] == job["id"])

    try:
        with (tmp_path / "studio.log").open("w") as log:
            process = subprocess.Popen([sys.executable, str(entry), "--config", str(config),
                "--ros-args", "-r", "requests:=studio_test_requests", "-r",
                "responses:=studio_test_responses"], env=environment, stdout=log, stderr=log)
            wait_until(lambda: publisher.get_subscription_count() == 1 and
                       observer.count_publishers("studio_test_responses") == 1)
            for operation, route in (("transcribe", "transcriptions"), ("translate", "translations")):
                for response_format in ("json", "verbose_json", "text"):
                    body = {"response_format": response_format, "temperature": 0}
                    reply = send({"id": f"{operation}-{response_format}", "operation": operation,
                                  "body": body, "audio_base64": base64.b64encode(audio).decode()})
                    assert "error" not in reply, reply
                    assert reply["done"] is True
                    assert reply["data"] == ("Hello, café.\n" if response_format == "text" else {"text": "Hello, café."})
                    assert seen[-1] == (f"/v1/audio/{route}",
                                        {"response_format": response_format, "temperature": "0"}, audio)
            reply = send({"id": "bad-audio", "operation": "translate", "body": {}, "audio_base64": "!"})
            assert reply["error"]
            body = {"input": "Hello, café.", "response_format": "wav"}
            reply = send({"id": "speech-after-error", "operation": "speech", "body": body})
            assert "error" not in reply, reply
            assert base64.b64decode(reply["audio_base64"]) == audio
            assert reply["content_type"] == "audio/wav"
            assert reply["metadata"] == {"X-RTF": "0.25", "X-Engine": "controlled-test"}
            assert seen[-1] == ("/v1/audio/speech", body, None)

            # Both admitted requests must reach the backend before either
            # completes; excess work must be refused without reaching it.
            for identity in ("held-one", "held-two"):
                publisher.publish(String(data=json.dumps({"id": identity,
                    "operation": "speech", "body": {"input": "hold"}})))
            wait_until(both_started.is_set, timeout=3)
            reply = send({"id": "over-capacity", "operation": "speech", "body": body})
            assert reply["error"] == "Request capacity reached"
            release.set()
            wait_until(lambda: all(any(x["id"] == identity for x in responses)
                                   for identity in ("held-one", "held-two")))
            for reply in responses:
                if reply["id"] in ("held-one", "held-two"):
                    assert "error" not in reply, reply
                    assert base64.b64decode(reply["audio_base64"]) == audio
    finally:
        release.set()
        if process is not None:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=10)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
        observer.destroy_node()
        rclpy.try_shutdown()
        server.shutdown()
        thread.join()
        server.server_close()
