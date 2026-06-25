"""Bounded background VLM requests for a configured detector class."""

from __future__ import annotations

from queue import Empty, Full, Queue
import threading
import time
from urllib import error, request

import numpy as np

from .helpers import (
    Config,
    crop_box,
    label_for_box,
    request_object_color_sentence,
)


class ObjectVLMCommenter:
    def __init__(self, cfg: Config, labels: list[str]):
        self.cfg = cfg
        self.labels = labels
        self.queue: Queue[tuple[np.ndarray, str]] = Queue(
            maxsize=cfg.vlm_max_pending_requests
        )
        self.stop_event = threading.Event()
        self.worker = threading.Thread(target=self._run, daemon=True)
        self.lock = threading.Lock()
        self.last_enqueue_at = 0.0
        self.in_flight = False
        self.started = False

    def start(self) -> None:
        if self.cfg.vlm_enabled and not self.started:
            self.worker.start()
            self.started = True

    def try_enqueue(self, frame: np.ndarray, boxes: list[dict]) -> None:
        if not self.cfg.vlm_enabled:
            return
        now = time.monotonic()
        if now - self.last_enqueue_at < self.cfg.vlm_interval_seconds:
            return
        if self._pending_count() >= self.cfg.vlm_max_pending_requests:
            print("vlm: queue busy, dropping request", flush=True)
            self.last_enqueue_at = now
            return

        selected_boxes = [
            box
            for box in boxes
            if label_for_box(box, self.labels).lower() in set(self.cfg.trigger_classes)
        ]
        if not selected_boxes:
            return

        box = max(selected_boxes, key=lambda item: item["score"])
        detected_class = label_for_box(box, self.labels).lower()
        try:
            crop = crop_box(frame, box).copy()
            self.queue.put_nowait((crop, detected_class))
            self.last_enqueue_at = now
        except Full:
            print("vlm: queue full, dropping request", flush=True)
            self.last_enqueue_at = now

    def close(self) -> None:
        self.stop_event.set()
        if self.started:
            self.worker.join(timeout=1.0)

    def _pending_count(self) -> int:
        with self.lock:
            return self.queue.qsize() + int(self.in_flight)

    def _set_in_flight(self, value: bool) -> None:
        with self.lock:
            self.in_flight = value

    def _run(self) -> None:
        while not self.stop_event.is_set():
            try:
                crop, detected_class = self.queue.get(timeout=0.2)
            except Empty:
                continue

            self._set_in_flight(True)
            try:
                if self._server_ready():
                    response = request_object_color_sentence(crop, detected_class, self.cfg)
                    if response:
                        print(f"vlm: {response}", flush=True)
            except (TimeoutError, OSError, error.URLError) as exc:
                print(f"vlm unavailable: {exc}", flush=True)
            except Exception as exc:
                print(f"vlm request failed: {exc}", flush=True)
            finally:
                self._set_in_flight(False)
                self.queue.task_done()

    def _server_ready(self) -> bool:
        url = f"http://{self.cfg.vlm_host}:{self.cfg.vlm_port}/v1/models"
        try:
            timeout = min(self.cfg.vlm_timeout_seconds, 5.0)
            with request.urlopen(url, timeout=timeout) as res:
                return 200 <= res.status < 300
        except (TimeoutError, OSError, error.URLError) as exc:
            print(
                f"vlm unavailable: could not access server at "
                f"{self.cfg.vlm_host}:{self.cfg.vlm_port}: {exc}",
                flush=True,
            )
            return False
