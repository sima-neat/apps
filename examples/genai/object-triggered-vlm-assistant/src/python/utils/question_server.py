"""Small HTTP question API for current-frame and recent-history VLM answers."""

from __future__ import annotations

import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .helpers import (
    Config,
    crop_box,
    preview_image_data_uri,
    request_vlm_answer,
)
from .memory import (
    FrameRingBuffer,
    ObjectTrackMemory,
    decode_jpeg_rgb,
)


def parse_seconds_ago(question: str) -> float | None:
    text = question.lower()
    match = re.search(
        r"(\d+(?:\.\d+)?)\s*(seconds?|secs?|s|minutes?|mins?|m)\s+ago",
        text,
    )
    if match is None:
        return None
    value = float(match.group(1))
    unit = match.group(2)
    if unit.startswith("m") and unit not in {"ms"}:
        value *= 60.0
    return value


def is_past_question(question: str, explicit_seconds_ago: float | None) -> bool:
    if explicit_seconds_ago is not None:
        return explicit_seconds_ago > 0
    text = question.lower()
    return any(word in text for word in ("ago", "earlier", "previous", "before"))


def infer_class_from_question(question: str, classes: list[str]) -> str | None:
    text = question.lower()
    for class_name in sorted(classes, key=len, reverse=True):
        escaped = re.escape(class_name.lower())
        plural = re.escape(plural_class_name(class_name.lower()))
        if re.search(rf"\b({escaped}|{plural})\b", text):
            return class_name.lower()
    return None


PERSON_CLASSES = {"person"}
VEHICLE_CLASSES = {"bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat"}
ANIMAL_CLASSES = {"bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"}
SIGNAL_CLASSES = {"traffic light", "stop sign", "parking meter"}
FOOD_CLASSES = {"banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake"}
FURNITURE_CLASSES = {"chair", "couch", "potted plant", "bed", "dining table", "toilet"}
ELECTRONIC_CLASSES = {"tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "refrigerator"}


def plural_class_name(class_name: str) -> str:
    irregular = {
        "person": "people",
        "mouse": "mice",
        "sheep": "sheep",
    }
    if class_name in irregular:
        return irregular[class_name]
    if class_name.endswith("s"):
        return class_name
    return f"{class_name}s"


def prompt_templates_for_class(class_name: str) -> list[str]:
    c = class_name.lower()
    plural = plural_class_name(c)
    if c in PERSON_CLASSES:
        return [
            "What is the person doing?",
            "What color is the person's clothing?",
            "What is the person wearing?",
            "Is the person carrying anything?",
            "Where is the person in the scene?",
            "Which direction is the person facing?",
            "Is the person's face visible?",
            "Is the person standing or sitting?",
            "How many people are visible?",
            "Describe the person briefly.",
        ]
    if c in VEHICLE_CLASSES:
        return [
            f"What color is the {c}?",
            f"Is there readable text on the {c}?",
            f"Is there a logo visible on the {c}?",
            f"Where is the {c} in the scene?",
            f"Which direction is the {c} facing?",
            f"Does the {c} appear moving or parked?",
            f"Is the {c} damaged?",
            f"What part of the {c} is visible?",
            f"Is anything attached to the {c}?",
            f"Describe the {c} briefly.",
        ]
    if c in ANIMAL_CLASSES:
        return [
            f"What color is the {c}?",
            f"What is the {c} doing?",
            f"Where is the {c} in the scene?",
            f"Which direction is the {c} facing?",
            f"Is the {c} standing, sitting, or moving?",
            f"Is the {c} alone or near something?",
            f"Is the {c} fully visible?",
            f"What size does the {c} appear to be?",
            f"How many {plural} are visible?",
            f"Describe the {c} briefly.",
        ]
    if c in SIGNAL_CLASSES:
        return [
            f"What color is the {c}?",
            f"Is the {c} readable or clear?",
            f"Where is the {c} in the scene?",
            f"What state is the {c} showing?",
            f"Is the {c} partially blocked?",
            f"Is there text visible on the {c}?",
            f"Is the {c} close or far away?",
            f"How many {plural} are visible?",
            f"What is near the {c}?",
            f"Describe the {c} briefly.",
        ]
    if c in FOOD_CLASSES:
        return [
            f"What color is the {c}?",
            f"Where is the {c} in the scene?",
            f"Is the {c} whole or cut?",
            f"Is the {c} on a plate or surface?",
            f"Is the {c} packaged?",
            f"Is there readable text near the {c}?",
            f"How many {plural} are visible?",
            f"What is next to the {c}?",
            f"Is the {c} clearly visible?",
            f"Describe the {c} briefly.",
        ]
    if c in FURNITURE_CLASSES:
        return [
            f"What color is the {c}?",
            f"Where is the {c} in the scene?",
            f"Is the {c} occupied or empty?",
            f"What material does the {c} appear to be?",
            f"Is the {c} fully visible?",
            f"Is anything on the {c}?",
            f"Is the {c} damaged or unusual?",
            f"How many {plural} are visible?",
            f"What is near the {c}?",
            f"Describe the {c} briefly.",
        ]
    if c in ELECTRONIC_CLASSES:
        return [
            f"What color is the {c}?",
            f"Where is the {c} in the scene?",
            f"Is the {c} on or off?",
            f"Is there readable text on the {c}?",
            f"Is a logo visible on the {c}?",
            f"Is the {c} being held or used?",
            f"Is the {c} fully visible?",
            f"What is near the {c}?",
            f"How many {plural} are visible?",
            f"Describe the {c} briefly.",
        ]
    return [
        f"What color is the {c}?",
        f"Where is the {c} in the scene?",
        f"Is the {c} clearly visible?",
        f"Is there readable text on or near the {c}?",
        f"Is a logo visible on the {c}?",
        f"What is next to the {c}?",
        f"Is the {c} being held or used?",
        f"How many {plural} are visible?",
        f"What condition is the {c} in?",
        f"Describe the {c} briefly.",
    ]


CHAT_HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Object-Triggered VLM Chat</title>
  <style>
    :root { color-scheme: dark; font-family: system-ui, -apple-system, sans-serif; }
    body { margin: 0; background: #111827; color: #e5e7eb; }
    main { max-width: 920px; margin: 0 auto; padding: 24px; }
    h1 { font-size: 22px; margin: 0 0 4px; }
    .hint { color: #9ca3af; margin-bottom: 18px; }
    .toolbar { display: flex; align-items: center; gap: 10px; margin: 14px 0 10px; color: #d1d5db; flex-wrap: wrap; }
    select { border: 1px solid #374151; border-radius: 10px; padding: 8px 10px; background: #030712; color: #e5e7eb; }
    .class-panel { border: 1px solid #374151; border-radius: 12px; padding: 10px; background: #030712; margin-bottom: 10px; }
    .class-title { color: #d1d5db; font-size: 13px; margin-bottom: 8px; }
    .class-toggles { display: flex; flex-wrap: wrap; gap: 8px; max-height: 150px; overflow-y: auto; }
    .class-toggle { border: 1px solid #374151; border-radius: 999px; padding: 6px 10px; color: #d1d5db; cursor: pointer; font-size: 13px; user-select: none; }
    .class-toggle.active { background: #22c55e; border-color: #22c55e; color: #052e16; font-weight: 700; }
    .status { color: #9ca3af; font-size: 12px; min-height: 16px; margin: 6px 0 10px; }
    #chat { display: flex; flex-direction: column; gap: 12px; min-height: 260px; }
    .msg { border-radius: 14px; padding: 12px 14px; line-height: 1.4; white-space: pre-wrap; }
    .user { align-self: flex-end; background: #2563eb; max-width: 78%; }
    .assistant { align-self: flex-start; background: #1f2937; max-width: 78%; }
    .meta { color: #9ca3af; font-size: 12px; margin-top: 8px; white-space: normal; }
    .evidence-img { display: block; margin-top: 10px; max-width: min(480px, 100%); border: 1px solid #374151; border-radius: 10px; }
    form { display: flex; gap: 10px; margin-top: 18px; position: sticky; bottom: 0; background: #111827; padding-top: 12px; }
    input { flex: 1; border: 1px solid #374151; border-radius: 12px; padding: 12px; background: #030712; color: #e5e7eb; font-size: 15px; }
    button { border: 0; border-radius: 12px; padding: 0 18px; background: #22c55e; color: #052e16; font-weight: 700; cursor: pointer; }
    button:disabled { opacity: .6; cursor: wait; }
    .examples { display: flex; flex-wrap: wrap; gap: 8px; margin: 14px 0 18px; }
    .chip { border: 1px solid #374151; border-radius: 999px; padding: 7px 10px; color: #d1d5db; cursor: pointer; font-size: 13px; }
  </style>
</head>
<body>
  <main>
    <h1>Object-Triggered VLM Chat</h1>
    <div class="hint">Ask about the current frame or recent {{TRIGGER_CLASSES_TEXT}} history.</div>
    <div class="class-panel">
      <div class="class-title">Tracked classes: click to select or unselect. At least one class must stay selected.</div>
      <div class="class-toggles" id="classToggles"></div>
    </div>
    <div class="status" id="classStatus"></div>
    <div class="toolbar">
      <label for="timeSelect">Time:</label>
      <select id="timeSelect">
        <option value="current">Current frame</option>
        <option value="1">1 second ago</option>
        <option value="3" selected>3 seconds ago</option>
        <option value="5">5 seconds ago</option>
        <option value="10">10 seconds ago</option>
        <option value="20">20 seconds ago</option>
        <option value="30">30 seconds ago</option>
        <option value="60">60 seconds ago</option>
        <option value="120">120 seconds ago</option>
      </select>
    </div>
    <div class="examples" id="promptChips"></div>
    <section id="chat"></section>
    <form id="form">
      <input id="question" autocomplete="off" placeholder="Ask a question...">
      <button id="send" type="submit">Send</button>
    </form>
  </main>
  <script>
    const chat = document.getElementById('chat');
    const form = document.getElementById('form');
    const input = document.getElementById('question');
    const send = document.getElementById('send');
    const timeSelect = document.getElementById('timeSelect');
    const classToggles = document.getElementById('classToggles');
    const classStatus = document.getElementById('classStatus');
    const promptChips = document.getElementById('promptChips');
    const classOptions = {{CLASS_OPTIONS_JSON}};
    let triggerClasses = {{TRIGGER_CLASSES_JSON}};
    let promptTemplatesByClass = {{PROMPT_TEMPLATES_BY_CLASS_JSON}};

    function arraysEqual(first, second) {
      return first.length === second.length && first.every((value, index) => value === second[index]);
    }

    function renderClassToggles() {
      classToggles.innerHTML = '';
      classOptions.forEach((name) => {
        const toggle = document.createElement('button');
        toggle.type = 'button';
        toggle.className = `class-toggle${triggerClasses.includes(name) ? ' active' : ''}`;
        toggle.textContent = name;
        toggle.addEventListener('click', () => {
          const nextClasses = triggerClasses.includes(name)
            ? triggerClasses.filter((className) => className !== name)
            : [...triggerClasses, name];
          updateTriggerClasses(nextClasses);
        });
        classToggles.appendChild(toggle);
      });
    }

    function setClassStatus(text) {
      classStatus.textContent = text || '';
    }

    function questionWithSelectedTime(template) {
      const seconds = timeSelect.value;
      if (seconds === 'current' || /\bnow\b|current frame/i.test(template)) {
        return template;
      }
      return `${template} ${seconds} seconds ago`;
    }

    function renderPromptChips() {
      promptChips.innerHTML = '';
      triggerClasses.forEach((className) => {
        const templates = promptTemplatesByClass[className] || [];
        templates.forEach((template) => {
          const chip = document.createElement('span');
          chip.className = 'chip';
          chip.textContent = questionWithSelectedTime(template);
          chip.addEventListener('click', () => ask(questionWithSelectedTime(template)));
          promptChips.appendChild(chip);
        });
      });
    }

    function addMessage(text, cls, meta, image) {
      const el = document.createElement('div');
      el.className = `msg ${cls}`;
      const body = document.createElement('div');
      body.textContent = text;
      el.appendChild(body);
      if (image) {
        const img = document.createElement('img');
        img.className = 'evidence-img';
        img.src = image;
        img.alt = 'VLM evidence image';
        el.appendChild(img);
      }
      if (meta) {
        const detail = document.createElement('div');
        detail.className = 'meta';
        detail.textContent = meta;
        el.appendChild(detail);
      }
      chat.appendChild(el);
    }

    async function ask(question) {
      chat.innerHTML = '';
      addMessage(question, 'user');
      send.disabled = true;
      input.disabled = true;
      try {
        const res = await fetch('/ask', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ question })
        });
        const data = await res.json();
        const meta = data.evidence ? `${data.mode || 'answer'} evidence: ${JSON.stringify(data.evidence)}` : (data.error || '');
        addMessage(data.answer || data.error || JSON.stringify(data), 'assistant', meta, data.evidence_image);
      } catch (err) {
        addMessage(`Request failed: ${err}`, 'assistant');
      } finally {
        send.disabled = false;
        input.disabled = false;
        input.focus();
      }
    }

    form.addEventListener('submit', (event) => {
      event.preventDefault();
      const question = input.value.trim();
      if (!question) return;
      input.value = '';
      ask(question);
    });

    async function updateTriggerClasses(nextClasses) {
      if (nextClasses.length === 0) {
        setClassStatus('Select at least one class.');
        renderClassToggles();
        return;
      }
      if (arraysEqual(nextClasses, triggerClasses)) return;
      send.disabled = true;
      input.disabled = true;
      classToggles.style.pointerEvents = 'none';
      try {
        const res = await fetch('/trigger-class', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ classes: nextClasses })
        });
        const data = await res.json();
        if (!res.ok || data.error) {
          throw new Error(data.error || `HTTP ${res.status}`);
        }
        triggerClasses = data.trigger_classes;
        promptTemplatesByClass = data.prompt_templates_by_class;
        renderClassToggles();
        renderPromptChips();
        setClassStatus('');
      } catch (err) {
        setClassStatus(`Failed to change classes: ${err}`);
        renderClassToggles();
      } finally {
        send.disabled = false;
        input.disabled = false;
        classToggles.style.pointerEvents = '';
        input.focus();
      }
    }

    timeSelect.addEventListener('change', renderPromptChips);
    renderClassToggles();
    renderPromptChips();
  </script>
</body>
</html>
"""

class QuestionServer:
    def __init__(
        self,
        cfg: Config,
        frame_memory: FrameRingBuffer,
        object_memory: ObjectTrackMemory,
        labels: list[str],
    ):
        self.cfg = cfg
        self.frame_memory = frame_memory
        self.object_memory = object_memory
        self.labels = labels
        self.class_options = list(dict.fromkeys([*labels, cfg.trigger_class]))
        self.server: ThreadingHTTPServer | None = None
        self.thread: threading.Thread | None = None

    def chat_html(self) -> str:
        return (
            CHAT_HTML.replace("{{TRIGGER_CLASSES_TEXT}}", ", ".join(self.cfg.trigger_classes))
            .replace("{{TRIGGER_CLASSES_JSON}}", json.dumps(self.cfg.trigger_classes))
            .replace(
                "{{PROMPT_TEMPLATES_BY_CLASS_JSON}}",
                json.dumps({
                    class_name: prompt_templates_for_class(class_name)
                    for class_name in self.cfg.trigger_classes
                }),
            )
            .replace("{{CLASS_OPTIONS_JSON}}", json.dumps(self.class_options))
        )

    def start(self) -> None:
        if not self.cfg.qa_enabled:
            return

        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):
                return

            def do_GET(self):
                if self.path in {"/", "/chat"}:
                    self._write_html(outer.chat_html())
                    return
                if self.path == "/health":
                    self._write_json({"status": "ok"})
                    return
                if self.path == "/memory/latest":
                    obs = outer.object_memory.latest()
                    self._write_json(
                        {
                            "latest": None
                            if obs is None
                            else {
                                "track_id": obs.track_id,
                                "frame_id": obs.frame_id,
                                "timestamp_ms": obs.timestamp_ms,
                                "class": obs.class_name,
                                "bbox": obs.bbox,
                                "score": obs.score,
                            }
                        }
                    )
                    return
                self.send_error(404, "not found")

            def do_POST(self):
                if self.path not in {"/ask", "/trigger-class"}:
                    self.send_error(404, "not found")
                    return

                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    body = self.rfile.read(length)
                    payload = json.loads(body.decode("utf-8") or "{}")
                    if self.path == "/trigger-class":
                        response, status = outer.update_trigger_class(payload)
                        self._write_json(response, status=status)
                    else:
                        response = outer.answer(payload)
                        self._write_json(response)
                except Exception as exc:
                    self._write_json({"error": str(exc)}, status=500)

            def _write_json(self, value, status=200):
                encoded = json.dumps(value).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

            def _write_html(self, value, status=200):
                encoded = value.encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

        self.server = ThreadingHTTPServer((self.cfg.qa_host, self.cfg.qa_port), Handler)
        self.thread = threading.Thread(
            target=self.server.serve_forever,
            name="question-server",
            daemon=True,
        )
        self.thread.start()
        print(
            f"qa: chat UI http://{self.cfg.qa_host}:{self.cfg.qa_port}/  api /ask",
            flush=True,
        )

    def close(self) -> None:
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
        if self.thread is not None:
            self.thread.join(timeout=1.0)

    def update_trigger_class(self, payload: dict) -> tuple[dict, int]:
        raw_classes = payload.get("classes")
        if raw_classes is None:
            raw_classes = [payload.get("class", "")]
        if isinstance(raw_classes, str):
            raw_classes = [raw_classes]
        requested = list(
            dict.fromkeys(
                str(class_name).strip().lower()
                for class_name in raw_classes
                if str(class_name).strip()
            )
        )
        if not requested:
            return {"error": "at least one class is required"}, 400
        available = {label.lower() for label in self.class_options}
        unknown = [class_name for class_name in requested if class_name not in available]
        if unknown:
            return {"error": f"unknown class: {', '.join(unknown)}"}, 400
        self.cfg.trigger_classes = requested
        self.cfg.trigger_class = requested[0]
        self.object_memory.set_trigger_classes(requested)
        return {
            "trigger_class": self.cfg.trigger_class,
            "trigger_classes": self.cfg.trigger_classes,
            "prompt_templates_by_class": {
                class_name: prompt_templates_for_class(class_name)
                for class_name in self.cfg.trigger_classes
            },
            "memory_reset": False,
            "memory_retained": True,
        }, 200

    def answer(self, payload: dict) -> dict:
        question = str(payload.get("question", "")).strip()
        if not question:
            return {"error": "question is required"}

        explicit_seconds_ago = payload.get("seconds_ago")
        if explicit_seconds_ago is not None:
            explicit_seconds_ago = float(explicit_seconds_ago)
        else:
            explicit_seconds_ago = parse_seconds_ago(question)

        if is_past_question(question, explicit_seconds_ago):
            return self._answer_past(question, explicit_seconds_ago, payload)
        return self._answer_current(question)

    def _answer_current(self, question: str) -> dict:
        latest = self.frame_memory.latest()
        if latest is None:
            return {"error": "no current frame available"}
        frame_id, timestamp_ms, frame_rgb = latest
        answer = request_vlm_answer(
            frame_rgb,
            self.cfg,
            question,
            system_prompt=(
                "Answer the user's question using only the current image. "
                "Do not rely on detector metadata. Be precise and brief. "
                "If the answer is not visible, say not visible."
            ),
            max_tokens=self.cfg.qa_max_tokens,
        )
        return {
            "mode": "current",
            "answer": answer,
            "evidence_image": preview_image_data_uri(frame_rgb),
            "evidence": {
                "frame_id": frame_id,
                "timestamp_ms": timestamp_ms,
            },
        }

    def _answer_past_frame(
        self,
        question: str,
        seconds_ago: float,
        target_timestamp_ms: int,
        latest_timestamp_ms: int,
    ) -> dict:
        frame = self.frame_memory.nearest(target_timestamp_ms)
        if frame is None:
            return {
                "mode": "past",
                "answer": "not visible",
                "error": "no retained frame history is available yet",
            }

        offset_seconds = abs(frame.timestamp_ms - target_timestamp_ms) / 1000.0
        if offset_seconds > self.cfg.qa_past_tolerance_seconds:
            return {
                "mode": "past",
                "answer": "not visible",
                "error": (
                    f"no retained frame within {self.cfg.qa_past_tolerance_seconds:.1f} "
                    "seconds of the requested time"
                ),
                "evidence": {
                    "nearest_frame_id": frame.frame_id,
                    "nearest_timestamp_ms": frame.timestamp_ms,
                    "requested_seconds_ago": round(float(seconds_ago), 3),
                    "offset_from_requested_seconds": round(offset_seconds, 3),
                },
            }

        frame_rgb = decode_jpeg_rgb(frame.jpeg)
        answer = request_vlm_answer(
            frame_rgb,
            self.cfg,
            question,
            system_prompt=(
                "Answer about a past video moment using only the provided retained "
                "video frame. Be precise and brief. If the requested detail is not "
                "visible in the frame, say not visible."
            ),
            metadata_text=(
                f"frame_id={frame.frame_id}; "
                f"observation_age_seconds={(latest_timestamp_ms - frame.timestamp_ms) / 1000.0:.2f}; "
                "source=nearest_retained_frame"
            ),
            max_tokens=self.cfg.qa_max_tokens,
        )
        return {
            "mode": "past",
            "answer": answer,
            "evidence_image": preview_image_data_uri(frame_rgb),
            "evidence": {
                "frame_id": frame.frame_id,
                "timestamp_ms": frame.timestamp_ms,
                "requested_class": None,
                "full_frame": True,
                "requested_seconds_ago": round(float(seconds_ago), 3),
                "offset_from_requested_seconds": round(offset_seconds, 3),
                "age_seconds": round(
                    (int(time.time() * 1000) - frame.timestamp_ms) / 1000.0,
                    3,
                ),
            },
        }

    def _answer_past(
        self,
        question: str,
        seconds_ago: float | None,
        payload: dict,
    ) -> dict:
        latest = self.frame_memory.latest()
        if latest is None:
            return {"error": "no current frame available"}
        _, latest_timestamp_ms, _ = latest
        if seconds_ago is None:
            seconds_ago = self.cfg.qa_default_past_seconds
        target_timestamp_ms = latest_timestamp_ms - int(float(seconds_ago) * 1000)
        track_id = payload.get("track_id")
        track_id = int(track_id) if track_id is not None else None
        requested_class = payload.get("class")
        requested_class = (
            str(requested_class).strip().lower()
            if requested_class
            else infer_class_from_question(question, self.cfg.trigger_classes)
        )

        if requested_class is None and track_id is None:
            return self._answer_past_frame(
                question,
                seconds_ago=float(seconds_ago),
                target_timestamp_ms=target_timestamp_ms,
                latest_timestamp_ms=latest_timestamp_ms,
            )

        observation = self.object_memory.find_near(
            target_timestamp_ms,
            tolerance_seconds=self.cfg.qa_past_tolerance_seconds,
            track_id=track_id,
            class_name=requested_class,
        )
        fallback_used = False
        if observation is None:
            observation = self.object_memory.nearest_retained(
                target_timestamp_ms,
                now_ms=latest_timestamp_ms,
                track_id=track_id,
                class_name=requested_class,
            )
            fallback_used = observation is not None
        if observation is None:
            return {
                "mode": "past",
                "answer": "not visible",
                "error": (
                    f"no {requested_class or ', '.join(self.cfg.trigger_classes)} observation retained in the last "
                    f"{self.cfg.memory_retention_seconds:.0f} seconds"
                ),
            }

        offset_seconds = abs(observation.timestamp_ms - target_timestamp_ms) / 1000.0
        if fallback_used and offset_seconds > self.cfg.qa_past_tolerance_seconds:
            return {
                "mode": "past",
                "answer": "not visible",
                "error": (
                    f"no {requested_class or ', '.join(self.cfg.trigger_classes)} observation within "
                    f"{self.cfg.qa_past_tolerance_seconds:.1f} seconds of the requested time"
                ),
                "evidence": {
                    "nearest_track_id": observation.track_id,
                    "nearest_frame_id": observation.frame_id,
                    "nearest_timestamp_ms": observation.timestamp_ms,
                    "nearest_class": observation.class_name,
                    "requested_seconds_ago": round(float(seconds_ago), 3),
                    "offset_from_requested_seconds": round(offset_seconds, 3),
                    "fallback_nearest_retained": True,
                },
            }

        frame = self.frame_memory.get(observation.frame_id)
        if frame is None:
            frame = self.frame_memory.nearest(observation.timestamp_ms)
        if frame is None:
            return {
                "mode": "past",
                "answer": "not visible",
                "error": "linked frame was evicted from memory",
            }

        frame_rgb = decode_jpeg_rgb(frame.jpeg)
        evidence_rgb = crop_box(
            frame_rgb,
            {"bbox": observation.bbox},
            padding_ratio=self.cfg.qa_crop_padding_ratio,
        )
        metadata = (
            f"detector_class={observation.class_name}; "
            f"track_id={observation.track_id}; "
            f"frame_id={observation.frame_id}; "
            f"bbox={observation.bbox}; "
            f"score={observation.score:.3f}; "
            f"observation_age_seconds={(latest_timestamp_ms - observation.timestamp_ms) / 1000.0:.2f}; "
            f"fallback_nearest_retained={str(fallback_used).lower()}; "
            f"requested_class={requested_class or 'any'}"
        )
        answer = request_vlm_answer(
            evidence_rgb,
            self.cfg,
            question,
            system_prompt=(
                "Answer about a past video moment using only the provided image crop "
                "and detector metadata. The detector metadata is the source of truth "
                "only for class, time, and location. Use the image crop for visual "
                "details such as color, readable text, brand markings, logos, and "
                "writing. If readable text is visible, transcribe it exactly. If "
                "only part of the text is readable, return the visible characters and "
                "say partial. Do not invent hidden, cropped, blurry, or illegible "
                "characters. If no text is readable, say no readable text visible. "
                "Be precise and brief."
            ),
            metadata_text=metadata,
            max_tokens=self.cfg.qa_max_tokens,
        )
        return {
            "mode": "past",
            "answer": answer,
            "evidence_image": preview_image_data_uri(evidence_rgb),
            "evidence": {
                "track_id": observation.track_id,
                "frame_id": observation.frame_id,
                "timestamp_ms": observation.timestamp_ms,
                "class": observation.class_name,
                "requested_class": requested_class,
                "bbox": observation.bbox,
                "score": observation.score,
                "crop_padding_ratio": self.cfg.qa_crop_padding_ratio,
                "fallback_nearest_retained": fallback_used,
                "requested_seconds_ago": round(float(seconds_ago), 3),
                "offset_from_requested_seconds": round(
                    offset_seconds,
                    3,
                ),
                "age_seconds": round(
                    (int(time.time() * 1000) - observation.timestamp_ms) / 1000.0,
                    3,
                ),
            },
        }
