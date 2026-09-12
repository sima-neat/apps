"""SupertonicTTS — client to the isolated Supertonic 3 (MLA) synthesis worker.

Supertonic 3 on Modalix (https://github.com/florianvoss-commit/supertonic-sima)
runs the eight-step vector field and the vocoder on the MLA through PyNeat and
the text front end on the CPU through ONNX Runtime. Its runtime environment
(``pyneat``, ``onnxruntime``, ``numpy==1.26.4``) is separate from the Studio UI
venv, so this class talks to a persistent worker process
(``supertonic_worker.py``) running in that environment over the same
length-prefixed stdin/stdout protocol the piper-tts worker uses. One shared
worker serves every voice and language.

The public interface matches the other engines (``synthesize``,
``synthesize_stream``, ``set_utterance_speed``, ``supports``) and adds
``set_voice`` for the ten Supertonic speakers (F1-F5, M1-M5).

This module imports neither ``pyneat`` nor ``supertonic_sima``; it is safe to
import in the main venv and on a development host. The segmenter below is pure
Python so it is unit-testable without the board.
"""

from __future__ import annotations

import io
import json
import logging
import os
import re
import struct
import subprocess
import threading
import unicodedata
import wave
from pathlib import Path

DEFAULT_REPO_ROOT = "/media/nvme/repos/supertonic-sima"
DEFAULT_APP_ROOT = "/media/nvme/supertonic-tts"
DEFAULT_VOICE = "M1"
VOICES = tuple(f"F{i}" for i in range(1, 6)) + tuple(f"M{i}" for i in range(1, 6))
VOICE_LABELS = {
    **{f"F{i}": f"F{i} (female {i})" for i in range(1, 6)},
    **{f"M{i}": f"M{i} (male {i})" for i in range(1, 6)},
}

# Compiled model contract: processed text, including the ``<lang>…</lang>``
# wrapper the engine adds, must be at most this many characters.
MAX_MODEL_CHARS = 192

_worker = None
_worker_lock = threading.Lock()


class WorkerDied(RuntimeError):
    """The worker reported a fatal runtime error and exited (status 3)."""


# --------------------------------------------------------------------------
# Environment discovery
# --------------------------------------------------------------------------

def repo_root():
    return Path(os.environ.get("SUPERTONIC_REPO_ROOT") or DEFAULT_REPO_ROOT)


def app_root():
    return Path(os.environ.get("SUPERTONIC_APP_ROOT") or DEFAULT_APP_ROOT)


def _supertonic_python():
    """Interpreter of the Supertonic runtime venv, or None when not installed."""
    p = os.environ.get("SUPERTONIC_PYTHON")
    if p:
        return p if Path(p).exists() else None
    cand = app_root() / ".venv" / "bin" / "python"
    return str(cand) if cand.exists() else None


def available():
    """True when the runtime venv, the checkout and the model assets exist."""
    if _supertonic_python() is None:
        return False
    if not (repo_root() / "app" / "supertonic_sima" / "__init__.py").is_file():
        return False
    models = app_root() / "models"
    return (models / "supertonic-3" / "onnx" / "tts.json").is_file() and (
        models / "supertonic-3-sima" / "supertonic_vector_field_sima_mpk.tar.gz"
    ).is_file()


def _worker_script():
    return str(Path(__file__).resolve().parent / "supertonic_worker.py")


# --------------------------------------------------------------------------
# Text segmentation (port of the upstream browser playground's chunker)
# --------------------------------------------------------------------------

_ENDING_PUNCTUATION = re.compile(r"[.!?;:,'\"')\]}…。」】〉》›»]$")
_BOUNDARIES = ".!?:;…。！？：；｡"
_CLOSERS = "\"'”’)\\]」』】〉》›»"
_SENTENCE_RE = re.compile(
    f"[^{re.escape(_BOUNDARIES)}]+[{re.escape(_BOUNDARIES)}]+[{re.escape(_CLOSERS)}]*"
    f"|[^{re.escape(_BOUNDARIES)}]+$"
)
_WHITESPACE = re.compile(r"\s+")


def _fits(value, max_chars):
    # The engine appends a period when a segment lacks ending punctuation.
    return len(value) + (0 if _ENDING_PUNCTUATION.search(value) else 1) <= max_chars


def _split_long(segment, max_chars):
    parts = []
    remaining = segment.strip()
    while remaining and not _fits(remaining, max_chars):
        window = remaining[: max_chars + 1]
        cut = max(window.rfind(","), window.rfind("，"))
        if cut < max_chars // 2:
            cut = remaining[:max_chars].rfind(" ")
            if cut < 1:
                cut = max_chars - 1
        else:
            cut += 1
        parts.append(remaining[:cut].strip())
        remaining = remaining[cut:].strip()
    if remaining:
        parts.append(remaining)
    return parts


def segment_text(text, language, max_model_chars=MAX_MODEL_CHARS):
    """Split ``text`` into segments that each fit the compiled model contract.

    Sentences are kept whole where possible and packed together while they fit;
    an oversized sentence falls back to commas, then whitespace, then a hard
    cut. Returns an empty list for blank input.
    """
    max_chars = max_model_chars - len(f"<{language}></{language}>")
    cleaned = _WHITESPACE.sub(" ", unicodedata.normalize("NFKD", text or "")).strip()
    if not cleaned:
        return []
    sentences = [m.group(0).strip() for m in _SENTENCE_RE.finditer(cleaned)]
    sentences = [s for s in sentences if s] or [cleaned]
    chunks = []
    current = ""
    for sentence in sentences:
        for part in _split_long(sentence, max_chars):
            candidate = f"{current} {part}" if current else part
            if _fits(candidate, max_chars):
                current = candidate
                continue
            if current:
                chunks.append(current)
            current = part
    if current:
        chunks.append(current)
    return chunks


# --------------------------------------------------------------------------
# Worker process management (same framing as pipertts.py)
# --------------------------------------------------------------------------

def _ensure_worker():
    global _worker
    if _worker is not None and _worker.poll() is None:
        return _worker
    py = _supertonic_python()
    if not py:
        raise RuntimeError(
            "Supertonic runtime venv not found (set SUPERTONIC_PYTHON or run "
            "setup.sh with the supertonic-sima repository available)")
    env = dict(os.environ)
    env.setdefault("SUPERTONIC_REPO_ROOT", str(repo_root()))
    env.setdefault("SUPERTONIC_APP_ROOT", str(app_root()))
    _worker = subprocess.Popen(
        [py, _worker_script()],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0, env=env,
    )
    return _worker


def _read_exact(stream, n):
    chunks = []
    while n > 0:
        b = stream.read(n)
        if not b:
            raise RuntimeError("Supertonic worker closed the pipe")
        chunks.append(b)
        n -= len(b)
    return b"".join(chunks)


def _read_frame(proc):
    status = _read_exact(proc.stdout, 1)[0]
    length = struct.unpack(">I", _read_exact(proc.stdout, 4))[0]
    return status, _read_exact(proc.stdout, length)


def _discard_worker():
    global _worker
    try:
        if _worker is not None:
            _worker.kill()
            _worker.wait(timeout=5)
    except Exception:
        pass
    _worker = None


def shutdown_worker():
    """Stop the worker (releases its MLA runners). The next request respawns it."""
    with _worker_lock:
        _discard_worker()


def _request(req):
    with _worker_lock:
        proc = _ensure_worker()
        try:
            proc.stdin.write((json.dumps(req) + "\n").encode("utf-8"))
            proc.stdin.flush()
            status, payload = _read_frame(proc)
        except Exception:
            _discard_worker()
            raise
        if status != 0:
            raise RuntimeError(payload.decode("utf-8", "replace"))
        return payload


def _request_stream(req):
    """Yield WAV chunks while holding the one-worker request lock."""
    complete = False
    with _worker_lock:
        proc = _ensure_worker()
        try:
            proc.stdin.write((json.dumps(req) + "\n").encode("utf-8"))
            proc.stdin.flush()
            while True:
                status, payload = _read_frame(proc)
                if status == 2:
                    yield payload
                elif status == 0:
                    complete = True
                    return
                elif status == 3:
                    # Fatal: the worker is exiting. Reap it now so the next
                    # request spawns a fresh one instead of writing to a corpse.
                    complete = True
                    _discard_worker()
                    raise WorkerDied(payload.decode("utf-8", "replace"))
                else:
                    complete = True
                    raise RuntimeError(payload.decode("utf-8", "replace"))
        finally:
            # Drain to the terminal frame if the consumer left early so the next
            # request starts on a clean pipe; discard the worker only if the
            # pipe itself fails.
            if not complete:
                try:
                    while True:
                        status, _ = _read_frame(proc)
                        if status in (0, 1, 3):
                            complete = True
                            if status == 3:
                                _discard_worker()
                            break
                except Exception:
                    pass
                if not complete:
                    _discard_worker()


# --------------------------------------------------------------------------
# Engine wrapper
# --------------------------------------------------------------------------

class SupertonicTTS:
    """Client wrapper for the Supertonic 3 MLA engine (all voices/languages)."""

    def __init__(self, voice=DEFAULT_VOICE):
        info = json.loads(_request({"cmd": "load"}).decode("utf-8"))
        self.languages = set(info.get("languages") or [])
        self.voices = tuple(info.get("voices") or VOICES)
        self.sample_rate = int(info.get("sample_rate") or 44100)
        self.steps = int(info.get("steps") or 8)
        self.min_speed = float(info.get("min_speed") or 0.7)
        self.max_speed = float(info.get("max_speed") or 2.0)
        self.voice = voice if voice in self.voices else self.voices[0]
        self.speed = 1.0
        logging.info("SupertonicTTS ready (voice %s, %d steps, %d languages)",
                     self.voice, self.steps, len(self.languages))

    # -- configuration -----------------------------------------------------
    def set_voice(self, voice):
        if voice not in self.voices:
            return False
        self.voice = voice
        return True

    def set_utterance_speed(self, speed):
        try:
            speed = float(speed)
        except (TypeError, ValueError):
            speed = 1.0
        # The Studio slider allows 0.5x; the model contract starts at 0.7x.
        self.speed = max(self.min_speed, min(self.max_speed, speed))

    def supports(self, language):
        return bool(language) and language in self.languages

    # -- synthesis -----------------------------------------------------------
    def synthesize_stream(self, text, language=None, voice=None):
        """Yield one WAV ``BytesIO`` per text segment. ``voice`` overrides the
        configured speaker for this call only (ignored when unknown)."""
        language = language if self.supports(language) else "en"
        voice = voice if voice in self.voices else self.voice
        segments = segment_text(text, language)
        if not segments:
            return
        req = {
            "cmd": "synth_stream", "segments": segments, "voice": voice,
            "language": language, "speed": self.speed,
        }
        # The worker exits on a runtime failure (e.g. its MLA runners died under
        # an accelerator reset). If that happened before any audio was produced,
        # respawn it and retry this utterance once.
        for attempt in (0, 1):
            yielded = False
            stream = _request_stream(req)
            try:
                for data in stream:
                    yielded = True
                    yield io.BytesIO(data)
                return
            except WorkerDied as exc:
                if yielded or attempt:
                    raise
                logging.warning("Supertonic worker died (%s); respawning and retrying once", exc)
            finally:
                stream.close()

    def synthesize(self, text, language=None, voice=None):
        chunks = list(self.synthesize_stream(text, language=language, voice=voice))
        if not chunks:
            return io.BytesIO()
        if len(chunks) == 1:
            return chunks[0]
        frames = []
        wav_format = None
        for chunk in chunks:
            chunk.seek(0)
            with wave.open(chunk, "rb") as wav_file:
                current = (wav_file.getnchannels(), wav_file.getsampwidth(),
                           wav_file.getframerate())
                if wav_format is None:
                    wav_format = current
                elif current != wav_format:
                    raise RuntimeError("Supertonic stream changed WAV format between chunks")
                frames.append(wav_file.readframes(wav_file.getnframes()))
        output = io.BytesIO()
        with wave.open(output, "wb") as wav_file:
            wav_file.setnchannels(wav_format[0])
            wav_file.setsampwidth(wav_format[1])
            wav_file.setframerate(wav_format[2])
            wav_file.writeframes(b"".join(frames))
        output.seek(0)
        return output

    def save_audio(self, buffer, filename):
        with open(filename, "wb") as f:
            buffer.seek(0)
            f.write(buffer.read())


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test the Supertonic worker client.")
    parser.add_argument("text")
    parser.add_argument("-v", "--voice", default=DEFAULT_VOICE, choices=VOICES)
    parser.add_argument("-l", "--lang", default="en")
    parser.add_argument("-o", "--output", default="output.wav")
    args = parser.parse_args()
    tts = SupertonicTTS(voice=args.voice)
    tts.save_audio(tts.synthesize(args.text, language=args.lang), args.output)
    print(f"Saved to {args.output}")
