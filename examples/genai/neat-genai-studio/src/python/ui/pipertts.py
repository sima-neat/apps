"""PiperTTS — client to an isolated piper-tts synthesis worker.

piper-tts and piper-plus both ship a top-level ``piper`` package, so they cannot
share one virtualenv. piper-tts is installed in a separate venv
(``.venv-pipertts``) and this class talks to a persistent worker process
(``pipertts_worker.py``) running there, over a length-prefixed stdin/stdout
protocol. One shared worker serves every rhasspy voice (loaded lazily by path).

The public interface matches the other engines and additionally exposes
``synthesize_stream`` using the voice's required encoder/decoder pair.

This module imports **no** ``piper`` package itself, so it is safe to import in
the piper-plus (main) venv.
"""

import base64
import io
import json
import os
import struct
import subprocess
import threading
import wave
from pathlib import Path

_worker = None
_worker_lock = threading.Lock()


def _pipertts_python():
    """Locate the Python interpreter of the isolated piper-tts venv, or None."""
    p = os.environ.get("PIPERTTS_PYTHON")
    if p and Path(p).exists():
        return p
    here = Path(__file__).resolve()
    # .../src/python/ui/pipertts.py -> example dir is parents[3]
    candidates = []
    if len(here.parents) > 3:
        candidates.append(here.parents[3] / ".venv-pipertts" / "bin" / "python")
    candidates.append(Path.cwd() / ".venv-pipertts" / "bin" / "python")
    for cand in candidates:
        if cand.exists():
            return str(cand)
    return None


def _worker_script():
    return str(Path(__file__).resolve().parent / "pipertts_worker.py")


def _ensure_worker():
    global _worker
    if _worker is not None and _worker.poll() is None:
        return _worker
    py = _pipertts_python()
    if not py:
        raise RuntimeError(
            "piper-tts venv not found (set PIPERTTS_PYTHON or run setup.sh to "
            "create .venv-pipertts)")
    _worker = subprocess.Popen(
        [py, _worker_script()],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0,
    )
    return _worker


def _read_exact(stream, n):
    chunks = []
    while n > 0:
        b = stream.read(n)
        if not b:
            raise RuntimeError("piper-tts worker closed the pipe")
        chunks.append(b)
        n -= len(b)
    return b"".join(chunks)


def _discard_worker():
    global _worker
    try:
        if _worker is not None:
            _worker.kill()
            _worker.wait(timeout=5)
    except Exception:
        pass
    _worker = None


def _request(req):
    global _worker
    with _worker_lock:
        proc = _ensure_worker()
        try:
            proc.stdin.write((json.dumps(req) + "\n").encode("utf-8"))
            proc.stdin.flush()
            status = _read_exact(proc.stdout, 1)[0]
            length = struct.unpack(">I", _read_exact(proc.stdout, 4))[0]
            payload = _read_exact(proc.stdout, length)
        except Exception:
            # A broken worker is unusable — kill it so the next call respawns.
            _discard_worker()
            raise
        if status != 0:
            raise RuntimeError(payload.decode("utf-8", "replace"))
        return payload


def _request_stream(req):
    """Yield framed WAV chunks while holding the one-worker request lock."""
    global _worker
    complete = False
    with _worker_lock:
        proc = _ensure_worker()
        try:
            proc.stdin.write((json.dumps(req) + "\n").encode("utf-8"))
            proc.stdin.flush()
            while True:
                status = _read_exact(proc.stdout, 1)[0]
                length = struct.unpack(">I", _read_exact(proc.stdout, 4))[0]
                payload = _read_exact(proc.stdout, length)
                if status == 2:
                    yield payload
                elif status == 0:
                    complete = True
                    return
                else:
                    raise RuntimeError(payload.decode("utf-8", "replace"))
        finally:
            # Leaving early would leave unread chunk frames in stdout and corrupt
            # the next request, so discard this worker and respawn on demand.
            if not complete:
                _discard_worker()


def prepare_voice_for_streaming(model_path):
    """Split a newly downloaded voice using the isolated Piper environment."""
    model_path = Path(model_path)
    encoder = model_path.with_suffix(".enc.onnx")
    decoder = model_path.with_suffix(".dec.onnx")
    if (
        encoder.is_file()
        and decoder.is_file()
        and min(encoder.stat().st_mtime, decoder.stat().st_mtime)
        >= model_path.stat().st_mtime
    ):
        return True
    py = _pipertts_python()
    if not py:
        return False
    try:
        result = subprocess.run(
            [
                py,
                str(Path(__file__).resolve().parent.parent / "split_voices.py"),
                str(model_path),
            ],
            check=False,
            timeout=180,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0 and encoder.is_file() and decoder.is_file()


class PiperTTS:
    """Client wrapper for a single rhasspy piper-tts voice (one model path)."""

    DEFAULT_MODEL_PATH = "assets/en_US-amy-low.onnx"
    DEFAULT_SAMPLE_RATE = 22050

    def __init__(self, model_path=None, sample_rate=DEFAULT_SAMPLE_RATE, use_cuda=False,
                 config=None):
        self.model_path = str(model_path or self.DEFAULT_MODEL_PATH)
        self.sample_rate = sample_rate
        self.speed = 1.0
        # Warm the split voice now so missing/invalid encoder or decoder models
        # fail fast. Dedicated Piper voices never load the unsplit ONNX model.
        _request({"cmd": "load", "model": self.model_path})

    def set_utterance_speed(self, speed):
        try:
            speed = float(speed)
        except (TypeError, ValueError):
            speed = 1.0
        self.speed = max(0.5, min(2.0, speed))

    def synthesize(self, text, language=None):
        chunks = list(self.synthesize_stream(text, language=language))
        if not chunks:
            return io.BytesIO()
        frames = []
        wav_format = None
        for chunk in chunks:
            chunk.seek(0)
            with wave.open(chunk, "rb") as wav_file:
                current_format = (
                    wav_file.getnchannels(), wav_file.getsampwidth(),
                    wav_file.getframerate(),
                )
                if wav_format is None:
                    wav_format = current_format
                elif current_format != wav_format:
                    raise RuntimeError("Piper stream changed WAV format between chunks")
                frames.append(wav_file.readframes(wav_file.getnframes()))
        output = io.BytesIO()
        with wave.open(output, "wb") as wav_file:
            wav_file.setnchannels(wav_format[0])
            wav_file.setsampwidth(wav_format[1])
            wav_file.setframerate(wav_format[2])
            wav_file.writeframes(b"".join(frames))
        output.seek(0)
        return output

    def synthesize_stream(self, text, language=None):
        stream = _request_stream({
            "cmd": "synth_stream", "model": self.model_path,
            "text": text, "speed": self.speed,
        })
        try:
            for data in stream:
                yield io.BytesIO(data)
        finally:
            stream.close()

    def synthesize_base64(self, text):
        return base64.b64encode(self.synthesize(text).read()).decode("utf-8")

    def save_audio(self, buffer, filename):
        with open(filename, "wb") as f:
            buffer.seek(0)
            f.write(buffer.read())


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test the piper-tts worker client.")
    parser.add_argument("text")
    parser.add_argument("-m", "--model", default=PiperTTS.DEFAULT_MODEL_PATH)
    parser.add_argument("-o", "--output", default="output.wav")
    args = parser.parse_args()
    tts = PiperTTS(model_path=args.model)
    tts.save_audio(tts.synthesize(args.text), args.output)
    print(f"Saved to {args.output}")
