#!/usr/bin/env python3
"""Persistent Supertonic 3 synthesis worker (runs in the Supertonic runtime venv).

Supertonic runs its vector field and vocoder on the MLA through PyNeat and its
duration predictor / text encoder on the CPU through ONNX Runtime. Those
packages (``pyneat``, ``onnxruntime``, ``numpy==1.26.4``) are not part of the
Studio UI environment, so the engine lives in the venv created by the
`supertonic-sima <https://github.com/florianvoss-commit/supertonic-sima>`_
repository (``scripts/setup_devkit.sh``) and the UI talks to it through this
subprocess. A crashed or reset worker is simply respawned by the client, which
also covers the Studio's supervised accelerator reset: restarting the MLA
dispatcher tears down every runner in every process, including this one.

Protocol over stdin/stdout (identical to ``pipertts_worker.py``; the worker
duplicates its stdout before the Neat runtime initializes because the runtime
redirects fd 1, see ``_claim_protocol_channel``):

    request  (stdin) : one JSON object per line
        {"cmd": "load"}
        {"cmd": "synth_stream", "segments": ["..."], "voice": "M1",
         "language": "en", "speed": 1.0}
    response (stdout): 1 status byte + 4-byte big-endian length + payload.
        0 = complete, 1 = error (bad request; worker keeps serving),
        2 = streaming WAV chunk, 3 = fatal error (the worker exits right after
        this frame; the client respawns it). ``load`` answers with a status-0
        JSON payload describing the engine (languages, voices, sample rate).
        A streaming response ends with an empty status-0 frame.

Each element of ``segments`` must already fit the compiled 192-character
contract (the client splits, see ``supertonic_tts.segment_text``). One WAV is
emitted per segment so playback can start after the first one. Trailing
silence generated at the end of every segment but the last is trimmed with a
short retained tail and fade, so consecutive segments join without a gap.

Environment:
    SUPERTONIC_REPO_ROOT   supertonic-sima checkout (its ``app/`` is imported)
    SUPERTONIC_APP_ROOT    models root; ``models/supertonic-3`` and
                           ``models/supertonic-3-sima`` live below it
    SUPERTONIC_STEPS       Euler denoising steps, 5-12 (default 8)
    SUPERTONIC_THREADS     ONNX Runtime intra-op threads (default min(8, cpus))
"""

from __future__ import annotations

import io
import json
import os
import struct
import sys
import wave
from pathlib import Path

import numpy as np


DEFAULT_REPO_ROOT = "/media/nvme/repos/supertonic-sima"
DEFAULT_APP_ROOT = "/media/nvme/supertonic-tts"
DEFAULT_SEED = 1101
# Tail trimming between segments (mirrors the upstream browser playground).
FRAME_SECONDS = 0.01
RETAINED_TAIL_SECONDS = 0.08
MIN_TRIM_SECONDS = 0.12
FADE_SECONDS = 0.02


def _import_runtime():
    repo_root = Path(os.environ.get("SUPERTONIC_REPO_ROOT") or DEFAULT_REPO_ROOT)
    app_dir = repo_root / "app"
    if not (app_dir / "supertonic_sima" / "__init__.py").is_file():
        raise RuntimeError(
            f"supertonic_sima package not found under {app_dir}; set "
            "SUPERTONIC_REPO_ROOT to a supertonic-sima checkout"
        )
    if str(app_dir) not in sys.path:
        sys.path.insert(0, str(app_dir))
    import supertonic_sima  # noqa: WPS433 - deferred, venv-specific import

    return supertonic_sima


def _build_engine(runtime):
    app_root = Path(os.environ.get("SUPERTONIC_APP_ROOT") or DEFAULT_APP_ROOT)
    model_root = app_root / "models"
    compiled = model_root / "supertonic-3-sima"
    steps = int(os.environ.get("SUPERTONIC_STEPS") or 8)
    if not 5 <= steps <= 12:
        raise RuntimeError("SUPERTONIC_STEPS must be between 5 and 12")
    threads = int(os.environ.get("SUPERTONIC_THREADS") or min(8, os.cpu_count() or 1))
    return runtime.SupertonicModalix(
        model_dir=model_root / "supertonic-3",
        mpk_path=compiled / "supertonic_vector_field_sima_mpk.tar.gz",
        runtime_data_path=compiled / "supertonic_runtime_data.npz",
        artifact_manifest_path=compiled / "artifact_manifest.json",
        vocoder_backend="mla",
        vocoder_mpk_path=compiled / "supertonic_vocoder_sima_bf16_mpk.tar.gz",
        vocoder_manifest_path=compiled / "vocoder_bf16_manifest.json",
        threads=max(1, threads),
        timeout_ms=120_000,
        verify_hashes=True,
        steps=steps,
    )


def trim_trailing_silence(waveform: np.ndarray, sample_rate: int) -> np.ndarray:
    """Drop generated trailing silence, keeping a short tail with a fade-out.

    Frame RMS is measured over 10 ms windows; frames at or below 5 % of the
    peak RMS (with an absolute floor) count as silence. Nothing is trimmed when
    less than 120 ms would be removed, so ordinary sentence endings stay intact.
    """
    samples = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if samples.size == 0:
        return samples
    frame = max(1, round(sample_rate * FRAME_SECONDS))
    frame_count = -(-samples.size // frame)
    padded = np.zeros(frame_count * frame, dtype=np.float32)
    padded[: samples.size] = samples
    rms = np.sqrt(np.mean(padded.reshape(frame_count, frame) ** 2, axis=1))
    threshold = max(0.0015, float(rms.max()) * 0.05)
    active = np.nonzero(rms > threshold)[0]
    if active.size == 0:
        return samples
    last_active = int(active[-1])
    cutoff = min(samples.size, (last_active + 1) * frame + round(sample_rate * RETAINED_TAIL_SECONDS))
    if samples.size - cutoff < round(sample_rate * MIN_TRIM_SECONDS):
        return samples
    trimmed = samples[:cutoff].copy()
    fade = min(round(sample_rate * FADE_SECONDS), cutoff)
    if fade > 1:
        trimmed[cutoff - fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
    return trimmed


def wav_bytes(samples: np.ndarray, sample_rate: int) -> bytes:
    pcm = np.clip(samples * 32767.0, -32767.0, 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())
    return buf.getvalue()


def _claim_protocol_channel():
    """Return an unbuffered writer on the parent's stdout pipe.

    The Neat runtime re-plumbs file descriptor 1 while a model loads (it
    points at a runtime-owned pipe afterwards), so anything written through
    ``sys.stdout`` from then on never reaches the client. Duplicate the
    original descriptor first and send every protocol frame through the copy;
    fd 1 itself is pointed at stderr so stray prints cannot corrupt the frames.
    """
    protocol_fd = os.dup(sys.stdout.fileno())
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.stdout = sys.stderr
    return os.fdopen(protocol_fd, "wb", buffering=0)


def main() -> None:
    out = _claim_protocol_channel()

    def respond(status: int, payload: bytes) -> None:
        out.write(bytes([status]))
        out.write(struct.pack(">I", len(payload)))
        out.write(payload)
        out.flush()

    runtime = None
    engine = None

    def ensure_engine():
        nonlocal runtime, engine
        if runtime is None:
            runtime = _import_runtime()
        if engine is None:
            engine = _build_engine(runtime)
        return engine

    def describe() -> bytes:
        eng = ensure_engine()
        return json.dumps({
            "languages": sorted(runtime.AVAILABLE_LANGUAGES),
            "voices": list(runtime.AVAILABLE_VOICES),
            "sample_rate": int(eng.sample_rate),
            "steps": int(eng.steps),
            "min_speed": runtime.MIN_SPEED,
            "max_speed": runtime.MAX_SPEED,
        }).encode("utf-8")

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            cmd = req.get("cmd")
            if cmd == "load":
                respond(0, describe())
                continue
            if cmd != "synth_stream":
                raise ValueError(f"unsupported command: {cmd}")
            eng = ensure_engine()
            voice = str(req.get("voice") or "M1")
            language = str(req.get("language") or "en")
            if voice not in runtime.AVAILABLE_VOICES:
                raise ValueError(f"unsupported voice {voice!r}")
            if language not in runtime.AVAILABLE_LANGUAGES:
                raise ValueError(f"unsupported language {language!r}")
            speed = float(req.get("speed", 1.0))
            speed = max(runtime.MIN_SPEED, min(runtime.MAX_SPEED, speed))
            seed = int(req.get("seed", DEFAULT_SEED))
            segments = [str(s) for s in req.get("segments", []) if str(s).strip()]
            for index, segment in enumerate(segments):
                result = eng.synthesize(
                    segment, voice=voice, language=language, speed=speed, seed=seed
                )
                waveform = result.waveform
                if index < len(segments) - 1:
                    waveform = trim_trailing_silence(waveform, result.sample_rate)
                respond(2, wav_bytes(waveform, result.sample_rate))
            respond(0, b"")
        except (ValueError, KeyError, TypeError) as exc:
            # Bad request (unknown voice, text too long, ...): report and keep serving.
            try:
                respond(1, f"{type(exc).__name__}: {exc}".encode("utf-8", "replace"))
            except Exception:
                pass
        except Exception as exc:  # noqa: BLE001
            # Anything else is a runtime failure, typically the MLA runners dying
            # under an accelerator reset. Report it as fatal, then exit so the
            # client respawns a fresh process (reloading the models).
            try:
                respond(3, f"{type(exc).__name__}: {exc}".encode("utf-8", "replace"))
            except Exception:
                pass
            raise SystemExit(1)


if __name__ == "__main__":
    main()
