#!/usr/bin/env python3
"""Persistent piper-tts synthesis worker (runs in the isolated .venv-pipertts).

piper-tts and piper-plus both ship a top-level ``piper`` package and cannot
coexist in one venv, so piper-tts lives in its own virtualenv and the app talks
to it through this subprocess. Protocol over stdin/stdout:

    request  (stdin) : one JSON object per line, e.g.
        {"cmd": "load",  "model": "/abs/voice.onnx"}
        {"cmd": "synth", "model": "/abs/voice.onnx", "text": "...", "speed": 1.0}
    response (stdout): 1 status byte (0 = ok, 1 = error) + 4-byte big-endian
        length + payload (WAV bytes on ok; UTF-8 error text on error)

Voices are loaded lazily and cached by model path, so ONE worker process serves
every rhasspy language. This module must run in a venv that has ``piper-tts``.
"""

import io
import json
import struct
import sys
import wave


def main():
    voices = {}

    def get_voice(model):
        if model not in voices:
            from piper import PiperVoice
            try:
                from piper import SynthesisConfig
                cfg = SynthesisConfig(
                    length_scale=1.0, noise_scale=0.667, noise_w_scale=0.8,
                    volume=1.0, normalize_audio=True,
                )
            except ImportError:
                cfg = None
            voices[model] = [PiperVoice.load(model), cfg]
        return voices[model]

    out = sys.stdout.buffer

    def respond(status, payload):
        out.write(bytes([status]))
        out.write(struct.pack(">I", len(payload)))
        out.write(payload)
        out.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            model = req["model"]
            voice, cfg = get_voice(model)
            if req.get("cmd", "synth") == "load":
                respond(0, b"")
                continue
            if cfg is not None:
                try:
                    cfg.length_scale = 1.0 / max(0.5, min(2.0, float(req.get("speed", 1.0))))
                except (TypeError, ValueError):
                    pass
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wav_file:
                if cfg is not None:
                    voice.synthesize_wav(req.get("text", ""), wav_file, syn_config=cfg)
                else:
                    voice.synthesize_wav(req.get("text", ""), wav_file)
            respond(0, buf.getvalue())
        except Exception as exc:  # noqa: BLE001 - report to the client, keep serving
            try:
                respond(1, str(exc).encode("utf-8", "replace"))
            except Exception:
                pass


if __name__ == "__main__":
    main()
