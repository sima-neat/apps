#!/usr/bin/env python3
"""Persistent piper-tts synthesis worker (runs in the isolated .venv-pipertts).

piper-tts and piper-plus both ship a top-level ``piper`` package and cannot
coexist in one venv, so piper-tts lives in its own virtualenv and the app talks
to it through this subprocess. Protocol over stdin/stdout:

    request  (stdin) : one JSON object per line, e.g.
        {"cmd": "load",  "model": "/abs/voice.onnx"}
        {"cmd": "synth_stream", "model": "/abs/voice.onnx", "text": "..."}
    response (stdout): 1 status byte + 4-byte big-endian length + payload.
        0 = complete, 1 = error, 2 = streaming WAV chunk. A streaming response
        ends with an empty status-0 frame.

Voices are loaded lazily and cached by model path, so ONE worker process serves
every rhasspy language. This module must run in a venv that has ``piper-tts``.
"""

import io
import json
import os
import struct
import sys
import wave

import numpy as np
import onnxruntime


DEFAULT_NUM_THREADS = 8
CHUNK_FRAMES = 45       # about 0.5 seconds of audio
CHUNK_PADDING = 10      # decoder context on each side; trimmed before emitting
STREAM_GAIN = 2.0       # makeup for whole-utterance peak normalization


def main():
    voices = {}

    def make_voice_shell(model):
        """Piper phonemizer/config without loading the original ONNX session."""
        from piper import PiperVoice
        from piper.config import PiperConfig

        with open(f"{model}.json", encoding="utf-8") as config_file:
            config = PiperConfig.from_dict(json.load(config_file))
        return PiperVoice(session=None, config=config)

    def make_session(model):
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = DEFAULT_NUM_THREADS
        return onnxruntime.InferenceSession(
            model, sess_options=options, providers=["CPUExecutionProvider"]
        )

    def enable_streaming(model, entry):
        if entry.get("enc") is not None:
            return True
        base = os.path.splitext(model)[0]
        encoder_path = f"{base}.enc.onnx"
        decoder_path = f"{base}.dec.onnx"
        if not (os.path.exists(encoder_path) and os.path.exists(decoder_path)):
            return False

        enc = make_session(encoder_path)
        dec = make_session(decoder_path)
        dec_input = dec.get_inputs()[0].name

        phonemes = entry["voice"].phonemize("Ready.")[0]
        ids = entry["voice"].phonemes_to_ids(phonemes)
        latent = enc.run(None, encoder_args(ids, entry["cfg"]))[0]
        audio = dec.run(None, {dec_input: latent})[0].squeeze()
        entry.update(
            enc=enc,
            dec=dec,
            dec_input=dec_input,
            upsample=round(audio.shape[0] / latent.shape[2]),
        )
        return True

    def get_voice(model):
        if model not in voices:
            try:
                from piper import SynthesisConfig
                cfg = SynthesisConfig(
                    length_scale=1.0, noise_scale=0.667, noise_w_scale=0.8,
                    volume=1.0, normalize_audio=True,
                )
            except ImportError:
                cfg = None
            voices[model] = {
                "voice": make_voice_shell(model), "cfg": cfg,
                "enc": None, "dec": None, "dec_input": None, "upsample": None,
            }
        entry = voices[model]
        if not enable_streaming(model, entry):
            base = os.path.splitext(model)[0]
            raise RuntimeError(
                "split Piper voice is required; missing "
                f"{base}.enc.onnx or {base}.dec.onnx"
            )
        return entry

    def encoder_args(phoneme_ids, cfg):
        return {
            "input": np.expand_dims(np.array(phoneme_ids, dtype=np.int64), 0),
            "input_lengths": np.array([len(phoneme_ids)], dtype=np.int64),
            "scales": np.array(
                [cfg.noise_scale, cfg.length_scale, cfg.noise_w_scale],
                dtype=np.float32,
            ),
        }

    def wav_bytes(sample_rate, samples):
        pcm = np.clip(samples * 32767.0, -32767.0, 32767.0).astype(np.int16)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm.tobytes())
        return buf.getvalue()

    def stream_voice(entry, text):
        voice = entry["voice"]
        cfg = entry["cfg"]
        for phonemes in voice.phonemize(text):
            if not phonemes:
                continue
            ids = voice.phonemes_to_ids(phonemes)
            latent = entry["enc"].run(None, encoder_args(ids, cfg))[0]
            total_frames = latent.shape[2]
            start = 0
            while start < total_frames:
                end = min(start + CHUNK_FRAMES, total_frames)
                padded_start = max(0, start - CHUNK_PADDING)
                padded_end = min(total_frames, end + CHUNK_PADDING)
                window = np.ascontiguousarray(latent[:, :, padded_start:padded_end])
                audio = entry["dec"].run(
                    None, {entry["dec_input"]: window}
                )[0].squeeze()
                head = (start - padded_start) * entry["upsample"]
                tail = audio.shape[0] - (padded_end - end) * entry["upsample"]
                samples = np.clip(audio[head:tail] * STREAM_GAIN * cfg.volume, -1.0, 1.0)
                yield wav_bytes(voice.config.sample_rate, samples)
                start = end

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
            cmd = req.get("cmd")
            entry = get_voice(model)
            cfg = entry["cfg"]
            if cmd == "load":
                respond(0, b"")
                continue
            if cmd != "synth_stream":
                raise ValueError(f"unsupported command: {cmd}")
            if cfg is not None:
                try:
                    cfg.length_scale = 1.0 / max(0.5, min(2.0, float(req.get("speed", 1.0))))
                except (TypeError, ValueError):
                    pass
            for chunk in stream_voice(entry, req.get("text", "")):
                respond(2, chunk)
            respond(0, b"")
        except Exception as exc:  # noqa: BLE001 - report to the client, keep serving
            try:
                respond(1, str(exc).encode("utf-8", "replace"))
            except Exception:
                pass


if __name__ == "__main__":
    main()
