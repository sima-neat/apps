#!/usr/bin/env python3
"""Ask the runtime whether it can load a speech model. Used by setup.sh.

Exit codes
  0  loaded and transcribed
  3  the runtime REFUSED this model (its message on stdout) — do not configure it
  1  could not tell (no runtime, busy accelerator, bad arguments)

Encoder layout requirements have changed between runtime builds in both
directions, so this asks the runtime instead of inspecting filenames: only an
explicit refusal is treated as a refusal, and anything ambiguous is reported as
unknown so a busy accelerator never discards a working model.
"""

from __future__ import annotations

import io
import sys
import wave

# Phrases by which a runtime says "this model is not for me", as opposed to
# transient accelerator trouble.
_REFUSAL = (
    "unsupported legacy",
    "requires",
    "model file does not exist",
    "unsupported",
    "incompatible",
)
_INCONCLUSIVE = ("mlashm", "mla_load", "dispatcher", "failed to acquire",
                 "cannot allocate memory", "out of memory", "shm memory handle")


def _silence_wav(seconds: float = 1.0, rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(b"\x00\x00" * int(rate * seconds))
    return buf.getvalue()


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: probe_asr.py <model-dir>", file=sys.stderr)
        return 1
    model_dir = sys.argv[1]
    try:
        import pyneat
    except Exception as exc:  # noqa: BLE001
        print(f"pyneat unavailable: {exc}")
        return 1

    tmp = None
    try:
        import tempfile, os
        fd, tmp = tempfile.mkstemp(suffix=".wav")
        with os.fdopen(fd, "wb") as fh:
            fh.write(_silence_wav())
        model = pyneat.ASRModel(model_dir)
        try:
            request = pyneat.GenerationRequest()
            for attr, value in (("audio_file", tmp), ("language", "en")):
                try:
                    setattr(request, attr, value)
                except Exception:  # noqa: BLE001 - older/newer field names
                    pass
            model.run(request)
        except AttributeError:
            model.run(tmp)
    except Exception as exc:  # noqa: BLE001
        detail = str(exc).replace("\n", " ")
        print(detail)
        low = detail.lower()
        if any(k in low for k in _INCONCLUSIVE):
            return 1          # accelerator trouble: says nothing about the model
        if any(k in low for k in _REFUSAL):
            return 3          # the runtime rejected this model
        return 1
    finally:
        if tmp:
            try:
                import os
                os.unlink(tmp)
            except OSError:
                pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
