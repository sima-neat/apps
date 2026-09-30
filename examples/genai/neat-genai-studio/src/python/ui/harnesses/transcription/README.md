# Transcription API harness

A single static page for trying the Studio's OpenAI-compatible speech-to-text
endpoint. Served at `/solutions/transcription/` by the Studio and opened from the
**Solutions** grid (the *Transcription API* card).

Audio comes from the microphone (`MediaRecorder`, same container preference as
the Studio's own recorder: WebM/Opus, then Ogg/Opus, MP4, WAV) or from a chosen
file. The clip is previewed in the page and uploaded with its real filename and
MIME type; the model server decodes it with libavformat, so WebM, MP4/AAC, WAV,
MP3 and FLAC all work. WAV is the portable choice for scripted calls.

What it calls (relative to its own origin):

- `POST /v1/audio/transcriptions` as `multipart/form-data` with `file`,
  optional `language` (omitted for auto-detect), optional `model` (default: the
  active speech-to-text model) and `response_format` (`json`, `verbose_json`,
  `text`). The transcript is shown with the `verbose_json` metadata (language
  and whether it was detected, `no_speech_prob`, `avg_logprob`, the Studio's
  ignore decision) and the `X-ASR-Model` / `X-Elapsed-Time` headers; a `curl`
  equivalent is printed for scripting.

Microphone capture needs a secure context (HTTPS or localhost); the Studio
serves HTTPS by default. Rules, as for the rest of the suite: zero external
network requests and all assets local. Audit:

```bash
grep -nE 'src=|href=|@import|@font-face|url\(' transcription/index.html transcription/style.css transcription/app.js
```
