# Speech API harness

A single static page for trying the Studio's OpenAI-compatible text-to-speech
endpoint without the chat UI. Served at `/solutions/speech/` by the Studio and
opened from the **Solutions** grid (the *Speech API* card).

What it calls (relative to its own origin, so no CORS and no mixed content):

- `GET /v1/audio/voices` on load, to fill the engine, voice and language pickers.
  Engines that are installed but not loaded are marked; picking one loads it on
  the first request.
- `POST /v1/audio/speech` with `{input, model, voice?, language, speed,
  response_format: "wav"}`. The WAV is played in the page and offered for
  download; the `X-Engine`, `X-Voice`, `X-Speed`, `X-Audio-Duration`,
  `X-Elapsed-Time` and `X-RTF` headers are shown as stats, and the exact JSON
  body plus a `curl` equivalent are printed so the same call can be scripted.

Errors are shown verbatim: a 400 carries `param`, a 503 carries the engine's
refusal `reason` (for example a language the chosen engine cannot speak).

Rules, as for the rest of the suite: zero external network requests, all assets
local (`../SiMaSentry.png` favicon, `style.css`, `app.js`), settings in
`localStorage` under `sima-studio:speech-harness`. Audit:

```bash
grep -nE 'src=|href=|@import|@font-face|url\(' speech/index.html speech/style.css speech/app.js
```
