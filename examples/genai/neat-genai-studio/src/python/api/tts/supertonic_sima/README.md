# Supertonic 3 runtime (vendored)

The Studio's Supertonic 3 text-to-speech engine. Vendored from
<https://github.com/florianvoss-commit/supertonic-sima> (`app/supertonic_sima/`) at
commit `3b837b3e1b6a378ab8c24c3c04b079429b67e237`, at the upstream author's request,
so the Studio does not clone or execute an external repository at install time.

Only the runtime closure is included: `__init__.py`, `audio.py`, `engine.py`,
`inputs.py`, `text.py`. Upstream's `config.py` (argparse defaults for its example
apps) and `server.py` (its standalone HTTP server) are deliberately omitted; the
graph surgery and compilation tooling stays upstream. The local changes are in
`engine.py`: paths and model-asset checksums are supplied by the
worker. Tensor bindings preserve the batch axis and convert explicitly between
NCHW and NHWC, including the vocoder's batch-one axis. The vector estimator
runs conditional guidance in slot 0 and
unconditional guidance in slot 1 of one batch-two call per Euler step; the
vocoder remains batch one.

It runs only inside `supertonic_worker.py`, in the isolated `.venv-supertonic`
environment that `setup.sh` builds (PyNeat, onnxruntime, numpy). The Studio
UI process never imports it. `setup.sh` downloads the CPU frontend and voices
from the pinned `Supertone/supertonic-3` revision, runtime constants from
`florianvoss/supertonic-3-sima`, and both compiled models from the SiMa model
zoo. `model_assets.json` supplies the compiled-model URLs and the checksums
shared by setup and execution. See `THIRD_PARTY_TTS_MODELS.md` for attribution.


## Refreshing from upstream

1. Pick the upstream commit and review its diff against the current vendored copy.
2. Copy the five files, keep the provenance header at the top of each, update the
   commit in the headers and here, and re-apply the local changes above.
3. If the model contract changed (new MPK, runtime data or manifest), update the
   pinned revisions in `setup.sh` and checksums in `model_assets.json`.
4. Run the unit suite and the DevKit checks in the README's Verify section.
