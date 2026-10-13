# Supertonic 3 runtime (vendored)

The Studio's Supertonic 3 text-to-speech engine. Vendored from
<https://github.com/florianvoss-commit/supertonic-sima> (`app/supertonic_sima/`) at
commit `3b837b3e1b6a378ab8c24c3c04b079429b67e237`, at the upstream author's request,
so the Studio does not clone or execute an external repository at install time.

Only the runtime closure is included: `__init__.py`, `audio.py`, `engine.py`,
`inputs.py`, `text.py`. Upstream's `config.py` (argparse defaults for its example
apps) and `server.py` (its standalone HTTP server) are deliberately omitted; the
graph surgery and compilation tooling stays upstream. The one local change is the
removal of two absolute default paths from `engine.py` (the worker passes every
path explicitly).

It runs only inside `supertonic_worker.py`, in the isolated `.venv-supertonic`
environment that `setup.sh` builds (PyNeat, onnxruntime, numpy 1.26). The Studio
UI process never imports it. The model files it needs are downloaded by `setup.sh`
from Hugging Face at pinned revisions (`Supertone/supertonic-3` and the compiled
MLA packages in `florianvoss/supertonic-3-sima`) and verified by checksum; see
`THIRD_PARTY_TTS_MODELS.md` for attribution and licence notes.

## Refreshing from upstream

1. Pick the upstream commit and review its diff against the current vendored copy.
2. Copy the five files, keep the provenance header at the top of each, update the
   commit in the headers and here, and re-apply the local change above.
3. If the model contract changed (new MPK, runtime data or manifest), update the
   pinned Hugging Face revisions and checksums in `setup.sh`.
4. Run the unit suite and the DevKit checks in the README's Verify section.
