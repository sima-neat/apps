#!/usr/bin/env python3
"""Compatibility shim — temporary, remove one release after src/python/ui/
became src/python/api/.

The voice splitter is ``api/tts/split_voices.py``. ``run.sh update`` from a
release before the rename runs that release's ``migrate_piper_voices``, which
calls this path after the new source is in place. Forward to the new location
so that update still succeeds. Later updates call the new path directly.
"""

import os
import runpy
import sys

_REAL = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "api", "tts", "split_voices.py",
)

if not os.path.isfile(_REAL):
    sys.stderr.write(
        "split_voices.py moved to api/tts/split_voices.py and the target is "
        "missing; re-run setup.sh.\n"
    )
    raise SystemExit(1)

# Run the relocated script as __main__ with the same arguments (argv[1] is the
# assets dir passed by migrate_piper_voices).
sys.argv[0] = _REAL
runpy.run_path(_REAL, run_name="__main__")
