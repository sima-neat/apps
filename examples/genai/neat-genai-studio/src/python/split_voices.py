#!/usr/bin/env python3
"""Compatibility shim — temporary, remove one release after the ui/tts/ move.

The voice splitter moved to ``ui/tts/split_voices.py``. A customer running
``run.sh update`` from a pre-move release executes the *old* in-memory
``migrate_piper_voices`` function, which still invokes this path — and by then
the pull has relocated the real script. Forward to the new location so that one
in-flight update still succeeds. Fresh installs and later updates call the new
path directly and never hit this shim.
"""

import os
import runpy
import sys

_REAL = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "ui", "tts", "split_voices.py"
)

if not os.path.isfile(_REAL):
    sys.stderr.write(
        "split_voices.py moved to ui/tts/split_voices.py and the target is "
        "missing; re-run setup.sh.\n"
    )
    raise SystemExit(1)

# Run the relocated script as __main__ with the same arguments (argv[1] is the
# assets dir passed by migrate_piper_voices).
sys.argv[0] = _REAL
runpy.run_path(_REAL, run_name="__main__")
