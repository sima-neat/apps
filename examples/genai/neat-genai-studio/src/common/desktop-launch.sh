#!/usr/bin/env bash
# Started by the Neat GenAI Studio desktop icon (installed by setup.sh with
# CREATE_DESKTOP_ICON=1) inside a terminal window: starts the Studio, or finds
# it already running, and opens the web UI in the default browser once it
# answers. Closing the window or pressing Ctrl+C stops the Studio. If the
# Studio fails, the window stays open so the error can be read.
EXAMPLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${EXAMPLE_DIR}" || exit 1
"${EXAMPLE_DIR}/run.sh" --open-browser
status=$?
case "${status}" in
  0|129|130|143) ;;   # clean exit, window closed, Ctrl+C, stopped
  *)
    printf '\nNeat GenAI Studio exited with status %s.\n' "${status}"
    read -r -p "Press Enter to close this window… " _ || true ;;
esac
exit "${status}"
