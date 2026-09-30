# Shared by setup.sh and run.sh (sourced, not executed): read one scalar from the
# app.tts.supertonic section of a Studio config written by setup.sh.
#   supertonic_config_value <config.yaml> <key>   -> value, or nothing
# Handles double/single-quoted values (a "#" inside quotes is kept) and plain
# scalars with a trailing comment. Missing file or key prints nothing.
supertonic_config_value() {
  local config="$1" key="$2"
  [[ -f "${config}" ]] || return 0
  awk -v key="${key}" '
    /^  tts:/ {tts=1; next}
    tts && /^  [a-z]/ {tts=0}
    tts && /^    supertonic:/ {st=1; next}
    tts && st && /^    [a-z]/ {st=0}
    tts && st && $1 == key":" {
      v = $0
      sub(/^[ \t]*[A-Za-z_]+:[ \t]*/, "", v)
      if (v ~ /^"/) {
        v = substr(v, 2); i = index(v, "\"")
        if (i > 0) v = substr(v, 1, i - 1)
      } else if (v ~ /^\x27/) {
        v = substr(v, 2); i = index(v, "\x27")
        if (i > 0) v = substr(v, 1, i - 1)
      } else {
        sub(/[ \t]+#.*$/, "", v)
        sub(/[ \t]+$/, "", v)
      }
      print v; exit
    }
  ' "${config}" 2>/dev/null || true
}
