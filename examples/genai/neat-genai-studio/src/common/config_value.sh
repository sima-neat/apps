# Shared by setup.sh and run.sh (sourced, not executed): read scalars from a
# Studio config written by setup.sh the way shared.config reads them.
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

# One scalar from the app.web section:  web_config_scalar <config.yaml> <key>
# Quoted values lose their quotes, plain ones their trailing comment.
web_config_scalar() {
  local config="$1" key="$2"
  [[ -f "${config}" ]] || return 0
  awk -v key="${key}" '
    /^  web:/ {f=1; next}
    f && /^  [a-z]/ {f=0}
    f && $1 == key":" {
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

# Truthiness as shared.config._load_bool: 1/true/yes/on in any case.
config_true() {
  case "$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}
