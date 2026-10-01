# Shared by setup.sh and run.sh (sourced, not executed): read values from a
# Studio config the way shared.config.load_ui_config does. Nesting follows the
# YAML indentation (any width), and like the loader the settings live under
# `app:` or, when there is no `app:` key, at the top level.

# yaml_lookup <config.yaml> <dotted.path> <scalar|list>
#   scalar: the value with surrounding quotes or a trailing comment removed.
#   list:   a block list ("- a" items) under the key, joined with commas.
yaml_lookup() {
  local config="$1" path="$2" mode="$3"
  [[ -f "${config}" ]] || return 0
  awk -v want="${path}" -v mode="${mode}" '
    function strip(v,   i) {
      sub(/^[ \t]+/, "", v)
      if (v ~ /^"/)            { v = substr(v, 2); i = index(v, "\""); if (i > 0) v = substr(v, 1, i - 1) }
      else if (v ~ /^\x27/)    { v = substr(v, 2); i = index(v, "\x27"); if (i > 0) v = substr(v, 1, i - 1) }
      else                     { sub(/[ \t]+#.*$/, "", v); sub(/[ \t]+$/, "", v) }
      return v
    }
    /^[ \t]*(#.*)?$/ { next }                       # blank lines and comments
    {
      match($0, /^ */); ind = RLENGTH; line = substr($0, ind + 1)
      if (collecting) {
        if (ind > list_ind && line ~ /^- /) {
          item = line; sub(/^- [ \t]*/, "", item); item = strip(item)
          out = out (out == "" ? "" : ",") item; next
        }
        print out; found = 1; exit
      }
      while (depth > 0 && ind <= at[depth]) depth--
      if (line ~ /^[A-Za-z_][A-Za-z0-9_-]*:([ \t]|$)/) {
        key = line; sub(/:.*/, "", key)
        val = line; sub(/^[^:]*:/, "", val)
        depth++; keys[depth] = key; at[depth] = ind
        p = keys[1]; for (i = 2; i <= depth; i++) p = p "." keys[i]
        if (p == want) {
          if (mode == "list") { collecting = 1; list_ind = ind; out = ""; next }
          print strip(val); found = 1; exit
        }
      }
    }
    END { if (collecting && !found) print out }
  ' "${config}" 2>/dev/null || true
}

# Authoritative path: ask the application's own loader (shared/config_query.py)
# with a Python that has PyYAML, so flow-style mappings, anchors and every other
# YAML form mean the same to the shell scripts as to the UI. The awk reader
# below is only the fallback when no such Python is available (for example
# before setup.sh has built the venv).
_CV_PYTHON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../python" 2>/dev/null && pwd)"
_studio_config_python() {
  local py
  for py in "${STUDIO_CONFIG_PYTHON:-}" "${_CV_PYTHON_DIR%/src/python}/.venv/bin/python" python3; do
    [[ -n "${py}" ]] && command -v "${py}" >/dev/null 2>&1 && { printf '%s' "${py}"; return 0; }
  done
  return 1
}
# studio_config_query <config.yaml> <field>   (fields: see config_query.py)
studio_config_query() {
  local py
  [[ -f "$1" && -n "${_CV_PYTHON_DIR}" ]] || return 1
  py="$(_studio_config_python)" || return 1
  "${py}" "${_CV_PYTHON_DIR}/shared/config_query.py" --config "$1" "$2" 2>/dev/null
}

# The loader reads `app:` when present, else the top level.
_studio_has_app_root() { grep -qE '^app:([[:space:]]|$)' "$1" 2>/dev/null; }
_studio_path() { if _studio_has_app_root "$1"; then printf 'app.%s' "$2"; else printf '%s' "$2"; fi; }

# supertonic_config_value <config.yaml> <key>   -> app.tts.supertonic.<key>
# (models_root and venv come from the loader, which also maps a legacy app_root;
# app_root itself is only needed by the fallback paths and is read directly)
supertonic_config_value() {
  case "$2" in
    models_root|venv) studio_config_query "$1" "supertonic.$2" && return 0 ;;
  esac
  yaml_lookup "$1" "$(_studio_path "$1" "tts.supertonic.$2")" scalar
}
# web_config_scalar <config.yaml> <key>         -> app.web.<key>
web_config_scalar() {
  studio_config_query "$1" "web.$2" && return 0
  local v; v="$(yaml_lookup "$1" "$(_studio_path "$1" "web.$2")" scalar)"
  [[ -n "${v}" ]] || v="$(yaml_lookup "$1" "$(_studio_path "$1" "web.$2")" list)"
  printf '%s\n' "${v}"
}
# web_config_list <config.yaml> <key>           -> app.web.<key> list, comma-joined
web_config_list() {
  studio_config_query "$1" "web.$2" && return 0
  yaml_lookup "$1" "$(_studio_path "$1" "web.$2")" list
}

# Truthiness as shared.config._load_bool: 1/true/yes/on in any case.
config_true() {
  case "$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}
