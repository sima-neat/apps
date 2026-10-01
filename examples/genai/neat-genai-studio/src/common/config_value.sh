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

# The loader reads `app:` when present, else the top level.
_studio_has_app_root() { grep -qE '^app:([[:space:]]|$)' "$1" 2>/dev/null; }
_studio_path() { if _studio_has_app_root "$1"; then printf 'app.%s' "$2"; else printf '%s' "$2"; fi; }

# supertonic_config_value <config.yaml> <key>   -> app.tts.supertonic.<key>
supertonic_config_value() { yaml_lookup "$1" "$(_studio_path "$1" "tts.supertonic.$2")" scalar; }
# web_config_scalar <config.yaml> <key>         -> app.web.<key>
web_config_scalar() { yaml_lookup "$1" "$(_studio_path "$1" "web.$2")" scalar; }
# web_config_list <config.yaml> <key>           -> app.web.<key> as a block list, comma-joined
web_config_list() { yaml_lookup "$1" "$(_studio_path "$1" "web.$2")" list; }

# Truthiness as shared.config._load_bool: 1/true/yes/on in any case.
config_true() {
  case "$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}
