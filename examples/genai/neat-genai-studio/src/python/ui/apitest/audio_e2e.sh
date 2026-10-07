#!/bin/bash
# End-to-end check of the Studio's audio workflow against a running instance
# (web or --backend-only): every step must produce useful output, not just 200.
#   1. /health reports the model server reachable and an ASR model active
#   2. /v1/audio/voices lists at least one server-side engine
#   3. /v1/audio/speech synthesizes an English sentence (RIFF WAV, > 1 s)
#   4. /v1/audio/transcriptions of that WAV contains the sentence's key words
#   5. /v1/audio/speech in German, then /v1/audio/translations returns English
#   6. /v1/chat/completions answers (only when a chat model is loaded)
#
# Usage: ./audio_e2e.sh [host:port | base URL]   (http://… when app.web.https is false)
# Exit status 0 when every step passes, 1 otherwise. Needs curl and python3.

if [ -n "$1" ] && [[ "$1" == http://* || "$1" == https://* ]]; then
  BASE="${1%/}"
elif [ -n "$1" ]; then
  BASE="${STUDIO_SCHEME:-https}://$1"
else
  BASE="${STUDIO_URL:-${STUDIO_SCHEME:-https}://${MODALIX_HOST:-127.0.0.1:5000}}"
  BASE="${BASE%/}"
fi

TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
PASS=0; FAIL=0
pass() { PASS=$((PASS + 1)); printf '  [PASS] %s\n' "$*"; }
fail() { FAIL=$((FAIL + 1)); printf '  [FAIL] %s\n' "$*"; }
json() { python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(eval(sys.argv[2]))" "$@" 2>/dev/null; }
contains_all() {   # file-with-text words... (case-insensitive)
  local text; text="$(tr '[:upper:]' '[:lower:]' < "$1")"; shift
  local w; for w in "$@"; do [[ "$text" == *"$w"* ]] || return 1; done
}

echo "Studio audio end-to-end: ${BASE}"

# 1. health
if curl -ksS --max-time 10 "${BASE}/health" -o "$TMP/health.json" \
   && [ "$(json "$TMP/health.json" "d['model_server']['reachable'] and bool(d['asr_model'])")" = "True" ]; then
  pass "health: model server reachable, ASR $(json "$TMP/health.json" "d['asr_model']")"
else
  fail "health: model server unreachable or no ASR model ($(head -c 200 "$TMP/health.json" 2>/dev/null))"
fi

# 2. voices
ENGINE=default
if curl -ksS --max-time 10 "${BASE}/v1/audio/voices" -o "$TMP/voices.json" \
   && [ "$(json "$TMP/voices.json" "len(d['engines'])")" -ge 1 ] 2>/dev/null; then
  keys="$(json "$TMP/voices.json" "' '.join(e['key'] for e in d['engines'])")"
  [[ " $keys " == *" supertonic "* ]] && ENGINE=supertonic
  pass "voices: ${keys} (using ${ENGINE})"
else
  fail "voices: no server-side engine listed"
fi

speak() {   # text language out.wav -> 0 when a real WAV of > 1 s came back
  local code dur
  code="$(curl -ksS --max-time 120 -X POST "${BASE}/v1/audio/speech" -H 'Content-Type: application/json' \
    -d "{\"input\":\"$1\",\"model\":\"${ENGINE}\",\"language\":\"$2\"}" -D "$3.h" -o "$3" -w '%{http_code}')"
  dur="$(tr -d '\r' < "$3.h" | awk -F': ' 'tolower($1)=="x-audio-duration"{print $2}')"
  [ "$code" = 200 ] && [ "$(head -c 4 "$3")" = RIFF ] && python3 -c "import sys; sys.exit(0 if float(sys.argv[1] or 0) > 1.0 else 1)" "$dur"
}

# 3 + 4. speech -> transcription
EN="Good morning, the weather is lovely today and the garden is full of flowers."
if speak "$EN" en "$TMP/en.wav"; then
  pass "speech (en): $(stat -c %s "$TMP/en.wav") bytes"
  curl -ksS --max-time 120 -X POST "${BASE}/v1/audio/transcriptions" -F "file=@$TMP/en.wav" -F response_format=text -o "$TMP/en.txt"
  if contains_all "$TMP/en.txt" morning weather garden; then
    pass "transcription: $(tr -d '\n' < "$TMP/en.txt")"
  else
    fail "transcription does not contain the spoken words: $(head -c 200 "$TMP/en.txt")"
  fi
else
  fail "speech (en): no usable WAV"
fi

# 5. German speech -> translation to English
DE="Guten Morgen, wie geht es dir heute?"
if speak "$DE" de "$TMP/de.wav"; then
  pass "speech (de): $(stat -c %s "$TMP/de.wav") bytes"
  curl -ksS --max-time 120 -X POST "${BASE}/v1/audio/translations" -F "file=@$TMP/de.wav" -F response_format=verbose_json -o "$TMP/tr.json"
  if [ "$(json "$TMP/tr.json" "d.get('task')")" = translate ] \
     && [ "$(json "$TMP/tr.json" "d.get('language')")" = de ] \
     && json "$TMP/tr.json" "d['text']" > "$TMP/tr.txt" && contains_all "$TMP/tr.txt" morning; then
    pass "translation (de -> en): $(cat "$TMP/tr.txt")"
  else
    fail "translation: $(head -c 300 "$TMP/tr.json")"
  fi
else
  fail "speech (de): no usable WAV (the engine may lack German)"
fi

# 6. chat, when a chat model is loaded
MODEL="$(json "$TMP/health.json" "(d['chat_models_loaded'] or [''])[0]")"
if [ -n "$MODEL" ]; then
  curl -ksS --max-time 120 -X POST "${BASE}/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"${MODEL}\",\"stream\":false,\"max_tokens\":32,\"temperature\":0,\"messages\":[{\"role\":\"user\",\"content\":\"Reply with the single word: ready\"}]}" \
    -o "$TMP/chat.json"
  answer="$(json "$TMP/chat.json" "d['choices'][0]['message']['content'].strip()")"
  if [ -n "$answer" ]; then pass "chat (${MODEL}): ${answer}"; else fail "chat: $(head -c 200 "$TMP/chat.json")"; fi
else
  printf '  [SKIP] chat: no chat model loaded\n'
fi

printf '%s passed, %s failed\n' "$PASS" "$FAIL"
[ "$FAIL" -eq 0 ]
