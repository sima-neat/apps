#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# LLiMa backend API smoke test (direct inference backend port).
# Defaults match direct backend runtime on 9998, not web UI port 5000.
HOST="${LLIMA_HOST:-127.0.0.1}"
PORT="${LLIMA_PORT:-9998}"
SCHEME="${LLIMA_SCHEME:-http}"
BASE_URL="${SCHEME}://${HOST}:${PORT}"
UI_HOST="${LLIMA_UI_HOST:-${HOST}}"
UI_PORT="${LLIMA_UI_PORT:-5000}"
UI_SCHEME="${LLIMA_UI_SCHEME:-https}"
UI_BASE_URL="${UI_SCHEME}://${UI_HOST}:${UI_PORT}"
TIMEOUT_SECONDS="${LLIMA_TIMEOUT_SECONDS:-20}"
PROBE_TIMEOUT_SECONDS="${LLIMA_PROBE_TIMEOUT_SECONDS:-5}"
PROBE_REQUEST_TIMEOUT_SECONDS="${LLIMA_PROBE_REQUEST_TIMEOUT_SECONDS:-4}"
PERF_TIMEOUT_SECONDS="${LLIMA_PERF_TIMEOUT_SECONDS:-60}"
API_MODE="unknown"
API_MODE_OVERRIDE="${LLIMA_API_MODE:-openai}" # openai|direct|auto
VERBOSE="${LLIMA_TEST_VERBOSE:-1}"
LANG="${LLIMA_LANG:-en}"
TEST_AUDIO_FILE="${LLIMA_TEST_AUDIO_FILE:-$SCRIPT_DIR/test.wav}"
REQUIRE_AUDIO_TESTS="${LLIMA_REQUIRE_AUDIO_TESTS:-0}"
RESULTS=()
SUMMARY_PRINTED=0

log() {
  if [[ "${VERBOSE}" != "0" ]]; then
    printf "[%s] %s\n" "$(date '+%H:%M:%S')" "$*"
  fi
}

add_result() {
  local test_name="$1"
  local status="$2"   # PASS|WARN|FAIL
  local details="$3"
  RESULTS+=("${test_name}|${status}|${details}")
}

print_summary() {
  if [[ "${SUMMARY_PRINTED}" == "1" ]]; then
    return
  fi
  SUMMARY_PRINTED=1
  if (( ${#RESULTS[@]} == 0 )); then
    return
  fi

  echo ""
  echo "=============================="
  echo " Backend Runtime Test Summary "
  echo "=============================="
  local row name status details icon status_label
  local max_name=4  # len("Test")
  for row in "${RESULTS[@]}"; do
    IFS='|' read -r name status details <<< "$row"
    if (( ${#name} > max_name )); then
      max_name=${#name}
    fi
  done

  printf "%-${max_name}s  %-12s  %s\n" "Test" "Status" "Details"
  printf "%-${max_name}s  %-12s  %s\n" "$(printf '%*s' "$max_name" '' | tr ' ' '-')" "------------" "-------"

  for row in "${RESULTS[@]}"; do
    IFS='|' read -r name status details <<< "$row"
    case "$status" in
      PASS) icon="[OK]"; status_label="PASS" ;;
      WARN) icon="[!]"; status_label="WARN" ;;
      FAIL) icon="[X]"; status_label="FAIL" ;;
      *) icon="[?]"; status_label="$status" ;;
    esac
    printf "%-${max_name}s  %-12s  %s\n" "${name}" "${icon} ${status_label}" "${details}"
  done
  echo "=============================="
}

trap print_summary EXIT

warn_or_fail() {
  local message="$1"
  if [[ "${REQUIRE_AUDIO_TESTS}" == "1" ]]; then
    add_result "Audio Endpoint" "FAIL" "${message}"
    echo "❌ ${message}"
    exit 1
  fi
  add_result "Audio Endpoint" "WARN" "${message}"
  echo "⚠️  ${message}"
}

warn_or_fail_named() {
  local test_name="$1"
  local message="$2"
  if [[ "${REQUIRE_AUDIO_TESTS}" == "1" ]]; then
    add_result "${test_name}" "FAIL" "${message}"
    echo "❌ ${message}"
    exit 1
  fi
  add_result "${test_name}" "WARN" "${message}"
  echo "⚠️  ${message}"
}

echo "🔎 Testing LLiMa backend at ${BASE_URL}"
log "Probe timeout=${PROBE_TIMEOUT_SECONDS}s, probe request timeout=${PROBE_REQUEST_TIMEOUT_SECONDS}s, request timeout=${TIMEOUT_SECONDS}s"
log "API mode override=${API_MODE_OVERRIDE}"

wait_for_backend() {
  local max_attempts="${1:-30}"
  local attempt=1
  while (( attempt <= max_attempts )); do
    log "Readiness probe attempt ${attempt}/${max_attempts} ..."
    # First, confirm host:port is reachable at HTTP layer even if the model
    # is still warming up and inference endpoints are slow.
    root_code="$(curl -sS -o /dev/null -w "%{http_code}" --max-time "${PROBE_TIMEOUT_SECONDS}" "${BASE_URL}/" || true)"
    log "HTTP root probe status: ${root_code}"
    if [[ "$root_code" != "000" ]]; then
      if [[ "${API_MODE_OVERRIDE}" == "direct" ]]; then
        API_MODE="direct"
        log "Using forced API mode: direct"
        return 0
      elif [[ "${API_MODE_OVERRIDE}" == "openai" ]]; then
        API_MODE="openai"
        log "Using forced API mode: openai"
        return 0
      fi

      # Then probe supported API style.
      if curl -sS --max-time "${PROBE_REQUEST_TIMEOUT_SECONDS}" --fail \
        -H "Content-Type: application/json" \
        -X POST "${BASE_URL}/" \
        -d '{"text":"ping"}' >/dev/null 2>&1; then
        API_MODE="direct"
        log "Detected direct API mode via POST /"
        return 0
      fi
      if curl -sS --max-time "${PROBE_REQUEST_TIMEOUT_SECONDS}" --fail \
        -H "Content-Type: application/json" \
        -X POST "${BASE_URL}/v1/chat/completions" \
        -d '{"messages":[{"role":"user","content":"ping"}],"stream":false}' >/dev/null 2>&1; then
        API_MODE="openai"
        log "Detected OpenAI-compatible API mode via /v1/chat/completions"
        return 0
      fi

      # HTTP endpoint is up but inference endpoint may still be warming.
      # Continue probing instead of failing early.
      log "Backend reachable but inference endpoint not ready yet."
      sleep 1
      ((attempt++))
      continue
    fi

    # Fallback to direct endpoint probes for non-standard setups.
    if curl -sS --max-time "${PROBE_REQUEST_TIMEOUT_SECONDS}" --fail \
      -H "Content-Type: application/json" \
      -X POST "${BASE_URL}/" \
      -d '{"text":"ping"}' >/dev/null 2>&1; then
      API_MODE="direct"
      log "Detected direct API mode via fallback probe."
      return 0
    fi
    if curl -sS --max-time "${PROBE_REQUEST_TIMEOUT_SECONDS}" --fail \
      -H "Content-Type: application/json" \
      -X POST "${BASE_URL}/v1/chat/completions" \
      -d '{"messages":[{"role":"user","content":"ping"}],"stream":false}' >/dev/null 2>&1; then
      API_MODE="openai"
      log "Detected OpenAI-compatible mode via fallback probe."
      return 0
    fi
    log "No endpoint responded yet; retrying."
    sleep 1
    ((attempt++))
  done
  return 1
}

if ! wait_for_backend 30; then
  add_result "Readiness" "FAIL" "Backend not reachable at ${BASE_URL}"
  echo "❌ Backend is not reachable at ${BASE_URL}"
  echo "   Debug: try 'curl -v ${BASE_URL}/' and verify routing/firewall from this host."
  echo "   Start backend first, for example: ./run.sh --backend-only"
  exit 1
fi

echo "✅ Backend is reachable"
add_result "Readiness" "PASS" "Backend reachable at ${BASE_URL} (mode=${API_MODE})"
echo "ℹ️  Detected API mode: ${API_MODE}"

if [[ "${API_MODE}" == "direct" ]]; then
  echo "🧪 Test 1: direct inference POST / (port ${PORT})"
  direct_response="$(
    curl -sS --fail --max-time "${TIMEOUT_SECONDS}" \
      -H "Content-Type: application/json" \
      -X POST "${BASE_URL}/" \
      -d @- <<EOF
{
  "text": "Reply with one short sentence."
}
EOF
  )"
  log "Direct response sample: ${direct_response:0:180}"

  python3 - "$direct_response" <<'PY'
import sys

raw = sys.argv[1]
if not raw or not raw.strip():
    raise SystemExit("Empty direct inference response")
print("✅ Direct inference response validated")
PY
  add_result "Chat Completion" "PASS" "Direct inference endpoint responded"
else
  echo "🧪 Test 1: OpenAI-compatible /v1/chat/completions (streaming)"
  openai_stream="$(
    curl -sS --fail --max-time "${TIMEOUT_SECONDS}" \
      -N \
      -H "Content-Type: application/json" \
      -X POST "${BASE_URL}/v1/chat/completions" \
      -d '{"messages":[{"role":"user","content":"Reply with one short sentence."}],"stream":true}'
  )"
  log "OpenAI stream sample: ${openai_stream:0:220}"
  if [[ "$openai_stream" != *"data:"* ]]; then
    add_result "Chat Completion" "FAIL" "OpenAI stream missing SSE data lines"
    echo "❌ OpenAI streaming response missing SSE data lines."
    exit 1
  fi
  if [[ "$openai_stream" != *"[DONE]"* ]]; then
    add_result "Chat Completion" "WARN" "OpenAI stream missing [DONE] marker"
    echo "⚠️  OpenAI streaming response missing [DONE] marker (continuing)."
  else
    add_result "Chat Completion" "PASS" "OpenAI stream returned SSE chunks + [DONE]"
    echo "✅ OpenAI streaming response validated"
  fi
fi

echo "📊 Test 1b: client-side performance (single prompt)"
if [[ "${API_MODE}" == "openai" ]]; then
  perf_start_ns="$(date +%s%N)"
  perf_stream="$(
    curl -sS --fail --max-time "${PERF_TIMEOUT_SECONDS}" \
      -N \
      -H "Content-Type: application/json" \
      -X POST "${BASE_URL}/v1/chat/completions" \
      -d '{"messages":[{"role":"user","content":"Tell me a short story in about 120 words."}],"stream":true}'
  )"
  perf_end_ns="$(date +%s%N)"

  python3 - "$perf_stream" "$perf_start_ns" "$perf_end_ns" <<'PY'
import json
import re
import sys

stream = sys.argv[1]
start_ns = int(sys.argv[2])
end_ns = int(sys.argv[3])
elapsed = max((end_ns - start_ns) / 1e9, 1e-9)

chunks = 0
text_parts = []
for raw in stream.splitlines():
    if not raw.startswith("data:"):
        continue
    payload = raw[len("data:"):].strip()
    if payload == "[DONE]":
        continue
    try:
        obj = json.loads(payload)
    except Exception:
        continue
    choices = obj.get("choices") or []
    if not choices:
        continue
    delta = choices[0].get("delta") or {}
    content = delta.get("content")
    if content:
        chunks += 1
        text_parts.append(content)

text = "".join(text_parts)
tokens = len(re.findall(r"\S+", text))
tps = tokens / elapsed if elapsed > 0 else 0.0
print(f"⏱️  Elapsed: {elapsed:.2f}s")
print(f"🔤 Output chars: {len(text)}")
print(f"🧮 Estimated output tokens: {tokens}")
print(f"🚀 Client-side TPS (estimated): {tps:.2f}")
print(f"🧩 Stream chunks with text: {chunks}")
PY
  add_result "Performance" "PASS" "Client-side TPS measured in streaming mode"
else
  perf_start_ns="$(date +%s%N)"
  perf_response="$(
    curl -sS --fail --max-time "${PERF_TIMEOUT_SECONDS}" \
      -H "Content-Type: application/json" \
      -X POST "${BASE_URL}/" \
      -d '{"text":"Tell me a short story in about 120 words."}'
  )"
  perf_end_ns="$(date +%s%N)"

  python3 - "$perf_response" "$perf_start_ns" "$perf_end_ns" <<'PY'
import json
import re
import sys

raw = sys.argv[1]
start_ns = int(sys.argv[2])
end_ns = int(sys.argv[3])
elapsed = max((end_ns - start_ns) / 1e9, 1e-9)

text = raw.strip()
try:
    obj = json.loads(raw)
    if isinstance(obj, dict):
        text = str(obj.get("text") or obj.get("response") or obj.get("output") or raw).strip()
except Exception:
    pass

tokens = len(re.findall(r"\S+", text))
tps = tokens / elapsed if elapsed > 0 else 0.0
print(f"⏱️  Elapsed: {elapsed:.2f}s")
print(f"🔤 Output chars: {len(text)}")
print(f"🧮 Estimated output tokens: {tokens}")
print(f"🚀 Client-side TPS (estimated): {tps:.2f}")
PY
  add_result "Performance" "PASS" "Client-side TPS measured in direct mode"
fi

echo "🎙️  Test 1c: /v1/audio/transcriptions"
if [[ ! -f "${TEST_AUDIO_FILE}" ]]; then
  warn_or_fail "Transcription test file not found: ${TEST_AUDIO_FILE}"
else
  trans_status="$(
    curl -sS -o /tmp/llima_transcribe.out -w "%{http_code}" \
      --max-time "${TIMEOUT_SECONDS}" \
      -X POST "${BASE_URL}/v1/audio/transcriptions" \
      -H "Content-Type: multipart/form-data" \
      -F "file=@${TEST_AUDIO_FILE}" \
      -F "language=${LANG}" || true
  )"
  log "/v1/audio/transcriptions status: ${trans_status}"
  if [[ "${trans_status}" != "200" ]]; then
    warn_or_fail "/v1/audio/transcriptions failed (HTTP ${trans_status})"
  else
    python3 - <<'PY'
import json
from pathlib import Path
raw = Path("/tmp/llima_transcribe.out").read_text(errors="ignore").strip()
if not raw:
    raise SystemExit("Empty transcription response")
try:
    data = json.loads(raw)
except Exception as e:
    raise SystemExit(f"Non-JSON transcription response: {e}")
text = data.get("text")
if text is None:
    # Accept alternative payloads but require at least one value present.
    if not isinstance(data, dict) or len(data) == 0:
        raise SystemExit("Transcription JSON had no usable fields")
print("✅ /v1/audio/transcriptions returned JSON payload")
PY
    add_result "Audio Transcription" "PASS" "/v1/audio/transcriptions returned valid JSON"
  fi
fi

echo "🎙️  Test 1d: TTS APIs on ${UI_BASE_URL} (port ${UI_PORT})"
ui_root_status="$(
  curl -k -sS -o /dev/null -w "%{http_code}" \
    --max-time "${PROBE_TIMEOUT_SECONDS}" \
    "${UI_BASE_URL}/" || true
)"
log "UI root probe status on ${UI_BASE_URL}: ${ui_root_status}"
if [[ "${ui_root_status}" == "000" ]]; then
  add_result "UI Port ${UI_PORT}" "WARN" "UI/TTS port not reachable, skipped /voices,/voices/select,/v1/audio/speech"
  echo "ℹ️  Port ${UI_PORT} is not reachable on ${UI_HOST}; skipping UI/TTS endpoint tests."
else
  add_result "UI Port ${UI_PORT}" "PASS" "UI/TTS port reachable at ${UI_BASE_URL}"

  voices_status="$(
    curl -k -sS -o /tmp/llima_ui_voices.out -w "%{http_code}" \
      --max-time "${TIMEOUT_SECONDS}" \
      "${UI_BASE_URL}/voices?lang=${LANG}" || true
  )"
  log "/voices status on ${UI_BASE_URL}: ${voices_status}"
  if [[ "${voices_status}" != "200" ]]; then
    warn_or_fail_named "TTS Voices" "/voices failed on ${UI_BASE_URL} (HTTP ${voices_status})"
  else
    voice_id="$(
      python3 - <<'PY'
import json
from pathlib import Path
raw = Path("/tmp/llima_ui_voices.out").read_text(errors="ignore").strip()
if not raw:
    raise SystemExit(0)
try:
    data = json.loads(raw)
except Exception:
    raise SystemExit(0)

def pick(node):
    if isinstance(node, dict):
        for k in ("voiceId", "voice_id", "id", "name"):
            v = node.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        for k in ("voices", "results", "data", "items"):
            if k in node:
                got = pick(node[k])
                if got:
                    return got
        for v in node.values():
            got = pick(v)
            if got:
                return got
    elif isinstance(node, list):
        for item in node:
            got = pick(item)
            if got:
                return got
    return ""

print(pick(data))
PY
    )"
    if [[ -z "${voice_id}" ]]; then
      warn_or_fail_named "TTS Voices" "No usable voiceId from /voices response"
    else
      add_result "TTS Voices" "PASS" "Retrieved voiceId=${voice_id}"

      select_status="$(
        curl -k -sS -o /tmp/llima_ui_voice_select.out -w "%{http_code}" \
          --max-time "${TIMEOUT_SECONDS}" \
          -H "Content-Type: application/json" \
          -X POST "${UI_BASE_URL}/voices/select" \
          -d "{\"voiceId\":\"${voice_id}\",\"lang\":\"${LANG}\"}" || true
      )"
      log "/voices/select status on ${UI_BASE_URL}: ${select_status}"
      if [[ "${select_status}" == "200" || "${select_status}" == "204" ]]; then
        add_result "TTS Voice Select" "PASS" "/voices/select returned HTTP ${select_status}"
      else
        warn_or_fail_named "TTS Voice Select" "/voices/select failed on ${UI_BASE_URL} (HTTP ${select_status})"
      fi
    fi
  fi

  speech_status="$(
    curl -k -sS -o /tmp/llima_ui_tts.wav -w "%{http_code}" \
      --max-time "${TIMEOUT_SECONDS}" \
      -H "Content-Type: application/json" \
      -X POST "${UI_BASE_URL}/v1/audio/speech" \
      -d "{\"input\":\"Hello from automated UI TTS test.\",\"language\":\"${LANG}\",\"response_format\":\"wav\"}" || true
  )"
  log "/v1/audio/speech status on ${UI_BASE_URL}: ${speech_status}"
  if [[ "${speech_status}" != "200" ]]; then
    warn_or_fail_named "TTS Speech" "/v1/audio/speech failed on ${UI_BASE_URL} (HTTP ${speech_status})"
  elif [[ ! -s /tmp/llima_ui_tts.wav ]]; then
    warn_or_fail_named "TTS Speech" "/v1/audio/speech returned empty audio payload"
  else
    speech_size="$(wc -c < /tmp/llima_ui_tts.wav | tr -d ' ')"
    add_result "TTS Speech" "PASS" "/v1/audio/speech returned ${speech_size} bytes"
  fi
fi

echo "🧪 Test 2: /stop endpoint on direct backend"
stop_status="$(
  curl -sS -o /tmp/llima_stop.out -w "%{http_code}" \
    --max-time "${TIMEOUT_SECONDS}" \
    -X POST "${BASE_URL}/stop" || true
)"
log "/stop HTTP status: ${stop_status}"
if [[ "$stop_status" == "200" || "$stop_status" == "204" ]]; then
  add_result "Stop Endpoint" "PASS" "/stop returned HTTP ${stop_status}"
  echo "✅ /stop succeeded (HTTP ${stop_status})"
else
  add_result "Stop Endpoint" "WARN" "/stop returned HTTP ${stop_status}"
  echo "⚠️  /stop returned HTTP ${stop_status} (continuing)"
fi

echo "🧪 Test 3: backend responsiveness after /stop"
if [[ "${API_MODE}" == "direct" ]]; then
  post_stop_status="$(
    curl -sS -o /tmp/llima_post_stop.out -w "%{http_code}" \
      --max-time "${TIMEOUT_SECONDS}" \
      -H "Content-Type: application/json" \
      -X POST "${BASE_URL}/" \
      -d '{"text":"hello again"}' || true
  )"
else
  post_stop_status="$(
    curl -sS -o /tmp/llima_post_stop.out -w "%{http_code}" \
      --max-time "${TIMEOUT_SECONDS}" \
      -H "Content-Type: application/json" \
      -X POST "${BASE_URL}/v1/chat/completions" \
      -d '{"messages":[{"role":"user","content":"hello again"}],"stream":true}' || true
  )"
fi
log "Post-/stop HTTP status: ${post_stop_status}"
if [[ "$post_stop_status" == "200" ]]; then
  add_result "Post-Stop Responsiveness" "PASS" "Backend responded after /stop"
  echo "✅ Backend still responsive after /stop"
else
  add_result "Post-Stop Responsiveness" "WARN" "HTTP ${post_stop_status} after /stop"
  echo "ℹ️  Backend returned HTTP ${post_stop_status} after /stop (acceptable if service is stopping)"
fi

echo "🎉 Backend runtime tests completed for ${BASE_URL}"
