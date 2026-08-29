#!/usr/bin/env bash
# serving-probe.sh — 6-probe serving liveness battery (R1 ops kit, ported from
# the Reederey87 production kit; see NOTICE).
#
# Probes (all functional — a quality battery lives in local/acceptance.sh):
#   1. health      — GET /health is 200
#   2. models      — GET /v1/models lists the served model
#   3. chat        — a minimal chat completion returns content
#   4. streaming   — an SSE stream yields at least one chunk + [DONE]
#   5. tools       — a tool-calling request returns a tool_calls message
#   6. metrics     — GET /metrics exposes vLLM counters
#
# Usage: local/serving-probe.sh [--base-url URL]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$SCRIPT_DIR")"

PORT_DEFAULT="$(grep -m1 -E '^[[:space:]]*PORT=' "$REPO/.env" 2>/dev/null | cut -d= -f2- | sed 's/[[:space:]]*#.*//; s/[[:space:]]*$//')"
PORT_DEFAULT="${PORT_DEFAULT:-8081}"
BASE_URL="http://127.0.0.1:${PORT_DEFAULT:-8081}"
MODEL="${GLM53_BENCH_MODEL:-GLM-5.3-Flash-EXL3}"
MODEL="$(grep -m1 -E '^[[:space:]]*SERVED_MODEL_NAME=' "$REPO/.env" 2>/dev/null | cut -d= -f2- | sed 's/[[:space:]]*#.*//; s/[[:space:]]*$//')"
MODEL="${MODEL:-GLM-5.3-Flash-EXL3}"
while [ $# -gt 0 ]; do
    case "$1" in
        --base-url) BASE_URL="$2"; shift 2 ;;
        *) echo "usage: $0 [--base-url URL]" >&2; exit 1 ;;
    esac
done

rc=0
probe() {
    local name="$1"; shift
    if "$@"; then echo "PASS $name"; else echo "FAIL $name" >&2; rc=1; fi
}

probe health curl -sf -m 10 "$BASE_URL/health"

probe models python3 - "$BASE_URL" "$MODEL" <<'EOF'
import json, sys, urllib.request
base, model = sys.argv[1], sys.argv[2]
data = json.loads(urllib.request.urlopen(base + "/v1/models", timeout=15).read())
ids = [m.get("id") for m in data.get("data", [])]
assert model in ids, f"model {model!r} not in {ids}"
EOF

probe chat python3 - "$BASE_URL" "$MODEL" <<'EOF'
import json, sys, urllib.request
base, model = sys.argv[1], sys.argv[2]
payload = {"model": model, "messages": [{"role": "user", "content": "Reply with the single word: ok"}],
           "max_tokens": 8, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(payload).encode(),
                             headers={"Content-Type": "application/json"})
resp = json.loads(urllib.request.urlopen(req, timeout=120).read())
assert str(resp["choices"][0]["message"].get("content", "")).strip(), "empty content"
EOF

probe streaming python3 - "$BASE_URL" "$MODEL" <<'EOF'
import json, sys, urllib.request
base, model = sys.argv[1], sys.argv[2]
payload = {"model": model, "messages": [{"role": "user", "content": "Count 1 to 5, digits only."}],
           "max_tokens": 24, "temperature": 0, "stream": True,
           "chat_template_kwargs": {"enable_thinking": False}}
req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(payload).encode(),
                             headers={"Content-Type": "application/json"})
chunks = 0
done = False
with urllib.request.urlopen(req, timeout=120) as resp:
    for raw in resp:
        line = raw.decode("utf-8", "replace").strip()
        if line == "data: [DONE]":
            done = True
        elif line.startswith("data: "):
            chunks += 1
assert chunks > 0, "no SSE chunks"
assert done, "stream never sent [DONE]"
EOF

probe tools python3 - "$BASE_URL" "$MODEL" <<'EOF'
import json, sys, urllib.request
base, model = sys.argv[1], sys.argv[2]
payload = {"model": model,
           "messages": [{"role": "user", "content": "What is the weather in Lisbon? Use the tool."}],
           "tools": [{"type": "function", "function": {"name": "get_weather",
                      "description": "Get weather.", "parameters": {"type": "object",
                      "properties": {"city": {"type": "string"}}, "required": ["city"]}}}],
           "max_tokens": 64, "temperature": 0,
           "chat_template_kwargs": {"enable_thinking": False}}
req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(payload).encode(),
                             headers={"Content-Type": "application/json"})
resp = json.loads(urllib.request.urlopen(req, timeout=120).read())
msg = resp["choices"][0]["message"]
assert msg.get("tool_calls"), f"no tool_calls: {str(msg)[:120]}"
EOF

probe metrics curl -sf -m 10 "$BASE_URL/metrics" -o /dev/null

if [ "$rc" = "0" ]; then
    echo "serving-probe: ALL 6 PROBES PASS"
else
    echo "serving-probe: FAILURES PRESENT" >&2
fi
exit "$rc"
