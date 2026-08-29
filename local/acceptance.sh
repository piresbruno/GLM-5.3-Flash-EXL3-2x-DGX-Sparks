#!/usr/bin/env bash
# acceptance.sh — 7-probe quality acceptance battery (R1 ops kit, ported from
# the Reederey87 production kit; see NOTICE).
#
# Probes (each prints PASS/FAIL; battery exits non-zero on any failure):
#   1. toolcall_battery  — tool-call battery (local/toolcall-probe.py, 1 rep)
#   2. thinking_default  — default request carries reasoning (thinking on)
#   3. thinking_off      — chat_template_kwargs override suppresses reasoning
#   4. vision            — an inline image turns into an image-aware answer
#   5. needle_16k        — fact retrieval from a ~16k-token haystack
#   6. needle_60k        — same at ~60k (R1-window probe; ACCEPTANCE_FULL=0 skips)
#   7. needle_16k_replay — the 16k needle asked twice; replay must answer
#                          identically AND report prefix-cache hits
#
# Usage: local/acceptance.sh [--base-url URL] [--needle-tokens N]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$SCRIPT_DIR")"

PORT_DEFAULT="$(grep -m1 -E '^[[:space:]]*PORT=' "$REPO/.env" 2>/dev/null | cut -d= -f2- | sed 's/[[:space:]]*#.*//; s/[[:space:]]*$//')"
PORT_DEFAULT="${PORT_DEFAULT:-8081}"
BASE_URL="http://127.0.0.1:${PORT_DEFAULT:-8081}"
NEEDLE_TOKENS=16000
while [ $# -gt 0 ]; do
    case "$1" in
        --base-url) BASE_URL="$2"; shift 2 ;;
        --needle-tokens) NEEDLE_TOKENS="$2"; shift 2 ;;
        *) echo "usage: $0 [--base-url URL] [--needle-tokens N]" >&2; exit 1 ;;
    esac
done
export GLM53_BENCH_BASE="$BASE_URL"

rc=0
probe() {  # probe <name> <cmd...> — flips rc on failure
    local name="$1"; shift
    if "$@"; then
        echo "PASS $name"
    else
        echo "FAIL $name" >&2
        rc=1
    fi
}

probe toolcall_battery python3 "$SCRIPT_DIR/toolcall-probe.py" --repeat 1 --min-pass 1

probe thinking_default python3 - "$BASE_URL" <<'EOF'
import json, sys, urllib.request
base = sys.argv[1]
payload = {"messages": [{"role": "user", "content": "What is 17*23? Think step by step."}],
           "max_tokens": 200, "temperature": 0}
req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(payload).encode(),
                             headers={"Content-Type": "application/json"})
resp = json.loads(urllib.request.urlopen(req, timeout=180).read())
msg = resp["choices"][0]["message"]
has_reasoning = bool(msg.get("reasoning_content") or msg.get("reasoning")
                     or "<think>" in str(msg.get("content", "")))
assert has_reasoning, f"no reasoning channel/content in default request: {str(msg)[:120]}"
assert "391" in str(msg.get("content", "")), "arithmetic answer missing from content"
EOF

probe thinking_off python3 - "$BASE_URL" <<'EOF'
import json, sys, urllib.request
base = sys.argv[1]
payload = {"messages": [{"role": "user", "content": "What is 17*23? Answer with the number only."}],
           "max_tokens": 16, "temperature": 0,
           "chat_template_kwargs": {"enable_thinking": False}}
req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(payload).encode(),
                             headers={"Content-Type": "application/json"})
resp = json.loads(urllib.request.urlopen(req, timeout=120).read())
msg = resp["choices"][0]["message"]
assert "391" in str(msg.get("content", "")), f"content missing answer: {str(msg)[:120]}"
assert not msg.get("reasoning_content") and "<think>" not in str(msg.get("content", "")), \
    "reasoning leaked with thinking off"
EOF

probe vision python3 - "$BASE_URL" <<'EOF'
import base64, json, sys, urllib.request
base = sys.argv[1]
# 2x2 PNG, solid red: a minimal real image; the model must acknowledge an image.
png = base64.b64encode(bytes.fromhex(
    "89504e470d0a1a0a0000000d4948445200000002000000020802000000fdd49a730000000c"
    "4944415408d763f8cfc0000003010100184d8e440000000049454e44ae426082"
)).decode()
payload = {"messages": [{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": "data:image/png;base64," + png}},
    {"type": "text", "text": "Do you see an image in this message? Answer yes or no."}]}],
    "max_tokens": 24, "temperature": 0,
    "chat_template_kwargs": {"enable_thinking": False}}
req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(payload).encode(),
                             headers={"Content-Type": "application/json"})
resp = json.loads(urllib.request.urlopen(req, timeout=180).read())
content = str(resp["choices"][0]["message"].get("content", "")).lower()
assert ("yes" in content or "red" in content or "image" in content), \
    f"vision answer unclear: {content[:120]}"
EOF

run_needle() {  # run_needle <tokens> <needle> <expected-substr> -> prints content
    python3 - "$BASE_URL" "$1" "$2" "$3" <<'EOF'
import json, sys, urllib.request
base, tokens, needle, expect = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
filler = "The logistics ledger lists crates, seals, and duty stamps transferred at the mid-way depot. "
half = max(1, int(tokens / 16 / 2))
prompt = filler * half + "\n" + needle + "\n" + filler * half
prompt += "\n\nQuestion: retrieve the exact fact planted above. Answer with only that fact."
payload = {"messages": [{"role": "user", "content": prompt}], "max_tokens": 32, "temperature": 0,
           "chat_template_kwargs": {"enable_thinking": False}}
req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(payload).encode(),
                             headers={"Content-Type": "application/json"})
resp = json.loads(urllib.request.urlopen(req, timeout=1200).read())
content = str(resp["choices"][0]["message"].get("content", ""))
assert expect in content, f"needle not retrieved: {content[:160]!r}"
print(content)
EOF
}

probe needle_16k run_needle "$NEEDLE_TOKENS" \
    "The emergency rendezvous code for convoy seven is AMBER-HERON-7741." \
    "AMBER-HERON-7741"

if [ "${ACCEPTANCE_FULL:-1}" = "1" ]; then
    probe needle_60k run_needle "$((NEEDLE_TOKENS * 4))" \
        "The secondary fallback frequency for the depot beacon is 118.62 megahertz." \
        "118.62"
else
    echo "SKIP needle_60k (ACCEPTANCE_FULL=0)"
fi

probe needle_16k_replay python3 - "$BASE_URL" "$NEEDLE_TOKENS" <<'EOF'
import json, sys, urllib.request
base, tokens = sys.argv[1], int(sys.argv[2])
needle = "The emergency rendezvous code for convoy seven is AMBER-HERON-7741."
filler = "The logistics ledger lists crates, seals, and duty stamps transferred at the mid-way depot. "
half = max(1, int(tokens / 16 / 2))
prompt = filler * half + "\n" + needle + "\n" + filler * half
prompt += "\n\nQuestion: retrieve the exact fact planted above. Answer with only that fact."
answers = []
cached = []
for _ in range(2):
    payload = {"messages": [{"role": "user", "content": prompt}], "max_tokens": 32,
               "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(base + "/v1/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    resp = json.loads(urllib.request.urlopen(req, timeout=1200).read())
    msg = resp["choices"][0]["message"]
    answers.append(str(msg.get("content", "")))
    usage = resp.get("usage", {})
    detail = (usage.get("prompt_tokens_details") or {}) if isinstance(usage, dict) else {}
    cached.append(int(detail.get("cached_tokens") or 0))
assert all("AMBER-HERON-7741" in a for a in answers), f"replay answers wrong: {answers!r}"
assert answers[0] == answers[1], f"replay answers differ: {answers!r}"
assert cached[1] > 0, f"replay reported no cached tokens: {cached}"
EOF

if [ "$rc" = "0" ]; then
    echo "acceptance: ALL PROBES PASS"
else
    echo "acceptance: FAILURES PRESENT — see the FAIL lines above" >&2
fi
exit "$rc"
