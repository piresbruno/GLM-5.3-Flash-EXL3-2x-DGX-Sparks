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
           "max_tokens": 512, "temperature": 0}
req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(payload).encode(),
                             headers={"Content-Type": "application/json"})
resp = json.loads(urllib.request.urlopen(req, timeout=240).read())
msg = resp["choices"][0]["message"]
has_reasoning = bool(msg.get("reasoning_content") or msg.get("reasoning")
                     or "<think>" in str(msg.get("content", "")))
assert has_reasoning, f"no reasoning channel/content in default request: {str(msg)[:160]}"
assert "391" in str(msg.get("content", "")), f"arithmetic answer missing: {str(msg)[:160]}"
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
import json, sys, urllib.request, zlib, struct, base64

def red_square_png():  # 64x64 white with a centered 32x32 red square
    W = H = 64
    rows = b""
    for y in range(H):
        row = b"\x00"
        for x in range(W):
            row += b"\xe0\x10\x10" if (16 <= x < 48 and 16 <= y < 48) else b"\xff\xff\xff"
        rows += row
    def chunk(typ, data):
        c = struct.pack(">I", len(data)) + typ + data
        return c + struct.pack(">I", zlib.crc32(typ + data) & 0xFFFFFFFF)
    ihdr = struct.pack(">IIBBBBB", W, H, 8, 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", zlib.compress(rows, 9)) + chunk(b"IEND", b""))

base = sys.argv[1]
png = base64.b64encode(red_square_png()).decode()
payload = {"messages": [{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": "data:image/png;base64," + png}},
    {"type": "text", "text": "What color is the square in this image? Answer with one word."}]}],
    "max_tokens": 16, "temperature": 0,
    "chat_template_kwargs": {"enable_thinking": False}}
req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(payload).encode(),
                             headers={"Content-Type": "application/json"})
resp = json.loads(urllib.request.urlopen(req, timeout=180).read())
content = str(resp["choices"][0]["message"].get("content", "")).lower()
assert "red" in content, f"vision answer unclear: {content[:160]}"
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
HIT = "vllm:prefix_cache_hits_total"

def hits():
    t = 0.0
    for line in urllib.request.urlopen(base + "/metrics", timeout=30).read().decode().splitlines():
        if line.startswith(HIT + "{") or line.startswith(HIT + " "):
            t += float(line.rsplit("}", 1)[1].strip())
    return t

needle = "The emergency rendezvous code for convoy seven is AMBER-HERON-7741."
filler = "The logistics ledger lists crates, seals, and duty stamps transferred at the mid-way depot. "
half = max(1, int(tokens / 16 / 2))
prompt = filler * half + "\n" + needle + "\n" + filler * half
prompt += "\n\nQuestion: retrieve the exact fact planted above. Answer with only that fact."
answers = []
ratios = []
for attempt in range(2):
    h0 = hits()
    payload = {"messages": [{"role": "user", "content": prompt}], "max_tokens": 32,
               "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(base + "/v1/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    resp = json.loads(urllib.request.urlopen(req, timeout=1200).read())
    msg = resp["choices"][0]["message"]
    answers.append(str(msg.get("content", "")))
    prompt_tokens = resp.get("usage", {}).get("prompt_tokens", 0)
    ratios.append((hits() - h0) / prompt_tokens if prompt_tokens else 0.0)
assert all("AMBER-HERON-7741" in a for a in answers), f"replay answers wrong: {answers!r}"
assert answers[0] == answers[1], f"replay answers differ: {answers!r}"
assert ratios[1] >= 0.9, f"replay hit {ratios[1]:.3f} < 0.9 (measured via {HIT} delta)"
EOF

if [ "$rc" = "0" ]; then
    echo "acceptance: ALL PROBES PASS"
else
    echo "acceptance: FAILURES PRESENT — see the FAIL lines above" >&2
fi
exit "$rc"
