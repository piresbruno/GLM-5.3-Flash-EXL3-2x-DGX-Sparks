#!/usr/bin/env bash
# metrics-alert.sh — spec-decode acceptance alerting (R1 ops kit, ported from
# the Reederey87 production kit; see NOTICE). Timer-friendly one-shot.
#
# Reads the vLLM Prometheus counters and alerts when the speculative-decoding
# acceptance rate collapses (a draft/verify pathology shows up here minutes
# before users feel it):
#   vllm:spec_decode_num_accepted_tokens_total
#   vllm:spec_decode_num_draft_tokens_total
# acceptance = accepted / draft (smoothed over the whole uptime counter).
#
# A threshold breach must persist CONSECUTIVE runs (state file) before the
# ALERT fires, so a single odd request cannot page anyone. Also flags NaN
# values in the counters (the known decode-pathology signature).
#
# Usage: local/metrics-alert.sh [--base-url URL] [--min-accept 0.80]
#        [--consecutive 3]
set -euo pipefail
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SELF_DIR}/.." && pwd)"
LOGDIR="$REPO/logs"
LOGFILE="$LOGDIR/metrics-alert.log"
STATE="$LOGDIR/.metrics-alert.state"
mkdir -p "$LOGDIR"

PORT_DEFAULT="8081"
# .env values may carry inline comments — strip them.
PORT_DEFAULT="$(grep -m1 -E '^[[:space:]]*PORT=' "$REPO/.env" 2>/dev/null | cut -d= -f2- | sed 's/[[:space:]]*#.*//; s/[[:space:]]*$//')"
PORT_DEFAULT="${PORT_DEFAULT:-8081}"
BASE_URL="http://127.0.0.1:${PORT_DEFAULT:-8081}"
MIN_ACCEPT="0.80"
CONSECUTIVE="3"
while [ $# -gt 0 ]; do
    case "$1" in
        --base-url) BASE_URL="$2"; shift 2 ;;
        --min-accept) MIN_ACCEPT="$2"; shift 2 ;;
        --consecutive) CONSECUTIVE="$2"; shift 2 ;;
        *) echo "usage: $0 [--base-url URL] [--min-accept R] [--consecutive N]" >&2; exit 1 ;;
    esac
done

now() { date '+%F %T'; }

metrics="$(curl -sf -m 15 "$BASE_URL/metrics" 2>/dev/null || true)"
if [ -z "$metrics" ]; then
    echo "$(now) metrics unreachable at $BASE_URL/metrics (server down?) — not counted as an acceptance breach" >> "$LOGFILE"
    exit 0
fi

val() {  # val <metric-name> — last sample of a counter
    printf '%s\n' "$metrics" | awk -v m="$1" '$1 ~ "^"m"{|^"m"_total{" {v=$NF} END {print v}'
}

accepted="$(val vllm:spec_decode_num_accepted_tokens)"
drafted="$(val vllm:spec_decode_num_draft_tokens)"
emitted="$(val vllm:spec_decode_num_emitted_tokens)"

if [ -z "$accepted" ] || [ -z "$drafted" ]; then
    echo "$(now) spec-decode counters absent (spec off? metrics renamed?) — nothing to alert on" >> "$LOGFILE"
    exit 0
fi

for v in "$accepted" "$drafted"; do
    case "$v" in *nan*|*NaN*) echo "$(now) ALERT: NaN in spec-decode counters (accepted=$accepted drafted=$drafted)" | tee -a "$LOGFILE" >&2; exit 0 ;; esac
done

rate="$(awk "BEGIN{ if ($drafted > 0) printf \"%.4f\", $accepted / $drafted; else print \"1.0000\" }")"
low="$(awk "BEGIN{exit !($rate < $MIN_ACCEPT)}" && echo yes || echo no)"

strikes=0
[ -f "$STATE" ] && strikes="$(cat "$STATE" 2>/dev/null || echo 0)"
if [ "$low" = "yes" ]; then
    strikes=$((strikes + 1))
    echo "$(now) acceptance ${rate} < ${MIN_ACCEPT} (strike ${strikes}/${CONSECUTIVE}; accepted=$accepted drafted=$drafted emitted=${emitted:-n/a})" >> "$LOGFILE"
    if [ "$strikes" -ge "$CONSECUTIVE" ]; then
        echo "$(now) ALERT: spec-decode acceptance ${rate} below ${MIN_ACCEPT} for ${strikes} consecutive checks" | tee -a "$LOGFILE" >&2
        strikes=0   # fire once per episode; re-arm after alerting
    fi
else
    if [ "$strikes" -gt 0 ]; then
        echo "$(now) acceptance recovered (${rate}) after ${strikes} strike(s)" >> "$LOGFILE"
    fi
    strikes=0
fi
printf '%s' "$strikes" > "$STATE"
exit 0
