#!/usr/bin/env bash
# cache-probe.sh — operator gate wrapper for the R1 cache probes.
#
# Runs the two multi-session cache protocols from local/cache-burst.py and
# applies the R1 bundle gates (see docs/CAMPAIGN-R1.md):
#   burst : 4 sessions x ~60k tokens x 3 rounds — rounds 2-3 hit >= 90%
#   solo  : ~110k-token prompt replayed once — replay hit >= 93%
#
# Usage:
#   local/cache-probe.sh [--out results/ab/<arm>/cache.json] [--skip-solo]
# Exit 0 only when every requested gate passes.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT=""
SKIP_SOLO=0
while [ $# -gt 0 ]; do
    case "$1" in
        --out) OUT="${2:?}"; shift 2 ;;
        --skip-solo) SKIP_SOLO=1; shift ;;
        *) echo "usage: $0 [--out FILE] [--skip-solo]" >&2; exit 1 ;;
    esac
done

rc=0
ARGS=(--protocol burst --gate)
if [ -n "$OUT" ]; then ARGS+=(--out "$OUT"); fi
python3 "$SCRIPT_DIR/cache-burst.py" "${ARGS[@]}" || rc=1

if [ "$SKIP_SOLO" = "0" ]; then
    ARGS=(--protocol solo --gate)
    if [ -n "$OUT" ]; then ARGS+=(--out "${OUT%.json}-solo.json"); fi
    python3 "$SCRIPT_DIR/cache-burst.py" "${ARGS[@]}" || rc=1
fi

if [ "$rc" = "0" ]; then
    echo "cache-probe: ALL GATES PASS"
else
    echo "cache-probe: GATE FAILURE — do not record this arm as cache-clean" >&2
fi
exit "$rc"
