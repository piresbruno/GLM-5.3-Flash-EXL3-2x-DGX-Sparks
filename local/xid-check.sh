#!/usr/bin/env bash
# xid-check.sh — GPU Xid error monitor (R1 ops kit, ported from the
# Reederey87 production kit; see NOTICE). Timer-friendly: stateless run,
# one-shot scan, appends findings to logs/xid.log.
#
# Strategy: Xid events surface in the kernel ring buffer ("NVRM: Xid").
# The check keeps the previously seen Xid line count in a state file and
# alerts on NEW entries. Fatal classes on this kit (GB10 UMA): 13 (graphics
# engine exception), 31 (MMU fault — usually the OOM class), 43/45 (GPU
# stopped responding), 48 (double-bit ECC), 62 (UV health), 79 (GPU fell off
# the bus — the "both nodes frozen" signature).
#
# dmesg may be restricted (kernel.dmesg_restrict=1) — falls back to
# `journalctl -k`. Runs fine as a user timer; install with sudo-capable
# journal access for best coverage.
#
# Usage: local/xid-check.sh [--repo DIR]
set -euo pipefail
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SELF_DIR}/.." && pwd)"
LOGDIR="$REPO/logs"
LOGFILE="$LOGDIR/xid.log"
STATE="$LOGDIR/.xid-check.state"
mkdir -p "$LOGDIR"

FATAL_CLASSES="13|31|43|45|48|62|79"

now() { date '+%F %T'; }

# Collect Xid lines from whichever kernel log source is readable.
lines=""
if command -v journalctl >/dev/null 2>&1; then
    lines="$(journalctl -k --no-pager 2>/dev/null | grep -E 'NVRM: Xid' || true)"
fi
if [ -z "$lines" ]; then
    lines="$(dmesg 2>/dev/null | grep -E 'NVRM: Xid' || true)"
fi

count="$(printf '%s' "$lines" | grep -c . || true)"
prev="$(cat "$STATE" 2>/dev/null || echo 0)"

fatal_new="$(printf '%s\n' "$lines" | tail -n +"$((prev + 1))" 2>/dev/null | grep -E "Xid (${FATAL_CLASSES})" || true)"
any_new="$((count - prev))"

if [ "$any_new" -gt 0 ]; then
    {
        echo "$(now) Xid events: +${any_new} (total ${count})"
        printf '%s\n' "$lines" | tail -n +"$((prev + 1))" | tail -n 20
    } >> "$LOGFILE"
    if [ -n "$fatal_new" ]; then
        echo "$(now) ALERT: FATAL-CLASS Xid detected — snapshot fleet state and check dmesg:" >> "$LOGFILE"
        printf '%s\n' "$fatal_new" | tail -n 5 >> "$LOGFILE"
        echo "$(now) ALERT: fatal Xid (see $LOGFILE)" >&2
    fi
fi

printf '%s' "$count" > "$STATE"
exit 0
