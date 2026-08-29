#!/usr/bin/env bash
# memfloor.sh — measure the memory floor of a workload on BOTH nodes and
# write the floor artifact, per the memory-floor methodology (D2, ported
# from Entrpi's memlog.sh).
#
# Usage:
#   tools/memfloor.sh <label> [-- <workload command...>]
#
# - Starts a 1 Hz MemAvailable sampler on the head and the worker (worker via
#   $WORKER_SSH from .env).
# - Runs the workload command (optional). If none is given, runs for
#   MEMFLOOR_SECONDS (default 300) or until Ctrl-C.
# - Writes results/ab/memfloor-<label>-<stamp>/floor.json + summary.md with
#   the minimum floor per node, and warns when the binding floor < 5 GiB
#   (the crash-review safety margin on this kit).
#
# The result belongs in the arm's arm.json (env_diff / metrics) for any
# KV_CACHE_MEMORY / GPU_MEM_UTIL / MAX_NUM_BATCHED_TOKENS-changing arm.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LABEL="${1:?usage: memfloor.sh <label> [-- <workload...>]}"
shift || true
[ "${1:-}" = "--" ] && shift || true

set -a
# shellcheck disable=SC1091
source "$SCRIPT_DIR/.env"
set +a
WORKER_USER="${WORKER_USER:-$USER}"
[ "$WORKER_USER" = "$USER" ] && WORKER_HOME="${WORKER_HOME:-$HOME}" || WORKER_HOME="${WORKER_HOME:-/home/${WORKER_USER}}"
WORKER_SSH="${WORKER_SSH:-${WORKER_USER}@${WORKER_IP}}"
PORT="${PORT:-8081}"
SECONDS_OVERRIDE="${MEMFLOOR_SECONDS:-300}"

STAMP="$(date +%Y%m%d-%H%M)"
OUT="$SCRIPT_DIR/results/ab/memfloor-${LABEL}-${STAMP}"
mkdir -p "$OUT"
HEAD_LOG="$OUT/head.memlog"
WORKER_LOG="$OUT/worker.memlog"

worker_ssh() { ssh -o BatchMode=yes -o ConnectTimeout=10 "$WORKER_SSH" "$@"; }

echo "memfloor '$LABEL': sampling head + worker into $OUT"
[ -x "$SCRIPT_DIR/tools/memlog.sh" ] || chmod +x "$SCRIPT_DIR/tools/memlog.sh"

# sampler on the worker (needs the script there) and locally
worker_ssh "mkdir -p /tmp/glm53-memfloor" 2>/dev/null || true
scp -q -o BatchMode=yes "$SCRIPT_DIR/tools/memlog.sh" "${WORKER_SSH}:/tmp/glm53-memfloor/memlog.sh"
worker_ssh "nohup bash /tmp/glm53-memfloor/memlog.sh '$WORKER_LOG' '$WORKER_IP' >/dev/null 2>&1 & echo worker sampler up" >/dev/null
bash "$SCRIPT_DIR/tools/memlog.sh" "$HEAD_LOG" "${HEAD_IP:-$(hostname)}" &
HEAD_PID=$!
trap 'kill $HEAD_PID 2>/dev/null || true; worker_ssh "pkill -f glm53-memfloor/memlog.sh" 2>/dev/null || true' EXIT

if [ "$#" -gt 0 ]; then
    echo "memfloor: running workload: $*"
    "$@"
else
    echo "memfloor: no workload given — sampling for ${SECONDS_OVERRIDE}s"
    sleep "$SECONDS_OVERRIDE"
fi

sleep 2
kill "$HEAD_PID" 2>/dev/null || true
wait "$HEAD_PID" 2>/dev/null || true
worker_ssh "pkill -f glm53-memfloor/memlog.sh" 2>/dev/null || true
sleep 1

head_floor="$(sort -n "$HEAD_LOG" | awk 'NR==1{print $2; exit}')"
worker_floor="$(sort -n "$WORKER_LOG" | awk 'NR==1{print $2; exit}')"
[ -n "$head_floor" ] && head_gib=$(awk -v k="$head_floor" 'BEGIN{printf "%.2f", k/1048576}')
[ -n "$worker_floor" ] && worker_gib=$(awk -v k="$worker_floor" 'BEGIN{printf "%.2f", k/1048576}')

binding=""
[ -n "$head_floor" ] && [ -n "$worker_floor" ] && {
  if [ "$head_floor" -le "$worker_floor" ]; then binding=head; else binding=worker; fi
}
cat > "$OUT/floor.json" <<EOF
{
  "label": "$LABEL",
  "stamp": "$STAMP",
  "head_floor_kib": ${head_floor:-null},
  "worker_floor_kib": ${worker_floor:-null},
  "binding_node": "${binding:-null}",
  "workload": [$(printf '"%s",' "$@" | sed 's/,$//')],
  "warning": "floors age ~1.5-2 GiB/day; fresh-boot floors are optimistic"
}
EOF

{
  echo "# memfloor ${LABEL} (${STAMP})"
  echo
  echo "| Node | Floor (KiB) | Floor (GiB) |"
  echo "|---|---:|---:|"
  echo "| head (${HEAD_IP:-?}) | ${head_floor:-n/a} | ${head_gib:-n/a} |"
  echo "| worker (${WORKER_IP}) | ${worker_floor:-n/a} | ${worker_gib:-n/a} |"
  echo
  if [ -n "$binding" ]; then
    echo "Binding node: **$binding** (min floor)."
    floor_kib="$([ "$binding" = head ] && echo "$head_floor" || echo "$worker_floor")"
    if [ -n "$floor_kib" ] && [ "$floor_kib" -lt 5242880 ]; then
      echo "**WARN: floor < 5 GiB (5242880 KiB). Do not raise KV_CACHE_MEMORY / GPU_MEM_UTIL on this state** (crash review 2026-08-29: <5 GiB risk band)."
    else
      echo "Floor is inside the safe band (>= 5 GiB)."
    fi
  else
    echo "Floor missing on one or both nodes — sampler failed?"
  fi
  echo
  echo "Workload: $*"
} > "$OUT/summary.md"

echo "---- memfloor summary ($OUT/summary.md) ----"
cat "$OUT/summary.md"
