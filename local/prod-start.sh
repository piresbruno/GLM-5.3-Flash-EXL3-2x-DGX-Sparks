#!/usr/bin/env bash
# prod-start.sh — memory-gated, config-shaped production restart (R1 ops kit,
# ported from the Reederey87 production kit; see NOTICE).
#
# Before touching the fleet it:
#   1. MEMORY GATE — requires MemFree >= PROD_MIN_MEMFREE_KB (default 8 GiB,
#      the crash-review floor) on BOTH nodes. A restart that boots weights
#      with a collapsed MemFree is the UVM-livelock dice roll; refuse loudly.
#   2. CONFIG-SHAPE GATE — hashes the serving-relevant .env knobs + inner
#      scripts + overlay patches + pinned image digest. When the shape CHANGED
#      since the last successful start, stale JIT caches (Triton / TileLang /
#      FlashInfer) are wiped first: kernels captured under the old geometry
#      can silently mis-serve the new one, and a wipe costs one re-JIT while
#      keeping the wrong-shape cache costs silent corruption risk.
#   3. WATCHDOG HANDSHAKE — clears the deliberate-stop sentinel (a prod-start
#      IS a deliberate start), then runs ./start.sh restart.
#
# Usage: local/prod-start.sh [--force] [extra ./start.sh args...]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$SCRIPT_DIR")"
LOGDIR="$REPO/logs"
mkdir -p "$LOGDIR"
SHAPE_FILE="$LOGDIR/.config-shape"

FORCE=0
START_ARGS=()
for a in "$@"; do
    [ "$a" = "--force" ] && { FORCE=1; continue; }
    START_ARGS+=("$a")
done

# shellcheck disable=SC1091
source "$REPO/.env" 2>/dev/null || true
WORKER_USER="${WORKER_USER:-$USER}"
WORKER_IP="${WORKER_IP:-10.0.0.2}"
WORKER_SSH_TARGET="${WORKER_SSH:-${WORKER_USER}@${WORKER_IP}}"
MIN_MEMFREE_KB="${PROD_MIN_MEMFREE_KB:-8388608}"   # 8 GiB crash-review floor

memfree_kb() { awk '/^MemFree:/{print $2}' /proc/meminfo 2>/dev/null || echo 0; }

# --- 1. memory gate ----------------------------------------------------------
if [ "$FORCE" = "0" ]; then
    head_free="$(memfree_kb)"
    worker_free="$(ssh -T -o BatchMode=yes -o ConnectTimeout=15 "$WORKER_SSH_TARGET" \
        "awk '/^MemFree:/{print \$2}' /proc/meminfo 2>/dev/null" 2>/dev/null || echo 0)"
    for side in "head:$head_free" "worker:$worker_free"; do
        node="${side%%:*}"; kb="${side#*:}"
        if [ "${kb:-0}" -lt "$MIN_MEMFREE_KB" ]; then
            echo "prod-start: REFUSED — ${node} MemFree $((kb / 1024 / 1024)) GiB < $((MIN_MEMFREE_KB / 1024 / 1024)) GiB floor (UVM-livelock dice roll). Free page cache (./cache_flusher.sh) or reboot, then retry; --force overrides." >&2
            exit 1
        fi
    done
    echo "prod-start: memory gate OK (head=$((head_free / 1024 / 1024)) GiB, worker=$((worker_free / 1024 / 1024)) GiB)"
else
    echo "prod-start: --force — memory gate skipped"
fi

# --- 2. config-shape hash ----------------------------------------------------
shape_srcs=(
    "$REPO/start.sh"
    "$REPO/.env"
    "$REPO/.glm53-exl3-head.inner.sh"
    "$REPO/.glm53-exl3-worker.inner.sh"
    "$REPO"/overlay/*.py
)
shape="$(cat "${shape_srcs[@]}" 2>/dev/null | sha256sum | awk '{print $1}')"

if [ -f "$SHAPE_FILE" ] && [ "$(cat "$SHAPE_FILE")" != "$shape" ]; then
    echo "prod-start: config shape CHANGED — wiping stale JIT caches (triton/tilelang/flashinfer) on both nodes"
    rm -rf "${CACHE_ROOT:-$HOME/.cache/vllm-glm53-flash}"/triton/* \
           "${CACHE_ROOT:-$HOME/.cache/vllm-glm53-flash}"/tilelang/* \
           "${CACHE_ROOT:-$HOME/.cache/vllm-glm53-flash}"/flashinfer/* 2>/dev/null || true
    ssh -T -o BatchMode=yes -o ConnectTimeout=15 "$WORKER_SSH_TARGET" \
        "rm -rf '${WORKER_VLLM_CACHE:-$HOME/.cache/vllm-glm53-flash}'/triton/* \
                '${WORKER_VLLM_CACHE:-$HOME/.cache/vllm-glm53-flash}'/tilelang/* \
                '${WORKER_VLLM_CACHE:-$HOME/.cache/vllm-glm53-flash}'/flashinfer/* 2>/dev/null || true" \
        || true
elif [ ! -f "$SHAPE_FILE" ]; then
    echo "prod-start: no previous config shape recorded — JIT caches left in place (first run)"
else
    echo "prod-start: config shape unchanged — JIT caches kept"
fi

# --- 3. watchdog handshake + restart -----------------------------------------
rm -f "$LOGDIR/.watchdog-paused"
echo "prod-start: ./start.sh restart ${START_ARGS[*]:-}"
cd "$REPO"
if ./start.sh restart "${START_ARGS[@]+"${START_ARGS[@]}"}"; then
    printf '%s' "$shape" > "$SHAPE_FILE"
    echo "prod-start: fleet healthy — config shape recorded"
else
    echo "prod-start: ./start.sh restart FAILED — shape NOT recorded (next run will wipe JIT caches again)" >&2
    exit 1
fi
