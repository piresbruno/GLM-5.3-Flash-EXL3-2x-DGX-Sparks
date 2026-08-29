#!/usr/bin/env bash
# check-updates.sh — image/driver drift guard (R1 ops kit, ported from the
# Reederey87 production kit; see NOTICE). Timer-friendly one-shot.
#
# Checks, in order:
#   1. DIGEST PIN: if .env pins IMAGE to <repo>@sha256:..., verify the local
#      image still matches that digest (a mismatched local store = someone
#      re-tagged or pruned — fix before the next restart silently boots a
#      different engine).
#   2. REGISTRY DRIFT (optional, PULL=1 kits): with a tag-pinned IMAGE and
#      docker manifest inspect available, compare the registry digest with
#      the local RepoDigest and warn when the registry has moved (a pull
#      would silently upgrade).
#   3. DRIVER BRANCH: 590.x deadlocks CUDAGraph capture on GB10 (R1 Phase-0
#      evidence) — warn loudly if either node drifts onto 590.x.
#
# Usage: local/check-updates.sh [--strict]
set -euo pipefail
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SELF_DIR}/.." && pwd)"
LOGDIR="$REPO/logs"
LOGFILE="$LOGDIR/check-updates.log"
mkdir -p "$LOGDIR"
STRICT=0
[ "${1:-}" = "--strict" ] && STRICT=1

now() { date '+%F %T'; }
rc=0

# .env values may carry inline comments (WORKER_USER=x  # note) — strip them.
env_get() { grep -m1 -E "^[[:space:]]*${1}=" "$REPO/.env" 2>/dev/null | cut -d= -f2- | sed 's/[[:space:]]*#.*//; s/[[:space:]]*$//' || true; }

IMAGE="$(env_get IMAGE)"
IMAGE="${IMAGE:-ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3}"

# --- 1. digest pin ----------------------------------------------------------
case "$IMAGE" in
    *@sha256:*)
        pin="${IMAGE#*@}"
        local_digests="$(docker image inspect -f '{{join .RepoDigests " "}}' "$IMAGE" 2>/dev/null || true)"
        if [ -z "$local_digests" ]; then
            echo "$(now) ALERT: local image store has NO entry for pinned $IMAGE — the next boot fails or silently re-pulls" | tee -a "$LOGFILE" >&2
            rc=1
        elif ! printf '%s' "$local_digests" | grep -q "$pin"; then
            echo "$(now) ALERT: local RepoDigests ($local_digests) do not contain the pin $pin" | tee -a "$LOGFILE" >&2
            rc=1
        else
            echo "$(now) digest pin OK: $pin" >> "$LOGFILE"
        fi
        ;;
    *)
        # --- 2. registry drift (tag-pinned images only) ----------------------
        if command -v docker >/dev/null 2>&1 && docker manifest inspect "$IMAGE" >/dev/null 2>&1; then
            reg_digest="$(docker manifest inspect --verbose "$IMAGE" 2>/dev/null | grep -m1 -oP '"digest":\s*"\Ksha256:[0-9a-f]+' || true)"
            local_digest="$(docker image inspect -f '{{if .RepoDigests}}{{index .RepoDigests 0}}{{end}}' "$IMAGE" 2>/dev/null | grep -oP 'sha256:[0-9a-f]+' || true)"
            if [ -n "$reg_digest" ] && [ -n "$local_digest" ] && [ "$reg_digest" != "$local_digest" ]; then
                echo "$(now) WARN: registry $IMAGE moved (${local_digest##sha256:} -> ${reg_digest##sha256:}) — a plain 'docker pull' would silently upgrade; pin the digest in .env" | tee -a "$LOGFILE" >&2
                [ "$STRICT" = "1" ] && rc=1
            else
                echo "$(now) registry matches local for $IMAGE" >> "$LOGFILE"
            fi
        fi
        ;;
esac

# --- 3. driver branch --------------------------------------------------------
check_driver() {  # check_driver <label> <host-cmd...>
    local label="$1"; shift
    local drv
    drv="$( "$@" 2>/dev/null | head -1 | tr -d '[:space:]' || true)"
    case "$drv" in
        590*) echo "$(now) ALERT: ${label} driver ${drv} is 590.x — 590.x deadlocks CUDAGraph capture on GB10 (R1 Phase 0). Plan a downgrade to 580.x." | tee -a "$LOGFILE" >&2; rc=1 ;;
        580*) echo "$(now) driver OK on ${label}: ${drv}" >> "$LOGFILE" ;;
        "") echo "$(now) WARN: driver version unreadable on ${label}" >> "$LOGFILE" ;;
        *) echo "$(now) INFO: ${label} driver ${drv} (not 580.x — R1 gate expects 580.x)" >> "$LOGFILE" ;;
    esac
}

check_driver "head" nvidia-smi --query-gpu=driver_version --format=csv,noheader

WORKER_SSH_TARGET="$(env_get WORKER_SSH)"
if [ -z "$WORKER_SSH_TARGET" ]; then
    WU="$(env_get WORKER_USER)"
    WI="$(env_get WORKER_IP)"
    WU="${WU:-$USER}"
    [ -n "$WI" ] && WORKER_SSH_TARGET="${WU}@${WI}"
fi
if [ -n "$WORKER_SSH_TARGET" ]; then
    if ssh -T -o BatchMode=yes -o ConnectTimeout=15 "$WORKER_SSH_TARGET" true 2>/dev/null; then
        drv="$(ssh -T -o BatchMode=yes -o ConnectTimeout=15 "$WORKER_SSH_TARGET" \
            "nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1" 2>/dev/null | tr -d '[:space:]' || true)"
        case "$drv" in
            590*) echo "$(now) ALERT: worker driver ${drv} is 590.x — 590.x deadlocks CUDAGraph capture on GB10 (R1 Phase 0). Plan a downgrade to 580.x." | tee -a "$LOGFILE" >&2; rc=1 ;;
            580*) echo "$(now) driver OK on worker: ${drv}" >> "$LOGFILE" ;;
            "") echo "$(now) WARN: worker driver version unreadable" >> "$LOGFILE" ;;
            *) echo "$(now) INFO: worker driver ${drv} (not 580.x — R1 gate expects 580.x)" >> "$LOGFILE" ;;
        esac
    else
        echo "$(now) WARN: worker ${WORKER_SSH_TARGET} unreachable — driver check skipped" >> "$LOGFILE"
    fi
fi

exit "$rc"
