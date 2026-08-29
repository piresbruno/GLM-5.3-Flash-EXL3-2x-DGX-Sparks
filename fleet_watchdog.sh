#!/usr/bin/env bash
# fleet_watchdog.sh — auto-recovery for the GLM-5.3-Flash EXL3 2-node fleet.
# Port of the NVFP4 recipe watchdog, adapted to this repo (2 ranks, :8081,
# containers glm53-exl3-head + glm53-exl3-worker, launch via ./start.sh restart).
#
# Runs on the HEAD node (rank 0), OUTSIDE the container. Probes /health —
# NOT /v1/models, which returns 200 even with a dead engine (/health 503s on
# EngineDeadError). R1 upgrade (Reederey87 ops kit): the three failure
# classes are distinguished —
#   CRASH  : head container not RUNNING (docker inspect) -> recover
#            immediately, no threshold wait.
#   WEDGE  : container running but /health failing -> recover after
#            FAIL_THRESHOLD consecutive misses (unchanged behavior).
#   DELIBERATE STOP : logs/.watchdog-paused sentinel (written by
#            ./start.sh stop, cleared by any launch) -> never recover while
#            it exists; the operator owns the fleet until they start it again.
# Two more R1 additions:
#   - RECOVER_BACKOFF_S (default 900): never attempt recovery more often
#     than this, so a boot-time failure (weight download, refused config)
#     cannot thrash into a restart loop.
#   - WATCHDOG_LIVENESS=1 adds an engine-wedge probe: /health can return 200
#     while the decode loop is stuck; a max_tokens=1 completion must answer
#     within LIVENESS_TIMEOUT_S (default 60) or the miss counts toward the
#     threshold. Default off (it adds load during a memory-critical window).
#
# After the failure decision it:
#   1. snapshots docker logs of both ranks (NVRM failures surface minutes late;
#      teardown destroys the evidence),
#   2. tears down BOTH ranks (a worker must never rendezvous with a dying head),
#   3. runs the GB10 memory ritual on both nodes (drop_caches, swappiness=0),
#   4. relaunches via ./start.sh restart (ships patches + starts worker, then head;
#      the launch clears the pause sentinel),
#   5. waits for /health (start.sh polls it internally).
#
# Single-instance via flock. Logs to <repo>/logs/fleet_watchdog.log.
# Usage: nohup ./fleet_watchdog.sh >/dev/null 2>&1 &
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/.env" 2>/dev/null || true

PORT="${PORT:-8081}"
HEALTH_URL="http://127.0.0.1:${PORT}/health"
CHECK_INTERVAL=60
FAIL_THRESHOLD=3
CURL_TIMEOUT=15
POST_TEARDOWN_SLEEP=10
RESTART_TIMEOUT=3600          # start.sh waits for ready internally; outer guard
RECOVER_BACKOFF_S=900         # min spacing between recovery attempts
WATCHDOG_LIVENESS=0           # 1 = also probe decode liveness (engine wedge)
LIVENESS_TIMEOUT_S=60
HEAD_CONTAINER="${HEAD_CONTAINER:-glm53-exl3-head}"
WORKER_CONTAINER="${WORKER_CONTAINER:-glm53-exl3-worker}"
WORKER_TARGET="${WORKER_SSH:-${WORKER_USER:-$USER}@${WORKER_IP:-}}"
LOCKFILE="$HOME/.fleet_watchdog.lock"
LOGDIR="$SCRIPT_DIR/logs"
LOGFILE="$LOGDIR/fleet_watchdog.log"
mkdir -p "$LOGDIR"

log() { printf '%s %s\n' "$(date '+%F %T')" "$*" >> "$LOGFILE"; }

BASE_URL_V1="http://127.0.0.1:${PORT}/v1"

healthy() { curl -sf -m "$CURL_TIMEOUT" -o /dev/null "$HEALTH_URL"; }

head_running() {  # true when the head container exists AND is running
  [ "$(docker inspect -f '{{.State.Running}}' "$HEAD_CONTAINER" 2>/dev/null || true)" = "true" ]
}

paused() { [ -f "$LOGDIR/.watchdog-paused" ]; }

liveness_ok() {  # engine-wedge probe: /health can 200 on a stuck decode loop
  curl -sf -m "$LIVENESS_TIMEOUT_S" "$BASE_URL_V1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"${SERVED_MODEL_NAME:-GLM-5.3-Flash-EXL3}\",\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}],\"max_tokens\":1,\"temperature\":0,\"chat_template_kwargs\":{\"enable_thinking\":false}}" \
    -o /dev/null
}

snapshot_logs() {  # evidence BEFORE teardown
  local stamp; stamp="$(date +%Y%m%d-%H%M%S)"
  log "snapshotting container logs (evidence before teardown)"
  docker logs "$HEAD_CONTAINER" > "$LOGDIR/watchdog-snapshot-head-$stamp.log" 2>&1 || true
  if [ -n "${WORKER_IP:-}" ]; then
    ssh -T -o BatchMode=yes -o ConnectTimeout=15 "$WORKER_TARGET" \
      "docker logs $WORKER_CONTAINER" \
      > "$LOGDIR/watchdog-snapshot-worker-$stamp.log" 2>&1 || true
  fi
}

teardown() {
  log "teardown: docker rm -f $HEAD_CONTAINER (local)"
  docker rm -f "$HEAD_CONTAINER" >/dev/null 2>&1 || true
  log "teardown: docker rm -f $WORKER_CONTAINER ($WORKER_TARGET)"
  if [ -n "${WORKER_IP:-}" ]; then
    ssh -T -o BatchMode=yes -o ConnectTimeout=15 "$WORKER_TARGET" \
      "docker rm -f $WORKER_CONTAINER" >/dev/null 2>&1 || true
  fi
}

mem_ritual() {  # GB10 NVRM allocator hygiene; WARN-only. $1 = "local"|"remote"
  local cmd='sync
    echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null 2>&1 || echo "WARN: drop_caches failed"
    echo 1 | sudo -n tee /proc/sys/vm/compact_memory >/dev/null 2>&1 || echo "WARN: compact_memory failed"
    sysctl -w vm.swappiness=0 >/dev/null 2>&1 || echo "WARN: swappiness failed"'
  if [ "${1:-local}" = "remote" ]; then
    if [ -n "${WORKER_IP:-}" ]; then
      ssh -T -o BatchMode=yes -o ConnectTimeout=15 "$WORKER_TARGET" "$cmd" >> "$LOGFILE" 2>&1 || true
    fi
  else
    bash -c "$cmd" >> "$LOGFILE" 2>&1 || true
  fi
}

recover() {
  log "=== RECOVERY START: $FAIL_THRESHOLD consecutive health failures ==="
  snapshot_logs
  teardown
  sleep "$POST_TEARDOWN_SLEEP"
  log "memory ritual: head"
  mem_ritual local
  log "memory ritual: worker"
  mem_ritual remote
  log "relaunching fleet via ./start.sh restart (guard ${RESTART_TIMEOUT}s) ..."
  if command -v timeout >/dev/null 2>&1; then
    (cd "$SCRIPT_DIR" && timeout "$RESTART_TIMEOUT" ./start.sh restart >> "$LOGFILE" 2>&1)
  else
    (cd "$SCRIPT_DIR" && ./start.sh restart >> "$LOGFILE" 2>&1)
  fi
  local rc=$?
  if [ "$rc" -ne 0 ]; then
    log "ERROR: ./start.sh restart exited rc=$rc; probing continues"
    return 1
  fi
  log "=== RECOVERY COMPLETE ==="
  return 0
}

### ---- main ---------------------------------------------------------------
exec 9>"$LOCKFILE"
if ! flock -n 9; then
  echo "fleet_watchdog already running (lock: $LOCKFILE)" >&2
  exit 1
fi
log "watchdog started (pid $$, interval ${CHECK_INTERVAL}s, threshold $FAIL_THRESHOLD, $HEALTH_URL, liveness=${WATCHDOG_LIVENESS}, backoff=${RECOVER_BACKOFF_S}s)"

fails=0
last_recover=0
was_paused=0
while true; do
  # Deliberate stop: ./start.sh stop writes the pause sentinel; a launch
  # clears it. While it exists the operator owns the fleet — never recover.
  if paused; then
    if [ "$was_paused" = "0" ]; then
      log "deliberate stop detected ($LOGDIR/.watchdog-paused) — recovery suspended until the next launch"
      was_paused=1
    fi
    fails=0
    sleep "$CHECK_INTERVAL"
    continue
  fi
  if [ "$was_paused" = "1" ]; then
    log "pause sentinel cleared — watchdog resumed"
    was_paused=0
    fails=0
  fi

  # Memory-pressure alarm: the UVM livelock's leading indicator is MemFree
  # collapsing (driver allocates from MemFree, not MemAvailable) minutes before
  # the shm_broadcast hang. Warn loudly when host free drops below the floor.
  memfree_kb=$(awk '/^MemFree:/{print $2}' /proc/meminfo 2>/dev/null)
  if [ -n "${memfree_kb:-}" ] && [ "${memfree_kb:-0}" -lt "${MEMFREE_ALARM_KB:-8388608}" ]; then  # <8 GiB
    log "MEMORY ALARM: host MemFree=${memfree_kb} kB < ${MEMFREE_ALARM_KB:-8388608} kB floor — UVM livelock risk (crash review 2026-08-29). Consider draining concurrent long-context load."
  fi

  # CRASH: container not running recovers immediately (no threshold wait).
  if ! head_running; then
    log "CRASH: $HEAD_CONTAINER not running (state=$(docker inspect -f '{{.State.Status}}' "$HEAD_CONTAINER" 2>/dev/null || echo absent)) — recovering now"
    if [ "$SECONDS" -lt $((last_recover + RECOVER_BACKOFF_S)) ]; then
      log "recovery backoff: last attempt $((SECONDS - last_recover))s ago (< ${RECOVER_BACKOFF_S}s) — waiting"
    else
      recover || log "recovery attempt failed; probing continues"
      last_recover=$SECONDS
    fi
    fails=0
    sleep "$CHECK_INTERVAL"
    continue
  fi

  if healthy; then
    if [ "$fails" -gt 0 ]; then log "health OK again after $fails failure(s)"; fi
    fails=0
  else
    fails=$((fails + 1))
    log "health FAIL ($fails/$FAIL_THRESHOLD): $HEALTH_URL"
    # WEDGE probe (opt-in): /health can 200 while decode is stuck.
    if [ "${WATCHDOG_LIVENESS:-0}" = "1" ] && ! liveness_ok; then
      fails=$((fails + 1))
      log "LIVENESS FAIL: no completion in ${LIVENESS_TIMEOUT_S}s ($fails/$FAIL_THRESHOLD)"
    fi
    if [ "$fails" -ge "$FAIL_THRESHOLD" ]; then
      if [ "$SECONDS" -lt $((last_recover + RECOVER_BACKOFF_S)) ]; then
        log "recovery backoff: last attempt $((SECONDS - last_recover))s ago (< ${RECOVER_BACKOFF_S}s) — waiting"
      else
        recover || log "recovery attempt failed; probing continues"
        last_recover=$SECONDS
      fi
      fails=0
    fi
  fi
  sleep "$CHECK_INTERVAL"
done
