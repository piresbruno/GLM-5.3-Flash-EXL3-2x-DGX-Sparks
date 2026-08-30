#!/usr/bin/env bash
# cache_flusher.sh — keep GB10 page cache small during model load so NVRM can
# allocate the KV slab (NVIDIA KB 5776 remedy). Port of the NVFP4 recipe.
#
# 2026-08-29 D0 validation: the Cached>40 GiB trigger alone never fired on
# this kit's load (loader keeps cache ~28-29 GiB while MemFree dips to
# ~0.9 GiB / MemAvailable 0.35 GiB). Add a MemFree-pressure trigger so the
# flush happens exactly when the UMA is starved, not when the cache is big.
#
# 2026-08-30 R1 freeze hardening: the 14.64 GiB-pin freeze hit AFTER engine
# init (NVRM NV_ERR_NO_MEMORY storm at API bring-up, 05:19->05:33) — the old
# fixed 25-min cap ended inside the danger window and the sidecar's flushing
# was invisible (stdout to /dev/null). Now:
#   - runs until the fleet is SERVING-STABLE: GLM53_HEALTH_URL OK on 5
#     consecutive probes after GLM53_FLUSHER_MIN_STABLE (default 900 s), or
#     hard cap GLM53_FLUSHER_CAP (default 2700 s);
#   - logs every flush + MemFree/Cached sample to GLM53_FLUSHER_LOG.
# Launch as a detached sidecar during weight load: CACHE_FLUSHER=1 ./start.sh
set -u

end=$((SECONDS + ${GLM53_FLUSHER_CAP:-2700}))
min_stable=${GLM53_FLUSHER_MIN_STABLE:-900}
url="${GLM53_HEALTH_URL:-http://127.0.0.1:${GLM53_PORT:-8081}/health}"
log="${GLM53_FLUSHER_LOG:-}"

logline() {
    [ -n "$log" ] || return 0
    printf '%s %s\n' "$(date '+%F %T')" "$*" >> "$log" 2>/dev/null || true
}

logline "flusher up (pid $$, url=$url, min_stable=${min_stable}s, cap=${end}s)"
stable=0
while [ "$SECONDS" -lt "$end" ]; do
    read -r memfree cached < <(awk '/^MemFree:|^Cached:/{print $2}' /proc/meminfo | tr '\n' ' ')
    memfree_gib=$(( ${memfree:-0} / 1048576 ))
    cached_gib=$(( ${cached:-0} / 1048576 ))
    if [ "${cached_gib:-0}" -gt 40 ] || [ "${memfree_gib:-0}" -lt 8 ]; then
        sync
        echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null 2>&1 \
            || echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null 2>&1
        logline "flush: memfree=${memfree_gib}GiB cached=${cached_gib}GiB"
    fi
    # Stability exit: serving healthy on 5 consecutive probes past the load
    # window. On the worker this probes the HEAD's API (GLM53_HEALTH_URL) —
    # both ranks finish load together (TP rendezvous), so head-healthy means
    # the worker's danger window is closing too.
    if [ "$SECONDS" -ge "$min_stable" ] && curl -sf -m 5 "$url" >/dev/null 2>&1; then
        stable=$((stable + 1))
        if [ "$stable" -ge 5 ]; then
            logline "stable: health OK x5 at ${SECONDS}s (memfree=${memfree_gib}GiB cached=${cached_gib}GiB) — flusher exiting"
            exit 0
        fi
    else
        stable=0
    fi
    sleep 5
done
logline "cap: ${SECONDS}s reached (memfree=${memfree_gib:-?}GiB cached=${cached_gib:-?}GiB) — flusher exiting"
