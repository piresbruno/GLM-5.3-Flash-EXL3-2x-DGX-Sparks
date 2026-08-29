#!/usr/bin/env bash
# cache_flusher.sh — keep GB10 page cache small during model load so NVRM can
# allocate the KV slab (NVIDIA KB 5776 remedy). Port of the NVFP4 recipe.
# Runs for 25 min max, flushes whenever Cached > 40 GiB.
# Launch as a detached sidecar during weight load: CACHE_FLUSHER=1 ./start.sh restart
#
# 2026-08-29 D0 validation: the Cached>40 GiB trigger alone never fired on
# this kit's load (loader keeps cache ~28-29 GiB while MemFree dips to
# ~0.9 GiB / MemAvailable 0.35 GiB). Add a MemFree-pressure trigger so the
# flush happens exactly when the UMA is starved, not when the cache is big.
end=$((SECONDS+1500))
while [ $SECONDS -lt $end ]; do
  read -r memfree cached < <(awk '/^MemFree:|^Cached:/{print $2}' /proc/meminfo | tr '\n' ' ')
  memfree_gib=$(( ${memfree:-0} / 1048576 ))
  cached_gib=$(( ${cached:-0} / 1048576 ))
  if [ "${cached_gib:-0}" -gt 40 ] || [ "${memfree_gib:-0}" -lt 8 ]; then
    sync
    echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null 2>&1 \
      || echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null 2>&1
  fi
  sleep 5
done
