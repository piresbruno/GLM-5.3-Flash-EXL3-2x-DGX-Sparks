#!/usr/bin/env bash
# cache_flusher.sh — keep GB10 page cache small during model load so NVRM can
# allocate the KV slab (NVIDIA KB 5776 remedy). Port of the NVFP4 recipe.
# Runs for 25 min max, flushes whenever Cached > 40 GiB. Launch as a detached
# sidecar during weight load: CACHE_FLUSHER=1 ./start.sh restart
end=$((SECONDS+1500))
while [ $SECONDS -lt $end ]; do
  c=$(awk '/^Cached:/{print int($2/1048576)}' /proc/meminfo)
  if [ "${c:-0}" -gt 40 ]; then
    sync
    echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null 2>&1 \
      || echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null 2>&1
  fi
  sleep 5
done
