#!/usr/bin/env bash
# 1 Hz MemAvailable sampler — the memory-floor methodology behind KV budget
# decisions on GB10 unified memory. Ported from
# Entrpi/glm-5.3-flash-exl3-2x-spark tools/memlog.sh (MIT; credit to Entrpi),
# extended with a node label for multi-box runs.
#
# Run on EACH box while driving load (saturation sweep + one long prefill),
# then read the minimum:
#   ./memlog.sh /tmp/memlog-head &      # start before the load
#   sort -n /tmp/memlog-head | head -1  # floor in KiB after the run
#
# Why: explicit KV budgets (--kv-cache-memory) bypass vLLM's profiling
# reserve, and GB10 unified memory fails as a swap wedge, not a graceful OOM.
# Keep the floor above ~5 GiB on the memory-binding box before raising
# KV_CACHE_MEMORY or GPU_MEM_UTIL. Floors also age ~1.5-2 GiB per day of
# workload — fresh-boot floors are optimistic (this kit's crash review:
# 0.87 froze both nodes; a 17.7 GiB pin crashed twice).
#
# tools/memfloor.sh wraps this for both nodes and writes the floor artifact.
OUT="${1:-/tmp/memlog}"
LABEL="${2:-$(hostname)}"
: > "$OUT"
while true; do
  awk -v l="$LABEL" '/MemAvailable/ {print l" "$2}' /proc/meminfo >> "$OUT"
  sleep 1
done
