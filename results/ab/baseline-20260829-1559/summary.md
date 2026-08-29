# Baseline 2026-08-29 15:59 (D1 comparison target)

Config: 0.85 / MAX_NUM_BATCHED_TOKENS=512 / CG_ESTIMATE=0 / no KV pin / DFlash2 k=7 /
fp8 KV / 1M / CACHE_FLUSHER=1 (old Cached>40 GiB trigger) — the crash-review
envelope. Boot #1 was the cache-flusher-validated healthy boot.

## Boot

| | |
|---|---|
| max_num_scheduled_tokens | 512 (boot log; env honored — the 1849 line is informational) |
| KV pool | **1,262,295 tokens (1.26× at 1M)**, 11.79 GiB available |
| Load window floors | head MemAvailable **0.35 GiB**, worker 4.78 GiB — survived, but razor-thin |
| Flusher | head sidecar ran but **never fired** (Cached 28-29 GiB < 40 while MemFree hit 0.94 GiB); worker sidecar **silently failed** (script never shipped) — both fixed in f249a23, re-validated on D1-1024 |
| Preemptions / NaN | 0 / 0 |

## Decode (repo protocol, temp 0, thinking off, 5×400 median)

| | tok/s | accept/step | TTFT |
|---|---:|---:|---:|
| Structured | **66.28** | 0.959 (6.71/7) | 0.456 s |
| Prose | **26.60** | 0.322 (2.25/7) | 0.455 s |

## Long prompt (D1 gate)

| | |
|---|---|
| 100k cold TTFT (3 rounds) | **275.7 s** (~363 tok/s prefill) |
| decode @100k KV | 27.0 tok/s |

## Hardware-limit reading (operator question)

Decode is engine-step-bound (~116 ms/step; Entrpi's fork measures the same →
hardware, not stack). Aggregate peaks ×2–×3 (~90 tok/s; ×4+ adds nothing).
Prefill is ~4× below the hardware (Entrpi: ~1,400 tok/s with 8192 chunks) —
D1 attacks exactly this. Memory is at the edge; the D0 flusher fix is the
margin play.

## Files

structured.json, prose.json, longprompt-100k.json (raw), arm.json.
