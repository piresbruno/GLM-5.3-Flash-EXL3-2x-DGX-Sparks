# D1-2048 (MAX_NUM_BATCHED_TOKENS 1024 → 2048) + adoption

| Metric | Baseline (512) | D1-1024 | D1-2048 | Gate |
|---|---:|---:|---:|---|
| 100k cold TTFT | 275.7 s | 224.4 s (−18.6%) | **204.7 s (−25.7%)** | PASS |
| 100k decode tok/s | 27.0 | 35.5 | **43.6** | — |
| Prefill effective | ~363 tok/s | ~445 | ~489 | — |
| Structured tok/s | 66.28 | 66.27 | 65.81 (−0.7%) | PASS |
| Prose tok/s | 26.60 | 27.35 | 27.61 (+3.8%) | PASS |
| Structured TTFT / ITL | 0.456 s / 15.1 ms | 0.464 / 15.1 | 0.457 / 15.2 | — |
| KV pool @1M | 1,262k (1.26×) | 1,102k (1.10×) | 1,012k (**1.01×**) | fits, but thin |
| Saturation floors | — | head 0.65 / wrk 5.17 | head 0.53 / wrk 4.32 GiB | WARN <5 GiB |

## Adoption decision (operator)

**Adopt MAX_NUM_BATCHED_TOKENS=2048 AND lower MAX_MODEL_LEN 1M → 900k.**

- The pool allocation is unchanged (leftover UMA at util 0.85); what 900k buys
  is margin on the fit check and concurrency: 1,012,077 / 900,000 = **1.12×**
  vs 1.01× at 1M — boots stop depending on the lottery, and the
  floors-age-1.5-2-GiB/day drift no longer threatens the fit.
- TTFT gain is preserved: prefill speed is driven by MNBT, not max-len.
- 4096 skipped: predicted pool ~950-960k tokens (trend −160k/−90k per
  doubling, saturating) fits 900k at ~1.06× but the 1024→2048 step already
  showed diminishing TTFT returns (−18.6% → −9.5% relative); another boot for
  ~10 s on a 100k prompt is not worth the margin.
- Memfloor warning (head 0.53 GiB binding under saturation) is the standing
  state of this kit at util 0.85 — the D2 rule now reads: do not raise
  KV_CACHE_MEMORY/GPU_MEM_UTIL, and the cache-flusher MemFree trigger is what
  keeps boots alive. Recorded in DECISION.md.
