# Baseline — 2026-08-29 (arm: base)

**Base:** `1df71c1` (padded slot-share + hybrid-APC + 1M default) on branch
`ab/execute-2026-08-29` @ C0 tooling. Image repo-digests match head/worker.
**Boot receipts:** pool 1,371,727 tokens; max concurrency 1.37× @1M (util 0.85);
slot-share line on both ranks; APC guard PASS; spec-decode sanity clear.

## Text decode (bench_decode, temp 0, thinking off, median of 5 × 400)

| phase | tok/s (median) | runs |
|---|---:|---|
| structured (count 1→200) | **66.8** | 65.1–67.1 |
| prose (hash-map) | **25.6** | 25.2–28.1 |

## Concurrency sweep (bench_concurrency, 2 waves/level, 400 tokens, salted)

| c | temp 0: agg / decode / TTFT / accept | temp 1.0: agg / decode / TTFT / accept |
|---|---|---|
| 1 | 35.8 / 37.0 / 374ms / 0.541 | 28.2 / 29.8 / 322ms / 0.391 |
| 2 | 43.9 / 23.3 / 456ms / 0.441 | 30.6 / 23.4 / 4058ms / 0.362 |
| 4 | 42.7 / 15.5 / 6728ms / 0.304 | 44.1 / 12.2 / 467ms / 0.273 |

Temp 0 vs 1.0: **+27% aggregate at c1** (0.541 vs 0.391 acceptance) — the sampler-tax
lever confirmed on this kit. TTFT spikes at c2/c4 = `GLM53_MIXED_PREFILL_CHUNK=skip`
floor delaying new prefills behind decodes (expected; protects decode).

## Vision (bench_vision, image+text, temp 0)

- Correctness **3/3**, no NaN. **Spec-decode exempt** (drafter is text-only).
- Warm: TTFT ~0.75–0.80s, decode 16.4–25.2 tok/s.
- Cold first MM request after boot: TTFT 11.2s (JIT) — one-time.

## Metrics

- Per-position acceptance: 0.312 / 0.223 / 0.159 / 0.113 / 0.083 / 0.063 / 0.047
  → positions 5–6 rarely accept (**supports C7** k=4/5 arm).
- Preemptions: **0**. vllm#53030 flat-1.00 trap: **clear**.
- Cumulative APC hit rate 7168/25825 ≈ 28% (dominated by salted bench traffic).

## Notes for arms

- Compare every arm against **this file's medians**, at matching temperature.
- The c2/c4 TTFT spikes are part of baseline behavior (skip policy), not a defect.
