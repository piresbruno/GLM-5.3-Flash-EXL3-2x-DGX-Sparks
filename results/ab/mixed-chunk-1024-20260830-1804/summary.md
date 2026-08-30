# Phase 3.5 — `GLM53_MIXED_PREFILL_CHUNK=1024` vs `skip` (issue #43) — 2026-08-30

Branch `checkpoint-d1-baseline`, image `…@sha256:9bb1557a…` (identical both ranks),
R1 bundle geometry fixed (MNBT 3584, LPT 1792, async 0, retention 0, 600k window,
auto pool). **GPU clocks identical both serves: SM 2190 MHz (operator underclock,
max 3003) — valid A/B; absolute values are ~10% below the morning R1 records
because of the underclock, which is why the baseline was re-served fresh
(AB-PLAN rule 6).** Single serve per arm per the Phase 3.5 runbook.

## Verdict table

| Metric | Gate | base (skip) `mixed-chunk-base-20260830-1741` | arm (1024) `mixed-chunk-1024-20260830-1804` | Verdict |
|---|---|---:|---:|---|
| Structured ×1 (tok/s, temp 0) | ±3% | 63.50 | 65.59 (**+3.3%**) | PASS |
| Prose ×1 (tok/s) | ±3% | 26.62 | 27.69 (**+4.0%**) | PASS |
| Conc ×2 aggregate tok/s | ±3% | 26.7 | **45.2 (+69%)** | PASS |
| Conc ×4 aggregate tok/s | ±3% | 34.5 | **47.5 (+38%)** | PASS |
| Conc ×2 TTFT | ≥10% better | 7046 ms | **428 ms (−94%)** | PASS |
| Conc ×4 TTFT | ≥10% better | 12233 ms | **528 ms (−96%)** | PASS |
| 100k solo prefill TTFT | record | 207.7 s (~482 tok/s) | 207.3 s | unchanged — solo prefill stays solo, as designed |
| HOL first token (1.2k behind 240k) | ≤30 s | 5.99 s | 6.23 s | PASS |
| Cache burst rounds 2–3 | ≥90% | 98.6% | 98.6% | PASS |
| Cache solo 110k replay | ≥93% | 96.9% | 96.9% | PASS |
| Acceptance (structured) | ±2 pt | 0.9588 | 0.9588 | PASS |
| Preemptions | 0 | 0 | 0 | PASS |
| serving-probe / acceptance | pass | 6/6, all | 6/6, all | PASS |

## Mechanism + cost (recorded honestly)

Under `skip`, a peer **cold prefill never mixes into decode steps**, so a
request arriving while others decode waits for solo scheduler slots: TTFT
7–12 s at ×2/×4. With `1024`, up to one solo-chunk worth of prefill tokens
rides in each decode step: TTFT collapses to ~0.5 s and aggregate throughput
rises 38–69% — this is the exact complaint in issue #43 ("more than 1 session
clogs everything").

The trade: **per-stream decode rate during contention drops** (×2 decode
tok/s 27.3→24.8, ×4 16.5→12.6) because decode steps now carry prefill chunks
(the TPOT-jitter concern behind issue #6). Every request still finishes
sooner (round wall 27.0→17.1 s at ×2, 46.4→33.7 s at ×4), solo ×1 decode
improves, and issue #43's interactive-queueing symptom is gone. Solo prefill
is untouched (mixing only applies when peers are decoding).

## Decision

**ADOPTED** per the Phase 3.5 gate (≥10% TTFT improvement, decode within ±3%,
acceptance ±2 pt, zero preemptions). `.env`/`.env.example` default flips to
`1024` in the adoption commit; README knob row updated. Fleet left serving
with the adopted value.
