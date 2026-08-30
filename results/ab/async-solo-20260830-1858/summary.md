# Phase 3.5 follow-ups — async-solo + NCCL QPS4 — 2026-08-30

Two single-serve arms on top of the adopted Phase 3.5 config
(`GLM53_MIXED_PREFILL_CHUNK=1024`, R1 bundle, auto pool). Reference =
`mixed-chunk-1024-20260830-1804` (same day, same image digest, clocks
**2190 MHz both ranks verified identical on every serve** — operator
underclock).

**Measurement caveat (recorded):** prose ×1 drifted −9% across both evening
serves (25.01/25.18 vs 27.69 @ 18:04; baseline 26.62 @ 17:41) while
structured held (−1.5/−2.2%) — evening thermal drift under the underclock
(54–62 °C). Cross-serve-time prose comparisons below ±5% are unreliable
today; same-session back-to-back A/B (Phase 3.5) remains valid, and its
verdict rested on deltas far beyond this band.

## Arm 1 — async-solo (`ASYNC_SCHEDULING=1`, no DSD) — REJECTED, closes D5

`results/ab/async-solo-20260830-1858/`

| Metric | ref (1024) | async-solo | Δ |
|---|---:|---:|---|
| Structured ×1 | 65.59 | 64.14 | −2.2% (within variance) |
| Prose ×1 | 27.69 | 25.01 | −9.7% (drift caveat) |
| Conc ×1 / ×2 / ×4 agg | 38.0 / 45.2 / 47.5 | 31.0 / 36.2 / 44.4 | **−18% / −20% / −7%** |
| 100k prefill TTFT | 207.3 s | 214.3 s | +3.4% |
| HOL first token | 6.23 s | 8.0 s | worse (gate pass) |
| Cache burst/solo gates | PASS | PASS | — |
| Preemptions | 0 | 0 | — |

**Verdict: REJECTED.** async-ON costs ×1 decode and delivers **no** ×2/×4
aggregate win on top of the mixing adoption — the conc ladder loses at every
level. This closes the D5 follow-up question: the DSD arm's ×1 regression
was indeed the async cost, and async no longer pays after Phase 3.5 (its
concurrency benefit is superseded by mixed prefill/decode chunking). DSD
stays dormant permanently — every DSD variant requires async.

## Arm 2 — `NCCL_IB_QPS_PER_CONNECTION=4` — REJECTED (recorded)

`results/ab/nccl-qps4-20260830-1932/` (forwarded to both ranks via the
`d63846b` wiring; default stays unset = NCCL default)

| Metric | ref (1024) | qps4 | Δ |
|---|---:|---:|---|
| **100k prefill TTFT (target)** | 207.3 s | 207.8 s | **unchanged** |
| HOL first token | 6.23 s | 6.06 s | unchanged |
| Structured ×1 | 65.59 | 64.58 | −1.5% (variance) |
| Prose ×1 | 27.69 | 25.18 | drift caveat |
| Conc ×1 / ×2 / ×4 agg | 38.0 / 45.2 / 47.5 | 36.5 / 37.3 / 50.3 | mixed, state-sensitive |
| Cache gates / preemptions | PASS / 0 | PASS / 0 | — |

**Verdict: REJECTED (recorded).** The P1-1 expectation (+9% at ≥256 MB
collective payloads) did not transfer to this kit: the targeted prefill
metric is byte-for-byte unchanged (2-node CX7, 3584-token chunks evidently
below the QPC payoff threshold). Wiring stays (harmless when unset) so the
arm is re-runnable after a CX7 reseat/soak or an MNBT change.

## Fleet state after the campaign

Restored to the adopted serving config (mixed=1024, async 0, no QPS override)
and left up.
